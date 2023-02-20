import itertools
import random
import time
import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
from src.utils.decode import ctdet_decode, mvdet_decode
from src.utils.nms import nms
from src.utils.meters import AverageMeter
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvselect import aggregate_feat, get_eps_thres


class BaseTrainer(object):
    def __init__(self, model, logdir, args, ):
        self.model = model
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def rollout(self, step, state, tgt, eps_thres):
        feat, init_prob, keep_cams = state
        overall_feat, (log_prob, value, action, entropy) = \
            self.model.select_module(feat, init_prob, keep_cams, eps_thres)
        task_loss, reward = self.task_loss_reward(overall_feat, tgt, step)
        # reward += entropy.detach() * self.args.beta_entropy
        done = torch.ones_like(reward) * (step == self.args.steps - 1)
        return task_loss, reward, done, (log_prob, value, action, entropy)

    def expand_episode(self, feat, keep_cams, tgt, eps_thres, misc):
        B, N, _, _, _ = feat.shape
        loss = []
        action_sum, return_avg = misc
        # consider all cameras as initial one
        for init_cam in range(N):
            log_probs, values, actions, entropies, rewards, task_loss_s = [], [], [], [], [], []
            # get result from using initial camera feature
            init_prob = F.one_hot(torch.tensor(init_cam).repeat(B), num_classes=N).cuda()

            # rollout episode
            for i in range(self.args.steps):
                task_loss, reward, done, (log_prob, value, action, entropy) = \
                    self.rollout(i, (feat, init_prob, keep_cams), tgt, eps_thres)
                # record state & transitions
                log_probs.append(log_prob)
                # values.append(value_i[:, 0])
                values.append((value * action).sum(1))
                actions.append(action)
                entropies.append(entropy)
                rewards.append(reward)
                task_loss_s.append(task_loss)
                # stats
                action_sum += action.detach().sum(dim=0)
                # update the init_prob
                init_prob += action

            log_probs, values, actions, entropies, rewards, task_loss_s = \
                torch.stack(log_probs), torch.stack(values), torch.stack(actions), \
                    torch.stack(entropies), torch.stack(rewards), torch.stack(task_loss_s)
            # calculate returns for each step in episode
            R = torch.zeros([B]).cuda()
            returns = torch.empty([self.args.steps, B]).cuda().float()
            for i in reversed(range(self.args.steps)):
                R = rewards[i] + self.args.gamma * R
                returns[i] = R
            return_avg = returns.mean(1) if return_avg is None else returns.mean(1) * 0.05 + return_avg * 0.95
            # policy & value loss
            value_loss = F.smooth_l1_loss(values, returns)
            # policy_loss = (-log_probs * (returns - values.detach())).mean()
            # task loss
            task_loss = task_loss_s.mean()
            # loss.append(value_loss + policy_loss + task_loss -
            #             entropies.mean() * self.args.beta_entropy * eps_thres)
            loss.append(value_loss + task_loss)
        loss = torch.stack(loss).mean()
        return loss, (action_sum, return_avg, value_loss)

    def task_loss_reward(self, overall_feat, tgt, step):
        raise NotImplementedError


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, args, ):
        super(PerspectiveTrainer, self).__init__(model, logdir, args, )

    def task_loss_reward(self, overall_feat, tgt, step):
        world_heatmap, world_offset = self.model.get_output(overall_feat)
        task_loss = focal_loss(world_heatmap, tgt, reduction='none')
        reward = torch.zeros_like(task_loss).cuda() if step < self.args.steps - 1 else -task_loss.detach()

        # dataloader, frame = misc
        # modas = torch.zeros([B]).cuda()
        # xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset,
        #                    reduce=dataloader.dataset.world_reduce).cpu()
        # grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
        # if dataloader.dataset.base.indexing == 'xy':
        #     positions = grid_xy
        # else:
        #     positions = grid_xy[:, :, [1, 0]]
        # for b in range(B):
        #     ids = scores[b].squeeze() > self.args.cls_thres
        #     pos, s = positions[b, ids], scores[b, ids, 0]
        #     ids, count = nms(pos, s, 20, np.inf)
        #     res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
        #     moda, modp, prec, recall, stats = evaluateDetection_py(res, f'{dataloader.dataset.gt_fname}.txt', frame)
        #     modas[b] = moda
        # reward = torch.zeros([B]).cuda() if step < self.args.steps - 1 else modas / 100

        return task_loss, reward

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses = 0
        t0 = time.time()
        action_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        return_avg = None
        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            feat, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(imgs.cuda(), affine_mats, self.args.down)
            if self.args.steps:
                eps_thres = get_eps_thres(epoch - 1 + batch_idx / len(dataloader), self.args.epochs)
                loss, (action_sum, return_avg, value_loss) = \
                    self.expand_episode(feat, keep_cams, world_gt['heatmap'], eps_thres, (action_sum, return_avg))
            else:
                overall_feat = aggregate_feat(feat, aggregation=self.model.aggregation)
                world_heatmap, world_offset = self.model.get_output(overall_feat)
                loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
                loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'], keep_cams.view(B * N, 1, 1, 1))
                loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])

                w_loss = loss_w_hm + loss_w_off  # + self.args.id_ratio * loss_w_id
                img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.args.id_ratio * loss_img_id
                loss = w_loss + img_loss / N * self.args.alpha
                if self.args.use_mse:
                    loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
                           self.args.alpha * F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) / N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            # logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train epoch: {epoch}, batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.3f}, time: {t_epoch:.1f}')
                if self.args.steps:
                    print(f'value loss: {value_loss:.3f}, eps: {eps_thres:.3f}, return: {return_avg[-1]:.2f}')
                    # print(f'value loss: {value_loss:.3f}, policy loss: {policy_loss:.3f}, '
                    #       f'return: {return_avg[-1]:.2f}, entropy: {entropies.mean():.3f}')
                    print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                                   zip(range(N), F.normalize(action_sum, p=1, dim=0).cpu())))
                pass
        return losses / len(dataloader), None

    def test(self, dataloader, init_cam=None):
        t0 = time.time()
        self.model.eval()
        K = len(init_cam) if init_cam is not None else 1
        losses = torch.zeros([K])
        modas, modps, precs, recalls = torch.zeros([K]), torch.zeros([K]), torch.zeros([K]), torch.zeros([K])
        res_list = [[] for _ in range(K)]
        action_sum = torch.zeros([K, dataloader.dataset.num_cam]).cuda()
        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            world_heatmaps, world_offsets, actions = [], [], []
            with torch.no_grad():
                if self.args.steps == 0 or init_cam is None:
                    (world_heatmap, world_offset), _, (_, _, action, _) = \
                        self.model(imgs.cuda(), affine_mats, self.args.down)
                    world_heatmaps.append(world_heatmap)
                    world_offsets.append(world_offset)
                    actions.append(action)
                else:
                    feat, _ = self.model.get_feat(imgs.cuda(), affine_mats, self.args.down)
                    # K, B, N
                    for k in range(K):
                        overall_feat, (_, _, action, _) = \
                            self.model.do_steps(feat, init_cam[k].repeat([B, 1]), self.args.steps, keep_cams)
                        world_heatmap, world_offset = self.model.get_output(overall_feat)
                        world_heatmaps.append(world_heatmap)
                        world_offsets.append(world_offset)
                        actions.append(action)
            for k in range(K):
                loss = focal_loss(world_heatmaps[k], world_gt['heatmap'])
                if self.args.use_mse:
                    loss = F.mse_loss(world_heatmaps[k], world_gt['heatmap'].cuda())
                if init_cam is not None:
                    # record actions
                    action_sum[k] += torch.cat(actions[k]).sum(dim=0)
                losses[k] += loss.item()

                xys = mvdet_decode(torch.sigmoid(world_heatmaps[k]), world_offsets[k],
                                   reduce=dataloader.dataset.world_reduce).cpu()
                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]
                for b in range(B):
                    ids = scores[b].squeeze() > self.args.cls_thres
                    pos, s = positions[b, ids], scores[b, ids, 0]
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    res_list[k].append(res)

        for k in range(K):
            if init_cam is not None:
                print(f'init camera {init_cam[k].nonzero()[0].item()}: MVSelect')
                idx = action_sum[k].nonzero().cpu()[:, 0]
                print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                               zip(idx, F.normalize(action_sum[k], p=1, dim=0).cpu()[idx])))

            res = torch.cat(res_list[k], dim=0).numpy() if res_list[k] else np.empty([0, 3])
            np.savetxt(f'{self.logdir}/test.txt', res, '%d')
            gt_fname = f'{dataloader.dataset.gt_fname}_{init_cam}.txt' \
                if self.args.eval_init_cam and init_cam is not None else f'{dataloader.dataset.gt_fname}.txt'
            moda, modp, precision, recall = evaluate(f'{self.logdir}/test.txt',
                                                     gt_fname,
                                                     dataloader.dataset.base.__name__,
                                                     dataloader.dataset.frames)
            print(f'Test, loss: {losses[k] / len(dataloader):.6f}, moda: {moda:.1f}%, modp: {modp:.1f}%, '
                  f'prec: {precision:.1f}%, recall: {recall:.1f}%' +
                  ('' if init_cam is not None else f'time: {time.time() - t0:.1f}s'))
            modas[k], modps[k], precs[k], recalls[k] = moda, modp, precision, recall

        if init_cam is not None:
            print('*************************************')
            print(f'MVSelect average moda {modas.mean():.1f}%, modp {modps.mean():.1f}%, '
                  f'prec {precs.mean():.1f}%, recall {recalls.mean():.1f}%, time: {time.time() - t0:.1f}')
            print('*************************************')
        return losses.mean() / len(dataloader), [modas.mean(), modps.mean(), precs.mean(), recalls.mean(), ]

    def test_cam_combination(self, dataloader, combinations):
        self.model.eval()
        t0 = time.time()
        K, N = combinations.shape
        loss_s, result_s = [], [[] for _ in range(K)]
        dataset_lvl_metrics, stats_s = [], []

        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]

            tgt = world_gt['heatmap'].unsqueeze(0).repeat([K, 1, 1, 1, 1])
            # K, B, N
            with torch.no_grad():
                (world_heatmap, world_offset), _ = \
                    self.model.forward_override_combination(imgs.cuda(), affine_mats, self.args.down, combinations)

            # decode
            xys = mvdet_decode(torch.sigmoid(world_heatmap.flatten(0, 1)), world_offset.flatten(0, 1),
                               reduce=dataloader.dataset.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            if dataloader.dataset.base.indexing == 'xy':
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]
            positions, scores = positions.unflatten(0, [K, B]), scores.unflatten(0, [K, B])
            for k in range(K):
                for b in range(B):
                    ids = scores[k, b].squeeze() > self.args.cls_thres
                    pos, s = positions[k, b, ids], scores[k, b, ids, 0]
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    result_s[k].append(res)
            loss = focal_loss(world_heatmap.flatten(0, 1), tgt.flatten(0, 1), reduction='none')
            loss_s.append(loss.unflatten(0, [K, B]).cpu().numpy())
        result_s = [torch.cat(result_s[k], 0) for k in range(K)]
        loss_s = np.concatenate(loss_s, 1)
        # eval
        for k in range(K):
            moda, modp, prec, recall, stats = evaluateDetection_py(result_s[k], f'{dataloader.dataset.gt_fname}.txt',
                                                                   frames=dataloader.dataset.frames)
            dataset_lvl_metrics.append([moda, modp, prec, recall])
            stats_s.append(stats)
        dataset_lvl_metrics = np.array(dataset_lvl_metrics)
        # K, num_frames
        tp_s = np.array([stats_s[k][0] for k in range(K)])
        instance_lvl_strategy = find_instance_lvl_strategy(tp_s, combinations)
        instance_lvl_oracle = [[] for _ in range(N)]
        for init_cam in range(N):
            instance_lvl_oracle_output = []
            for frame_idx, frame in enumerate(dataloader.dataset.frames):
                k = instance_lvl_strategy[init_cam, frame_idx]
                instance_lvl_oracle_output.append(result_s[k][result_s[k][:, 0] == frame])
            instance_lvl_oracle_output = np.concatenate(instance_lvl_oracle_output)
            moda, modp, prec, recall, _ = evaluateDetection_py(instance_lvl_oracle_output,
                                                               f'{dataloader.dataset.gt_fname}.txt',
                                                               frames=dataloader.dataset.frames)
            instance_lvl_oracle[init_cam] = [moda, modp, prec, recall]
        instance_lvl_oracle = np.array(instance_lvl_oracle)
        dataset_lvl_strategy = find_dataset_lvl_strategy(dataset_lvl_metrics[:, 0], combinations)
        dataset_lvl_best_prec = dataset_lvl_metrics[dataset_lvl_strategy, 0]
        print(f'{int(combinations.sum(1).mean())} steps, '
              f'dataset lvl best moda {dataset_lvl_best_prec.mean():.1f}%, '
              f'instance lvl oracle {instance_lvl_oracle.mean(0)[0]:.1f}%, time: {time.time() - t0:.1f}s')
        return loss_s.mean(1), dataset_lvl_metrics, instance_lvl_oracle


def find_instance_lvl_strategy(rewards, combinations):
    K, L = rewards.shape
    K, N = combinations.shape
    strategy = np.zeros([N, L], dtype=int)
    for init_cam in range(N):
        indices = np.nonzero(combinations[:, init_cam] == 1)[0]
        best_idx = np.argmax(rewards[indices], axis=0)
        strategy[init_cam] = indices[best_idx]
    return strategy


def find_dataset_lvl_strategy(rewards, combinations):
    K = rewards.shape
    K, N = combinations.shape
    strategy = np.zeros([N], dtype=int)
    for init_cam in range(N):
        indices = np.nonzero(combinations[:, init_cam] == 1)[0]
        best_idx = np.argmax(rewards[indices])
        strategy[init_cam] = indices[best_idx]
    return strategy
