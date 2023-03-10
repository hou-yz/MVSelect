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
from src.utils.meters import AverageMeter
from src.trainer import BaseTrainer, find_instance_lvl_strategy, find_dataset_lvl_strategy
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvselect import aggregate_feat, get_eps_thres


class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, logdir, args, ):
        super(ClassifierTrainer, self).__init__(model, logdir, args, )

    def task_loss_reward(self, overall_feat, tgt, step):
        output = self.model.get_output(overall_feat)
        task_loss = F.cross_entropy(output, tgt, reduction='none')
        reward = torch.zeros_like(task_loss) if step < self.args.steps - 1 else (output.argmax(1) == tgt).float()
        return task_loss, reward

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=200):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses, correct, miss = 0, 0, 1e-8
        t0 = time.time()
        action_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        return_avg = None
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            feat, _ = self.model.get_feat(imgs, None, self.args.down)
            if self.args.steps:
                eps_thres = get_eps_thres(epoch - 1 + batch_idx / len(dataloader), self.args.epochs)
                loss, (action_sum, return_avg, value_loss) = \
                    self.expand_episode(feat, keep_cams, tgt, eps_thres, (action_sum, return_avg))
            else:
                overall_feat = aggregate_feat(feat, keep_cams, self.model.aggregation)
                output = self.model.get_output(overall_feat)
                loss = F.cross_entropy(output, tgt)

                pred = torch.argmax(output, 1)
                correct += (pred == tgt).sum().item()
                miss += B - (pred == tgt).sum().item()

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
        return losses / len(dataloader), None if self.args.steps else correct / (correct + miss) * 100.0

    def test(self, dataloader, init_cam=None):
        t0 = time.time()
        self.model.eval()
        K = len(init_cam) if init_cam is not None else 1
        losses, correct, miss = torch.zeros([K]), torch.zeros([K]), torch.zeros([K]) + 1e-8
        action_sum = torch.zeros([K, dataloader.dataset.num_cam]).cuda()
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            outputs, actions = [], []
            with torch.no_grad():
                if self.args.steps == 0 or init_cam is None:
                    output, _, (_, _, action, _) = self.model(imgs, None, self.args.down)
                    outputs.append(output)
                    actions.append(action)
                else:
                    feat, _ = self.model.get_feat(imgs, None, self.args.down)
                    # K, B, N
                    for k in range(K):
                        overall_feat, (_, _, action, _) = \
                            self.model.do_steps(feat, init_cam[k].repeat([B, 1]), self.args.steps, keep_cams)
                        output = self.model.get_output(overall_feat)
                        outputs.append(output)
                        actions.append(action)

            for k in range(K):
                loss = F.cross_entropy(outputs[k], tgt)
                if init_cam is not None:
                    # record actions
                    action_sum[k] += torch.cat(actions[k]).sum(dim=0)
                losses[k] += loss.item()

                pred = torch.argmax(outputs[k], 1)
                correct[k] += (pred == tgt).sum().item()
                miss[k] += B - (pred == tgt).sum().item()

        for k in range(K):
            if init_cam is not None:
                print(f'init camera {init_cam[k].nonzero()[0].item()}: MVSelect')
                idx = action_sum[k].nonzero().cpu()[:, 0]
                print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                               zip(idx, F.normalize(action_sum[k], p=1, dim=0).cpu()[idx])))

            print(f'Test, loss: {losses[k] / len(dataloader):.3f}, prec: {correct[k] / (correct[k] + miss[k]):.2%}' +
                  ('' if init_cam is not None else f', time: {time.time() - t0:.1f}s'))

        if init_cam is not None:
            print('*************************************')
            print(f'MVSelect average prec {correct.sum() / (correct + miss).sum():.2%}, time: {time.time() - t0:.1f}')
            print('*************************************')
        return losses.mean() / len(dataloader), [correct.sum() / (correct + miss).sum() * 100.0, ]

    def test_cam_combination(self, dataloader, step=0):
        self.model.eval()
        t0 = time.time()
        candidates = np.eye(dataloader.dataset.num_cam)
        combinations = np.array(list(itertools.combinations(candidates, step + 1))).sum(1)
        K, N = combinations.shape
        loss_s, pred_s, gt_s = [], [], []
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            gt_s.append(tgt)
            tgt = tgt.unsqueeze(0).repeat([K, 1])
            # K, B, N
            with torch.no_grad():
                output, _ = self.model.forward_combination(imgs.cuda(), None, self.args.down, combinations, keep_cams)
            loss = F.cross_entropy(output.flatten(0, 1), tgt.flatten(0, 1).cuda(), reduction="none")
            pred = torch.argmax(output, -1)
            loss_s.append(loss.unflatten(0, [K, B]).cpu())
            pred_s.append(pred.cpu())
        loss_s, pred_s, gt_s = torch.cat(loss_s, 1), torch.cat(pred_s, 1), torch.cat(gt_s)
        # K, num_frames
        tp_s = (pred_s == gt_s[None, :]).float()
        # instance level selection
        instance_lvl_strategy = find_instance_lvl_strategy(tp_s, combinations)
        instance_lvl_oracle = np.take_along_axis(tp_s, instance_lvl_strategy, axis=0).mean(1).numpy()[:, None]
        # dataset level selection
        combination_idx = combinations[:, keep_cams[0].bool().numpy()].sum(1).astype(np.bool)
        dataset_lvl_prec = tp_s.mean(1).numpy()[:, None]
        dataset_lvl_strategy = find_dataset_lvl_strategy(dataset_lvl_prec, combinations)
        dataset_lvl_best_prec = dataset_lvl_prec[dataset_lvl_strategy]
        oracle_info = f'{step} steps, averave acc {dataset_lvl_prec[combination_idx].mean():.1%}, ' \
                      f'dataset lvl best {dataset_lvl_best_prec.mean():.1%}, ' \
                      f'instance lvl oracle {instance_lvl_oracle.mean():.1%}, time: {time.time() - t0:.1f}s'
        print(oracle_info)
        return loss_s.mean(1).numpy(), dataset_lvl_prec * 100.0, instance_lvl_oracle * 100.0, oracle_info
