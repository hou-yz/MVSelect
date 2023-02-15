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
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvselect import update_ema_variables, get_eps_thres


class ClassifierTrainer(object):
    def __init__(self, model, logdir, args, ):
        super(ClassifierTrainer, self).__init__()
        self.model = model
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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
                loss = []
                eps_thres = get_eps_thres(epoch - 1 + batch_idx / len(dataloader), self.args.epochs)
                # consider all cameras as initial one
                for cam in range(N):
                    log_probs, values, actions, entropies, rewards, task_loss = [], [], [], [], [], []
                    # get result from using initial camera feature
                    init_prob = F.one_hot(torch.tensor(cam).repeat(B), num_classes=N).cuda()
                    # init_prob = F.one_hot(torch.randint(N, [B]), num_classes=N).cuda()
                    # with torch.no_grad():
                    #     output_i = self.model.get_output((feat * init_prob[:, :, None, None, None]).sum(1))
                    # task_loss_last = F.cross_entropy(output_i, tgt, reduction='none')

                    # rollout episode
                    for i in range(self.args.steps):
                        overall_feat_i, (log_prob_i, value_i, action_i, entropy_i) = \
                            self.model.select_module(feat, init_prob, keep_cams, eps_thres)
                        output_i = self.model.get_output(overall_feat_i)
                        task_loss_i = F.cross_entropy(output_i, tgt, reduction='none')
                        # decrease in task loss means higher performance, hence positive reward
                        # reward_i = (task_loss_last - task_loss_i).detach()
                        # task_loss_last = task_loss_i.detach()
                        reward_i = torch.zeros_like(task_loss_i) if i < self.args.steps - 1 \
                            else (output_i.argmax(1) == tgt).float()
                        # reward_i += entropy_i.detach() * self.args.beta_entropy
                        # record state & transitions
                        log_probs.append(log_prob_i)
                        # values.append(value_i[:, 0])
                        values.append((value_i * action_i).sum(1))
                        actions.append(action_i)
                        entropies.append(entropy_i)
                        rewards.append(reward_i)
                        task_loss.append(task_loss_i)
                        # stats
                        action_sum += action_i.detach().sum(dim=0)
                        # update the init_prob
                        init_prob += action_i

                    log_probs, values, actions, entropies, rewards, task_loss = \
                        torch.stack(log_probs), torch.stack(values), torch.stack(actions), \
                            torch.stack(entropies), torch.stack(rewards), torch.stack(task_loss)
                    # calculate returns for each step in episode
                    R = torch.zeros([B]).cuda()
                    returns = torch.empty([self.args.steps, B]).cuda().float()
                    for i in reversed(range(self.args.steps)):
                        R = rewards[i] + self.args.gamma * R
                        returns[i] = R
                    # returns = (output_i.argmax(1) == tgt).float()[None, :].repeat([self.args.steps, 1]) - 0.5
                    if return_avg is None:
                        return_avg = returns.mean(1)
                    else:
                        return_avg = returns.mean(1) * 0.05 + return_avg * 0.95
                    # policy & value loss
                    value_loss = F.smooth_l1_loss(values, returns)
                    # policy_loss = (-log_probs * (returns - values.detach())).mean()
                    # task loss
                    task_loss = task_loss.mean()
                    # loss.append(value_loss + policy_loss + task_loss -
                    #             entropies.mean() * self.args.beta_entropy * eps_thres)
                    loss.append(value_loss + task_loss)
                loss = torch.stack(loss).mean()
                output = output_i
            else:
                overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
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
        return losses / len(dataloader), correct / (correct + miss) * 100.0

    def test(self, dataloader, init_cam=None):
        t0 = time.time()
        self.model.eval()
        losses, correct, miss = 0, 0, 1e-8
        action_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            # with autocast():
            with torch.no_grad():
                output, _, (log_probs, values, actions, entropies) = self.model(imgs, None, self.args.down, init_cam,
                                                                                self.args.steps)
            loss = F.cross_entropy(output, tgt)
            if init_cam is not None:
                # record actions
                action_sum += torch.cat(actions).sum(dim=0)

            losses += loss.item()

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += B - (pred == tgt).sum().item()

        if init_cam is not None:
            idx = action_sum.nonzero().cpu()[:, 0]
            print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                           zip(idx, F.normalize(action_sum, p=1, dim=0).cpu()[idx])))

        print(f'Test, loss: {losses / len(dataloader):.3f}, prec: {100. * correct / (correct + miss):.2f}%, '
              f'time: {time.time() - t0:.1f}s')

        return losses / len(dataloader), [correct / (correct + miss) * 100.0, ]

    def test_cam_combination(self, dataloader, init_cam):
        self.model.eval()
        loss_s, pred_s, gt_s = [], [], []
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            tgt = tgt.unsqueeze(1).repeat([1, N])
            with torch.no_grad():
                output, _, _ = self.model.forward_override_combination(imgs.cuda(), None, self.args.down, init_cam)
            loss = F.cross_entropy(output.flatten(0, 1), tgt.flatten(0, 1).cuda(), reduction="none")
            pred = torch.argmax(output, -1)
            loss_s.append(loss.cpu().unflatten(0, [B, N]))
            pred_s.append(pred.cpu())
            gt_s.append(tgt)
        loss_s, pred_s, gt_s = torch.cat(loss_s, 0), torch.cat(pred_s, 0), torch.cat(gt_s, 0)
        tp_s = pred_s == gt_s
        # instance level selection
        instance_lvl_max_prec = tp_s.max(dim=1)[0].float().mean().item()
        # dataset level selection
        dataset_lvl_prec = tp_s.float().mean(0).numpy()

        return loss_s.mean(0).numpy(), [instance_lvl_max_prec * 100.0, ], dataset_lvl_prec[:, None] * 100.0
