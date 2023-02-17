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
from src.trainer import BaseTrainer
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvselect import update_ema_variables, get_eps_thres


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
        return losses / len(dataloader), None if self.args.steps else correct / (correct + miss) * 100.0

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
