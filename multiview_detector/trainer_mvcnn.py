import random
import time
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from multiview_detector.models.mvdet import softmax_to_hard, masked_softmax


class ClassifierTrainer(object):
    def __init__(self, model, logdir, args, ):
        super(ClassifierTrainer, self).__init__()
        self.model = model
        self.args = args
        self.ce_loss = nn.CrossEntropyLoss()
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def train(self, epoch, dataloader, optimizer, scheduler=None, hard=None, identity_goal=False, log_interval=100):
        self.model.train()
        losses, correct, miss = 0, 0, 1e-8
        t0 = time.time()
        cam_prob_sum = torch.zeros([dataloader.dataset.num_cam])
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            if self.args.select:
                init_cam = np.concatenate([np.stack([b * np.ones(1), np.random.choice(
                    keep_cams[b].nonzero().squeeze(), 1, replace=False)]).T for b in range(B)])
                init_cam = torch.from_numpy(init_cam).long()
                # init_cam = keep_cams.nonzero()
                tgt = tgt[init_cam[:, 0]]
            else:
                init_cam = None
            cam_candidate, logits_prob = None, None
            reg_conf, reg_even = None, None
            output, (logits, cam_prob) = self.model(imgs, init_cam, keep_cams, hard)
            loss = self.ce_loss(output, tgt)

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += B - (pred == tgt).sum().item()

            # regularization
            if self.args.select:
                if init_cam is not None:
                    cam_candidate = keep_cams[init_cam[:, 0]].scatter(1, init_cam[:, [1]], 0).cuda()

                    # logits_prob = masked_softmax(logits, cam_candidate, dim=1)
                    # reg_conf = self.entropy(logits).mean()
                    # reg_even = -self.entropy(logits.mean(dim=0))
                    reg_conf = F.l1_loss(cam_prob, softmax_to_hard(cam_prob).detach())
                    reg_even = F.l1_loss(cam_prob.mean(dim=0), cam_candidate.sum(dim=0) / cam_candidate.sum())
                    loss += reg_conf * self.args.beta_conf + reg_even * self.args.beta_even
                    if identity_goal:
                        loss = F.l1_loss(cam_prob, cam_candidate / cam_candidate.sum(dim=1, keepdims=True))

            if init_cam is not None:
                cam_prob_sum += cam_prob.detach().cpu().sum(dim=0)

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
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.6f}, prec: {100. * correct / (correct + miss):.1f}%'
                      f'Time: {t_epoch:.1f}' + (f', prob: {cam_prob.detach().max(dim=1)[0].mean().item():.3f}'
                                                if self.args.select and init_cam is not None else ''))
                if self.args.select:
                    print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                                   zip(range(N), cam_prob_sum / cam_prob_sum.sum())))
                pass
            del imgs, keep_cams, init_cam, cam_candidate
            del logits, cam_prob, logits_prob
            del loss, output, tgt, pred
            del reg_conf, reg_even
        return losses / len(dataloader), correct / (correct + miss) * 100.0

    def test(self, dataloader, init_cam=None, override=None, visualize=False):
        self.model.eval()
        losses, correct, miss = 0, 0, 1e-8
        selected_cams = []
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            # with autocast():
            with torch.no_grad():
                output, (logits, cam_prob) = self.model(imgs, init_cam, override=override)
            loss = self.ce_loss(output, tgt)
            if init_cam is not None:
                selected_cams.extend(cam_prob.argmax(dim=1).detach().cpu().numpy().tolist())

            losses += loss.item()

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += B - (pred == tgt).sum().item()

        if init_cam is not None:
            unique_cams, unique_freq = np.unique(selected_cams, return_counts=True)
            print(' '.join('cam {} {} |'.format(cam, freq) for cam, freq in
                           zip(unique_cams, unique_freq / len(selected_cams))))

        print(f'Test, loss: {losses / len(dataloader):.3f}, prec: {100. * correct / (correct + miss):.2f}%, ')

        return losses / len(dataloader), correct / (correct + miss) * 100.0
