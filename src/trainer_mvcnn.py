import random
import time
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.meters import AverageMeter
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvdet import softmax_to_hard, masked_softmax


class ClassifierTrainer(object):
    def __init__(self, model, logdir, args, ):
        super(ClassifierTrainer, self).__init__()
        self.model = model
        self.args = args
        self.ce_loss = nn.CrossEntropyLoss()
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def train(self, epoch, dataloader, optimizer, scheduler=None, hard=None, log_interval=1000):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses, correct, miss = 0, 0, 1e-8
        t0 = time.time()
        overall_prob_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        # cam_pred_ema = torch.zeros([dataloader.dataset.num_cam]).cuda()
        cam_pred_ema = torch.zeros([dataloader.dataset.num_cam, dataloader.dataset.num_cam]).cuda()
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            if self.args.select:
                init_cam = np.concatenate([np.random.choice(
                    keep_cams[b].nonzero().squeeze(), 1, replace=False) for b in range(B)])
                init_cam = F.one_hot(torch.tensor(init_cam), num_classes=N).cuda()
            else:
                init_cam = None
            reg_conf, reg_even = None, None
            output, (cam_emb, cam_pred, overall_prob) = self.model(imgs, init_cam, keep_cams, hard)
            loss = self.ce_loss(output, tgt)

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += B - (pred == tgt).sum().item()

            if self.args.select:
                # reg_conf
                reg_conf = (1 - F.softmax(cam_emb, dim=1).max(dim=1)[0]).mean()
                # reg_even
                # if cam_pred_ema.sum().item() == 0:
                #     cam_pred_ema = cam_pred.mean(dim=0).detach()
                # else:
                #     cam_pred_ema = cam_pred_ema * 0.99 + cam_pred.mean(dim=0).detach() * 0.01
                # reg_even = -(cam_pred_ema - cam_pred.mean(dim=0)).norm()
                reg_even = 0
                for b in range(len(init_cam)):
                    cam = init_cam[b, 1].item()
                    if cam_pred_ema[cam].sum().item() == 0:
                        cam_pred_ema[cam] = cam_pred[b].detach()
                    reg_even += -(cam_pred_ema[cam] - cam_pred[b]).norm()
                    cam_pred_ema[cam] = cam_pred_ema[cam] * 0.99 + cam_pred[b].detach() * 0.01
                reg_even /= B
                # loss
                loss = loss + reg_conf * self.args.beta_conf + reg_even * self.args.beta_even
                # record
                overall_prob_sum += overall_prob.detach().sum(dim=0)

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
                      f'loss: {losses / (batch_idx + 1):.3f}, prec: {100. * correct / (correct + miss):.1f}%, '
                      f'Time: {t_epoch:.1f}' + (f', prob: {overall_prob.detach().max(dim=1)[0].mean().item():.3f}, '
                                                f'reg_conf: {reg_conf.item():.3f}, reg_even: {reg_even.item():.3f}'
                                                if self.args.select and init_cam is not None else ''))
                if self.args.select:
                    print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                                   zip(range(N), F.normalize(overall_prob_sum, p=1, dim=0).cpu())))
                pass
            del imgs, keep_cams, init_cam
            del cam_emb, cam_pred, overall_prob
            del loss, output, tgt, pred
            del reg_conf, reg_even
        return losses / len(dataloader), correct / (correct + miss) * 100.0

    def test(self, dataloader, init_cam=None, override=None, visualize=False):
        t0 = time.time()
        self.model.eval()
        losses, correct, miss = 0, 0, 1e-8
        selected_cams = []
        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            # with autocast():
            with torch.no_grad():
                output, (cam_emb, cam_pred, overall_prob) = self.model(imgs, init_cam, override=override)
            loss = self.ce_loss(output, tgt)
            if init_cam is not None:
                selected_cams.extend(overall_prob.argmax(dim=1).detach().cpu().numpy().tolist())

            losses += loss.item()

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += B - (pred == tgt).sum().item()

        if init_cam is not None:
            unique_cams, unique_freq = np.unique(selected_cams, return_counts=True)
            print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                           zip(unique_cams, unique_freq / len(selected_cams))))

        print(f'Test, loss: {losses / len(dataloader):.3f}, prec: {100. * correct / (correct + miss):.2f}%, '
              f'time: {time.time() - t0:.1f}s')

        return losses / len(dataloader), [correct / (correct + miss) * 100.0, ]
