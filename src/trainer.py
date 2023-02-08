import random
import time
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
from src.models.mvselect import softmax_to_hard, masked_softmax


class PerspectiveTrainer(object):
    def __init__(self, model, logdir, args, ):
        super(PerspectiveTrainer, self).__init__()
        self.model = model
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.entropy = Entropy()
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def train(self, epoch, dataloader, optimizer, scheduler=None, hard=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses = 0
        t0 = time.time()
        select_prob_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        # cam_pred_ema = torch.zeros([dataloader.dataset.num_cam]).cuda()
        cam_pred_ema = torch.zeros([dataloader.dataset.num_cam, dataloader.dataset.num_cam]).cuda()
        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            reg_conf, reg_even = None, None
            feat, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(imgs.cuda(), affine_mats, self.args.down)
            if self.args.select:
                init_prob = []
                overall_feat = []
                cam_emb, cam_pred, select_prob = [], [], []
                world_heatmap, world_offset = [], []
                for cam in range(N):
                    init_prob_i = F.one_hot(torch.tensor(cam).repeat(B), num_classes=N).cuda()
                    overall_feat_i, (cam_emb_i, cam_pred_i, select_prob_i) = \
                        self.model.cam_pred(feat, init_prob_i, keep_cams, hard)
                    world_heatmap_i, world_offset_i = self.model.get_output(overall_feat_i)
                    init_prob.append(init_prob_i)
                    overall_feat.append(overall_feat_i)
                    cam_emb.append(cam_emb_i)
                    cam_pred.append(cam_pred_i)
                    select_prob.append(select_prob_i)
                    world_heatmap.append(world_heatmap_i)
                    world_offset.append(world_offset_i)
                cam_emb, cam_pred = torch.stack(cam_emb, 1).flatten(0, 1), torch.stack(cam_pred, 1).flatten(0, 1)
                init_prob, select_prob = torch.stack(init_prob, 1).flatten(0, 1), \
                    torch.stack(select_prob, 1).flatten(0, 1)
                world_heatmap, world_offset = torch.stack(world_heatmap, 1).flatten(0, 1), \
                    torch.stack(world_offset, 1).flatten(0, 1)
                for key in world_gt.keys():
                    world_gt[key] = world_gt[key].repeat_interleave(N, 0)
            else:
                init_prob = None
                overall_feat, (cam_emb, cam_pred, select_prob) = self.model.cam_pred(feat, init_prob, keep_cams, hard)
                world_heatmap, world_offset = self.model.get_output(overall_feat)
            # loss = self.focal_loss(world_heatmap, world_gt['heatmap'])
            loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
            loss_w_off = self.regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
            # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
            loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'], keep_cams.view(B * N, 1, 1, 1))
            loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
            loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
            # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])

            w_loss = loss_w_hm + loss_w_off  # + self.args.id_ratio * loss_w_id
            img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.args.id_ratio * loss_img_id
            loss = w_loss + img_loss / N * self.args.alpha
            if self.args.use_mse:
                loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
                       self.args.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) / N

            if self.args.select:
                # reg_coverage
                # Rworld_coverage = dataloader.dataset.Rworld_coverage.cuda()
                # if self.args.eval_init_cam:
                #     loss = self.focal_loss(world_heatmap, world_gt['heatmap'], init_prob @ Rworld_coverage)
                # else:
                #     H, W = dataloader.dataset.Rworld_shape
                #     coverages = ((softmax_to_hard(select_prob) + init_prob.float()) @
                #                  Rworld_coverage.view(N, -1).float()).view(-1, 1, H, W).clamp(0, 1)
                #     loss = (self.focal_loss(world_heatmap, world_gt['heatmap'], coverages.detach()) +
                #             self.args.beta_coverage * (1 - coverages).mean())
                loss = loss_w_hm
                # reg_conf
                reg_conf = (1 - F.softmax(cam_emb, dim=1).max(dim=1)[0]).mean()
                # reg_even
                # if cam_pred_ema.sum().item() == 0:
                #     cam_pred_ema = cam_pred.mean(dim=0).detach()
                # else:
                #     cam_pred_ema = cam_pred_ema * 0.99 + cam_pred.mean(dim=0).detach() * 0.01
                # reg_even = -(cam_pred_ema - cam_pred.mean(dim=0)).norm()
                reg_even = 0
                for b in range(len(init_prob)):
                    cam = init_prob[b].nonzero().item()
                    if cam_pred_ema[cam].sum().item() == 0:
                        cam_pred_ema[cam] = cam_pred[b].detach()
                    reg_even += -(cam_pred_ema[cam] - cam_pred[b]).norm()
                    cam_pred_ema[cam] = cam_pred_ema[cam] * 0.99 + cam_pred[b].detach() * 0.01
                reg_even /= B
                # loss
                loss = loss + reg_conf * self.args.beta_conf + reg_even * self.args.beta_even
                # record
                select_prob_sum += select_prob.detach().sum(dim=0)

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
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.3f}, '
                      f'Time: {t_epoch:.1f}' + (f', prob: {select_prob.detach().max(dim=1)[0].mean().item():.3f}, '
                                                f'reg_conf: {reg_conf.item():.3f}, reg_even: {reg_even.item():.3f}'
                                                if self.args.select and init_prob is not None else ''))
                if self.args.select:
                    print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                                   zip(range(N), F.normalize(select_prob_sum, p=1, dim=0).cpu())))
                pass
            del imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams, init_prob
            del world_heatmap, world_offset, imgs_heatmap, imgs_offset, imgs_wh, cam_emb, cam_pred, select_prob
            del loss,  # w_loss, loss_w_hm, loss_w_off, img_loss, loss_img_hm, loss_img_off, loss_img_wh
            del reg_conf, reg_even
        return losses / len(dataloader), None

    def test(self, dataloader, init_cam=None):
        t0 = time.time()
        self.model.eval()
        losses = 0
        res_list = []
        selected_cams = []
        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            # with autocast():
            with torch.no_grad():
                (world_heatmap, world_offset), _, (cam_emb, cam_pred, select_prob) = \
                    self.model(imgs.cuda(), affine_mats, self.args.down, init_cam)
            loss = self.focal_loss(world_heatmap, world_gt['heatmap'])
            if self.args.use_mse:
                loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device))
            if init_cam is not None:
                selected_cams.extend(select_prob.argmax(dim=1).detach().cpu().numpy().tolist())

            losses += loss.item()

            if self.args.eval_init_cam and init_cam is not None:
                world_heatmap *= dataloader.dataset.Rworld_coverage[init_cam].cuda()
            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset, reduce=dataloader.dataset.world_reduce).cpu()
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
                res_list.append(res)

        if init_cam is not None:
            unique_cams, unique_freq = np.unique(selected_cams, return_counts=True)
            print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                           zip(unique_cams, unique_freq / len(selected_cams))))

        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(f'{self.logdir}/test.txt', res_list, '%d')
        gt_fname = f'{dataloader.dataset.gt_fname}_{init_cam}.txt' \
            if self.args.eval_init_cam and init_cam is not None else f'{dataloader.dataset.gt_fname}.txt'
        moda, modp, precision, recall = evaluate(f'{self.logdir}/test.txt',
                                                 gt_fname,
                                                 dataloader.dataset.base.__name__,
                                                 dataloader.dataset.frames)
        print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%, '
              f'\t loss: {losses / len(dataloader):.6f}, time: {time.time() - t0:.1f}s')

        return losses / len(dataloader), [moda, modp, precision, recall, ]

    def test_cam_combination(self, dataloader, init_cam):
        self.model.eval()
        loss_s, result_s = [[] for _ in range(self.model.num_cam)], [[] for _ in range(self.model.num_cam)]
        metric_s, stats_s = [], []
        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            with torch.no_grad():
                (world_heatmap, world_offset), _, _ = \
                    self.model.forward_override_combination(imgs.cuda(), affine_mats, self.args.down, init_cam)

            if self.args.eval_init_cam:
                world_heatmap *= dataloader.dataset.Rworld_coverage[init_cam].cuda()
            # decode
            xys = mvdet_decode(torch.sigmoid(world_heatmap.flatten(0, 1)), world_offset.flatten(0, 1),
                               reduce=dataloader.dataset.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            if dataloader.dataset.base.indexing == 'xy':
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]
            positions, scores = positions.unflatten(0, [B, N]), scores.unflatten(0, [B, N])
            for b in range(B):
                for cam in range(self.model.num_cam):
                    ids = scores[b, cam].squeeze() > self.args.cls_thres
                    pos, s = positions[b, cam, ids], scores[b, cam, ids, 0]
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    result_s[cam].append(res)

                    # Rworld_coverage = dataloader.dataset.Rworld_coverage.cuda()
                    # H, W = dataloader.dataset.Rworld_shape
                    # coverages = ((F.one_hot(torch.tensor(init_cam).repeat(B), num_classes=N) +
                    #               F.one_hot(torch.tensor(cam).repeat(B), num_classes=N)).cuda().float() @
                    #              Rworld_coverage.view(N, -1).float()).view(-1, 1, H, W).clamp(0, 1)
                    # loss = self.focal_loss(world_heatmap[[b], cam], world_gt['heatmap'][[b]], coverages.detach()) + \
                    #        self.args.beta_coverage * (1 - coverages).mean()
                    loss = self.focal_loss(world_heatmap[[b], cam], world_gt['heatmap'][[b]])
                    loss_s[cam].append(loss.cpu().item())
        result_s = [torch.cat(result_s[cam], 0) for cam in range(self.model.num_cam)]
        loss_s = np.array(loss_s)
        # eval
        gt_fname = f'{dataloader.dataset.gt_fname}_{init_cam}.txt' \
            if self.args.eval_init_cam and init_cam is not None else f'{dataloader.dataset.gt_fname}.txt'
        for cam in range(self.model.num_cam):
            moda, modp, prec, recall, stats = evaluateDetection_py(result_s[cam], gt_fname,
                                                                   frames=dataloader.dataset.frames)
            metric_s.append([moda, modp, prec, recall])
            stats_s.append(stats)
        metric_s = np.array(metric_s)
        tp_per_frame = np.array([stats_s[cam][0] for cam in range(self.model.num_cam)])
        oracle_result_strategy = np.argmax(tp_per_frame, axis=0)
        oracle_loss_strategy = np.argmin(loss_s, axis=0)
        oracle_result = []
        for frame_idx, frame in enumerate(dataloader.dataset.frames):
            cam = oracle_result_strategy[frame_idx]
            oracle_result.append(result_s[cam][result_s[cam][:, 0] == frame])
        oracle_result = np.concatenate(oracle_result)
        moda, modp, prec, recall, _ = evaluateDetection_py(oracle_result, gt_fname, frames=dataloader.dataset.frames)
        return loss_s.mean(1), [moda, modp, prec, recall], metric_s
