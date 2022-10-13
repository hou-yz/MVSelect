# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from multiview_detector.utils.tensor_utils import _transpose_and_gather_feat, _sigmoid
import torch.nn.functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, logits, dim=-1):
        return -torch.sum(F.softmax(logits, dim=dim) * F.log_softmax(logits, dim=dim), dim=dim)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        is_positive = target == 1
        mask = mask.to(output.device)

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.gamma) * is_positive * mask
        neg_loss = torch.log(1 - output) * torch.pow(output, self.gamma) * neg_weights * ~is_positive * mask

        if self.alpha >= 0:
            pos_loss *= self.alpha
            neg_loss *= (1 - self.alpha)

        loss = -(pos_loss + neg_loss).sum() / max(1, is_positive.sum())
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegCELoss(nn.Module):
    def __init__(self):
        super(RegCELoss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
        pred = _transpose_and_gather_feat(output, ind)
        if len(target[mask]) != 0:
            loss = F.cross_entropy(pred[mask], target[mask], reduction='sum')
            loss = loss / (mask.sum() + 1e-4)
        else:
            loss = 0
        return loss
