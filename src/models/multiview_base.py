import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.mvselect import get_init_feat, get_joint_feat, setup_args



class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation
        self.select_module = None

    def forward(self, imgs, M=None, down=1, init_prob=None, steps=0, keep_cams=None, visualize=False):
        feat, aux_res = self.get_feat(imgs, M, down, visualize)
        if self.select_module is None or init_prob is None or steps == 0:
            overall_feat = feat.mean(dim=1) if self.aggregation == 'mean' else feat.max(dim=1)[0]
            selection_res = (None, None, None, None)
        else:
            init_prob, _, _ = setup_args(imgs, init_prob)
            log_probs, values, actions, entropies = [], [], [], []
            for _ in range(steps):
                overall_feat, (log_prob, state_value, action, entropy) = self.select_module(feat, init_prob, keep_cams)
                init_prob += action
                log_probs.append(log_prob)
                values.append(state_value)
                actions.append(action)
                entropies.append(entropy)
            selection_res = (log_probs, values, actions, entropies)
        overall_res = self.get_output(overall_feat, visualize)
        return overall_res, aux_res, selection_res

    def get_feat(self, imgs, M, down=1, visualize=False):
        raise NotImplementedError

    def get_output(self, overall_feat, visualize=False):
        raise NotImplementedError

    def forward_override_combination(self, imgs, M, down, init_prob):
        B, N, C, H, W = imgs.shape
        init_prob, _, _ = setup_args(imgs, init_prob)

        feat, aux_res = self.get_feat(imgs, M, down)
        overall_feat_s, selection_prob_s = [], []
        for cam in range(self.num_cam):
            action = F.one_hot(torch.tensor(cam).repeat(B), num_classes=self.num_cam).to(imgs.device)
            overall_feat = get_joint_feat(feat, init_prob, action, self.aggregation)
            overall_feat_s.append(overall_feat)
            selection_prob_s.append(action)
        overall_feat_s = torch.stack(overall_feat_s, dim=1)
        selection_res_s = (None, None, torch.stack(selection_prob_s, dim=1))
        overall_result_s = self.get_output(overall_feat_s.flatten(0, 1))
        if isinstance(overall_result_s, tuple):
            overall_result_s = tuple(result.unflatten(0, [B, N]) for result in overall_result_s)
        else:
            overall_result_s = overall_result_s.unflatten(0, [B, N])
        return overall_result_s, aux_res, selection_res_s
