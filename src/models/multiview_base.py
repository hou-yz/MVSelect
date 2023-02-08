import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.mvselect import aggregate_init, aggregate_init_selection


class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation
        self.cam_pred = None

    def forward(self, imgs, M=None, down=1, init_prob=None, keep_cams=None, hard=None, visualize=False):
        feat, aux_res = self.get_feat(imgs, M, down, visualize)
        overall_feat, selection_res = self.cam_pred(feat, init_prob, keep_cams, hard)
        overall_res = self.get_output(overall_feat, visualize)
        return overall_res, aux_res, selection_res

    def get_feat(self, imgs, M, down=1, visualize=False):
        feat, aux_res = None, None
        return feat, aux_res

    def get_output(self, overall_feat, visualize=False):
        overall_result = None
        return overall_result

    def forward_override_combination(self, imgs, M, down, init_prob):
        B, N, C, H, W = imgs.shape
        if isinstance(init_prob, int):
            init_prob = F.one_hot(torch.tensor(init_prob).repeat(B), num_classes=N)
        elif isinstance(init_prob, np.ndarray):
            init_prob = F.one_hot(torch.tensor(init_prob), num_classes=N)
        init_prob = init_prob.bool().to(imgs.device)

        feat, aux_res = self.get_feat(imgs, M, down)
        init_feat = aggregate_init(feat, init_prob, self.aggregation)
        overall_feat_s, selection_prob_s = [], []
        for cam in range(self.num_cam):
            select_prob = F.one_hot(torch.tensor(cam).repeat(B), num_classes=self.num_cam).to(imgs.device)
            overall_feat = aggregate_init_selection(init_feat, init_prob, feat, select_prob, self.aggregation)
            overall_feat_s.append(overall_feat)
            selection_prob_s.append(select_prob)
        overall_feat_s = torch.stack(overall_feat_s, dim=1)
        selection_res_s = (None, None, torch.stack(selection_prob_s, dim=1))
        overall_result_s = self.get_output(overall_feat_s.flatten(0, 1))
        if isinstance(overall_result_s, tuple):
            overall_result_s = tuple(result.unflatten(0, [B, N]) for result in overall_result_s)
        else:
            overall_result_s = overall_result_s.unflatten(0, [B, N])
        return overall_result_s, aux_res, selection_res_s
