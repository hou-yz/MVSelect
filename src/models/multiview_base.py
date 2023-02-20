import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.mvselect import aggregate_feat, setup_args


class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation
        self.select_module = None

    def forward(self, imgs, M=None, down=1, init_prob=None, steps=0, keep_cams=None, visualize=False):
        feat, aux_res = self.get_feat(imgs, M, down, visualize)
        if self.select_module is None or init_prob is None or steps == 0:
            overall_feat = aggregate_feat(feat, aggregation=self.aggregation)
            selection_res = (None, None, None, None)
        else:
            overall_feat, selection_res = self.do_steps(feat, init_prob, steps, keep_cams)
        overall_res = self.get_output(overall_feat, visualize)
        return overall_res, aux_res, selection_res

    def do_steps(self, feat, init_prob, steps, keep_cams):
        assert steps > 0
        init_prob, _, _ = setup_args(feat, init_prob)
        log_probs, values, actions, entropies = [], [], [], []
        for _ in range(steps):
            overall_feat, (log_prob, state_value, action, entropy) = self.select_module(feat, init_prob, keep_cams)
            init_prob += action
            log_probs.append(log_prob)
            values.append(state_value)
            actions.append(action)
            entropies.append(entropy)
        selection_res = (log_probs, values, actions, entropies)
        return overall_feat, selection_res

    def get_feat(self, imgs, M, down=1, visualize=False):
        raise NotImplementedError

    def get_output(self, overall_feat, visualize=False):
        raise NotImplementedError

    def forward_override_combination(self, imgs, M, down, combinations):
        B, N, C, H, W = imgs.shape
        K, N = combinations.shape

        feat, aux_res = self.get_feat(imgs, M, down)

        # K, B, N
        combinations = torch.tensor(combinations, dtype=torch.float, device=imgs.device).unsqueeze(1).repeat([1, B, 1])

        # K, B, N, C, H, W
        overall_feat_s = [aggregate_feat(feat, combinations[k], self.aggregation) for k in range(K)]
        overall_result_s = [self.get_output(overall_feat_s[k]) for k in range(K)]
        if isinstance(overall_result_s[0], tuple):
            overall_result_s = tuple([torch.stack([overall_result_s[k][i] for k in range(K)])
                                      for i in range(len(overall_result_s[0]))])
        else:
            overall_result_s = torch.stack(overall_result_s)
        return overall_result_s, aux_res


if __name__ == '__main__':
    B, N, C, H, W = 2, 7, 512, 16, 16
    steps = 3
    feat = torch.randn([B, N, C, H, W], device='cuda')

    candidates = np.eye(N)
    combinations = np.array(list(itertools.combinations(candidates, steps + 1))).sum(1)
    K = len(combinations)
    combinations = torch.from_numpy(combinations).to(feat.device).unsqueeze(0).repeat([B, 1, 1])
    feat = feat.unsqueeze(1).repeat([1, K, 1, 1, 1, 1])

    overall_feat_s = [aggregate_feat(feat[:, k], combinations[:, k]) for k in range(K)]
    print((torch.stack(overall_feat_s, dim=1).flatten(0, 1) ==
           aggregate_feat(feat.flatten(0, 1), combinations.flatten(0, 1))).prod().item())
