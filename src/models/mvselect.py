import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(input, dim=-1, mask=None, epsilon=1e-8):
    if mask is None:
        mask = torch.ones_like(input, dtype=torch.bool)
    masked_exp = torch.exp(input) * mask.float()
    masked_sum = masked_exp.sum(dim, keepdim=True) + epsilon
    softmax = masked_exp / masked_sum
    return softmax


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -1, mask: torch.Tensor = None) -> torch.Tensor:
    # ~Gumbel(0,1)
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())
    # ~Gumbel(logits,tau)
    gumbels = (logits + gumbels) / tau
    y_soft = masked_softmax(gumbels, dim, mask)

    return y_soft


def softmax_to_hard(y_soft, dim=-1):
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


def aggregate_init(feat, init_prob, aggregation):
    init_feat = feat * init_prob[:, :, None, None, None]
    init_feat = init_feat.sum(dim=1) / init_prob.sum(dim=1).view(-1, 1, 1, 1) if aggregation == 'mean' \
        else init_feat.max(dim=1)[0]
    return init_feat


def aggregate_init_selection(init_feat, init_prob, feat, select_prob, aggregation):
    select_feat = (feat * select_prob[:, :, None, None, None]).sum(1)
    if aggregation == 'mean':
        overall_feat = (init_feat * init_prob.sum(1).view(-1, 1, 1, 1) + select_feat) / \
                       (init_prob + select_prob).sum(1).view(-1, 1, 1, 1)
    else:
        overall_feat = torch.stack([init_feat, select_feat], 1).max(1)[0]
    return overall_feat


class CamPredModule(nn.Module):
    def __init__(self, num_cam, hidden_dim, kernel_size=1, gumbel=True, random_select=False, aggregation='max'):
        super().__init__()
        self.cam_feat = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.kernel_size = kernel_size
        # if kernel_size == 1:
        #     stride, padding = 1, 0
        # elif kernel_size == 3:
        #     stride, padding = 2, 1
        # else:
        #     raise Exception
        # self.cam_feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding), nn.ReLU(),
        #                               nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding), nn.ReLU(), )
        # self.cam_feat[-2].weight.data.fill_(0)
        # self.cam_feat[-2].bias.data.fill_(0)
        # self.register_buffer('cam_emb', create_pos_embedding(num_cam, hidden_dim))
        # self.cam_emb = nn.Embedding(num_cam, num_cam)
        # self.cam_emb.weight.data.fill_(0)
        self.cam_emb = nn.Parameter(torch.zeros([num_cam, num_cam]))
        self.cam_pred = nn.Linear(hidden_dim, num_cam, bias=False)
        self.cam_pred.weight.data.fill_(0)
        self.gumbel = gumbel
        self.random_select = random_select
        self.aggregation = aggregation

    def forward(self, feat, init_prob, keep_cams=None, hard=None, ):
        B, N, C, H, W = feat.shape
        # init_prob should be of shape [B, N] in binary form
        if init_prob is None:
            cam_emb, cam_pred, select_prob = None, None, None
            overall_feat = feat.mean(dim=1) if self.aggregation == 'mean' else feat.max(dim=1)[0]
            return overall_feat, (cam_emb, cam_pred, select_prob)
        elif isinstance(init_prob, int):
            init_prob = F.one_hot(torch.tensor(init_prob).repeat(B), num_classes=N)
        elif isinstance(init_prob, np.ndarray):
            init_prob = F.one_hot(torch.tensor(init_prob), num_classes=N)
        init_prob = init_prob.bool().to(feat.device)
        if keep_cams is None:
            keep_cams = torch.ones([B, N], dtype=torch.bool)
        keep_cams = keep_cams.to(feat.device)
        if not self.training or hard is None:
            hard = True
        cam_candidate = ~init_prob & keep_cams
        init_feat = aggregate_init(feat, init_prob, self.aggregation)

        if not self.random_select:
            cam_emb = F.layer_norm(init_prob.float() @ self.cam_emb, [N])
            # cam_emb = F.layer_norm(self.cam_emb(init_prob.nonzero()[:, 1]), [N])
            # cam_feat = self.cam_feat(init_feat[:, :, None, None] if len(init_feat.shape) == 2 else init_feat)
            cam_feat = self.cam_feat(init_feat.amax(dim=[2, 3]))
            cam_pred = F.layer_norm(self.cam_pred(cam_feat), [N]) / 10
            logits = cam_pred + cam_emb
        else:
            logits = cam_pred = cam_emb = torch.randn([B, N], device=feat.device)
        if self.training:
            assert hard is True or hard is False, 'plz provide bool type {hard}'
            # gumbel softmax trick
            if self.gumbel:
                select_prob = gumbel_softmax(logits, dim=1, mask=cam_candidate)
            else:
                select_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
            select_prob_hard = softmax_to_hard(select_prob)
        else:
            select_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
            selected_cam = torch.argmax(select_prob, dim=1)
            select_prob_hard = F.one_hot(selected_cam, num_classes=N)

        select_prob = select_prob_hard if hard is True or not self.training else select_prob
        overall_feat = aggregate_init_selection(init_feat, init_prob, feat, select_prob, self.aggregation)
        return overall_feat, (cam_emb, cam_pred, select_prob)
