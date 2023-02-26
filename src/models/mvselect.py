import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


def setup_args(feat, init_prob, keep_cams=None):
    B, N, C, H, W = feat.shape
    # init_prob should be of shape [B, N] in binary form
    if isinstance(init_prob, int):
        init_prob = F.one_hot(torch.tensor(init_prob).repeat(B), num_classes=N)
    elif isinstance(init_prob, np.ndarray):
        init_prob = F.one_hot(torch.tensor(init_prob), num_classes=N)
    init_prob = init_prob.bool().to(feat.device)
    if keep_cams is None:
        keep_cams = torch.ones([B, N], dtype=torch.bool)
    keep_cams = keep_cams.to(feat.device)
    cam_candidate = ~init_prob & keep_cams
    return init_prob, keep_cams, cam_candidate


def create_pos_embedding(L, hidden_dim=128, temperature=10000, ):
    position = torch.arange(L).unsqueeze(1) / L * 2 * np.pi
    div_term = temperature ** (torch.arange(0, hidden_dim, 2) / hidden_dim)
    pe = torch.zeros(L, hidden_dim)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    return pe


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


def aggregate_feat(feat, selection=None, aggregation='mean'):
    if selection is None:
        overall_feat = feat.mean(dim=1) if aggregation == 'mean' else feat.max(dim=1)[0]
    else:
        selection = selection.bool()
        overall_feat = feat * selection[:, :, None, None, None]
        if aggregation == 'mean':
            overall_feat = overall_feat.sum(dim=1) / (selection.sum(dim=1).view(-1, 1, 1, 1) + 1e-8)
        elif aggregation == 'max':
            overall_feat = overall_feat.max(dim=1)[0]
        else:
            raise Exception
    return overall_feat


class CamSelect(nn.Module):
    def __init__(self, num_cam, hidden_dim, kernel_size=1, aggregation='max'):
        super().__init__()
        self.aggregation = aggregation
        if kernel_size == 1:
            stride, padding = 1, 0
        elif kernel_size == 3:
            stride, padding = 2, 1
        else:
            raise Exception
        self.feat_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding), nn.ReLU(), )
        self.cam_emb = nn.Parameter(F.normalize(torch.randn([num_cam, hidden_dim]), p=2, dim=1))
        self.emb_branch = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.action_head = nn.Linear(hidden_dim, num_cam)
        self.action_head.weight.data.fill_(0)
        self.action_head.bias.data.fill_(0)
        self.value_head = nn.Linear(hidden_dim, num_cam)
        self.value_head.weight.data.fill_(0)
        self.value_head.bias.data.fill_(0)

    def forward(self, feat, init_prob, keep_cams=None, eps_thres=0.0):
        B, N, C, H, W = feat.shape
        init_prob, keep_cams, cam_candidate = setup_args(feat, init_prob, keep_cams)
        init_feat = aggregate_feat(feat, init_prob, self.aggregation)

        cam_emb = init_prob.float() @ self.cam_emb
        cam_emb = self.emb_branch(cam_emb)
        cam_feat = self.feat_branch(init_feat).amax(dim=[2, 3])
        # action_logit = self.action_head(cam_emb + cam_feat)
        # action_prob = F.softmax(action_logit, dim=-1) * cam_candidate
        # entropy = -(F.log_softmax(action_logit, dim=-1) * action_prob).sum(1)
        # state_value = self.value_head(cam_emb + cam_feat)

        # if self.training:
        #     m = Categorical(action_prob)
        #     action = m.sample()
        #     log_prob = m.log_prob(action)
        # else:
        #     action = torch.argmax(action_prob, dim=-1)
        #     log_prob = torch.zeros([B], device=feat.device)

        # DQN
        action_value = self.value_head(cam_emb + cam_feat)
        if random.random() > eps_thres:
            action = torch.argmax(action_value + (cam_candidate.float() - 1) * 1e3, dim=-1)
        else:
            # m = Categorical(F.normalize(cam_candidate.float(), p=1, dim=1))
            # action = m.sample()
            action = torch.randint(N, [B], device=feat.device)

        action = F.one_hot(action, num_classes=N).bool()
        overall_feat = aggregate_feat(feat, init_prob + action, self.aggregation)
        # return overall_feat, (log_prob, state_value, action, entropy)
        return overall_feat, (torch.zeros([B], device=feat.device), action_value,
                              action, torch.zeros([B], device=feat.device))


def update_ema_variables(model, ema_model, alpha=0.99):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)
    for ema_bn, bn in zip(ema_model.modules(), model.module.modules() if hasattr(model, 'module') else model.modules()):
        if isinstance(bn, nn.BatchNorm2d) or isinstance(bn, nn.SyncBatchNorm):
            ema_bn.running_mean = ema_bn.running_mean * alpha + bn.running_mean * (1 - alpha)
            ema_bn.running_var = ema_bn.running_var * alpha + bn.running_var * (1 - alpha)


def get_eps_thres(epoch, total_epochs, eps_start=0.9, eps_end=0.05):
    eps_thres = eps_end + (eps_start - eps_end) * (np.cos(epoch / total_epochs * np.pi) + 1) / 2
    return eps_thres
