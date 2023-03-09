import os

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from src.models.multiview_base import MultiviewBase
from src.models.mvselect import CamSelect, setup_args


class MVCNN(nn.Module):
    def __init__(self, dataset, arch='resnet18', aggregation='max'):
        super().__init__()
        if arch == 'resnet18':
            self.base = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, dataset.num_class)
            base_dim = 512
        elif arch == 'vgg11':
            self.base = models.vgg11(pretrained=True).features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = models.vgg11(pretrained=True).classifier
            self.classifier[-1] = nn.Linear(4096, dataset.num_class)
            base_dim = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        # select camera based on initialization
        self.select_module = CamSelect(dataset.num_cam, base_dim, 1, aggregation)
        pass

    def forward(self, imgs, init_prob=None, steps=0, ):
        B, N, _, _, _ = imgs.shape
        if self.select_module is None or init_prob is None or steps == 0:
            feat = []
            for i in range(N):
                feat_i = self.get_feat(imgs, i)
                feat.append(feat_i)
        else:
            init_feat = self.get_feat(imgs, init_prob)
            feat = [init_feat]
            for _ in range(steps):
                init_prob, _, cam_candidate = setup_args(imgs, init_prob)
                cam_emb = init_prob.float() @ self.select_module.cam_emb
                cam_emb = self.select_module.emb_branch(cam_emb)
                cam_feat = self.select_module.feat_branch(init_feat).amax(dim=[2, 3])
                action_value = self.select_module.value_head(cam_emb + cam_feat)
                action = torch.argmax(action_value + (cam_candidate.float() - 1) * 1e3, dim=-1)

                feat.append(self.get_feat(imgs, action))
                init_feat, _ = torch.stack(feat, 1).max(1)
                init_prob += F.one_hot(action, num_classes=N).bool()
        overall_feat, _ = torch.stack(feat, 1).max(1)
        overall_res = self.get_output(overall_feat)
        return overall_res

    def get_feat(self, imgs, view_id):
        B, N, _, H, W = imgs.shape
        if isinstance(view_id, int):
            view_id = torch.tensor([view_id] * B)
        batch_id = torch.arange(B).to(imgs.device)
        view_id = view_id.to(imgs.device)

        imgs_feat = self.base(imgs[batch_id, view_id])
        imgs_feat = self.avgpool(imgs_feat)
        _, C, H, W = imgs_feat.shape
        return imgs_feat

    def get_output(self, overall_feat):
        overall_result = self.classifier(torch.flatten(overall_feat, 1))
        return overall_result


if __name__ == '__main__':
    from src.datasets import imgDataset
    from torch.utils.data import DataLoader
    from thop import profile
    import itertools
    import time
    import tqdm

    dataset = imgDataset('/home/houyz/Data/modelnet/modelnet40v2png_ori4', 20)
    dataloader = DataLoader(dataset, 8, num_workers=0)
    imgs, tgt, keep_cams = next(iter(dataloader))

    torch.backends.cudnn.benchmark = False
    model = MVCNN(dataset).cuda()
    model.eval()
    # init_cam, step = 0, 1
    init_cam, step = None, None
    t0 = time.time()
    # avoid bottleneck @ dataloader
    for _ in tqdm.tqdm(range(1000)):
        model(imgs.cuda(), init_cam, step)
    print(time.time() - t0)
    pass
