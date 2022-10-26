import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from multiview_detector.models.mvdet import CamPredModule


class MVCNN(nn.Module):
    def __init__(self, dataset, arch='vgg11', aggregation='max', ):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation

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
        self.cam_pred = CamPredModule(dataset.num_cam, base_dim)
        pass

    def forward(self, imgs, init_cam=None, keep_cams=None, hard=None, override=None, visualize=False):
        B, N, C, H, W = imgs.shape
        if keep_cams is None:
            keep_cams = torch.ones([B, N], dtype=torch.bool)
        keep_cams = keep_cams.to(imgs.device)
        if self.training and init_cam is not None and hard is None:
            hard = False
        imgs = imgs.view(B * N, C, H, W)

        imgs_feat = self.base(imgs)
        imgs_feat = self.avgpool(imgs_feat) * keep_cams.view(B * N, 1, 1, 1)
        _, C, H, W = imgs_feat.shape
        imgs_feat = imgs_feat.view(B, N, C, H, W)

        if init_cam is not None:
            overall_feat, (logits, cam_prob) = self.cam_pred(init_cam, imgs_feat, keep_cams, hard, override)
        else:
            overall_feat, (logits, cam_prob) = imgs_feat, (None, None)
        overall_feat = overall_feat.mean(dim=1) if self.aggregation == 'mean' else overall_feat.max(dim=1)[0]
        overall_result = self.classifier(torch.flatten(overall_feat, 1))

        return overall_result, (logits, cam_prob)


if __name__ == '__main__':
    from multiview_detector.datasets import imgDataset
    from torch.utils.data import DataLoader

    dataset = imgDataset('/home/houyz/Data/modelnet/modelnet40_images_new_12x', 12, mode='multi')
    dataloader = DataLoader(dataset, 2, False, num_workers=0)
    imgs, tgt, keep_cams = next(iter(dataloader))
    model = MVCNN(dataset)
    keep_cams[0, 3] = 0
    model.train()
    res = model(imgs, keep_cams.nonzero(), keep_cams)
    model.eval()
    res = model(imgs, 2, override=5)
    res = model(imgs, keep_cams.nonzero(), override=5)
    pass
