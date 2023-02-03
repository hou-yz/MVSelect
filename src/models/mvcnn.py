import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from src.models.multiview_base import MultiviewBase
from src.models.mvselect import CamPredModule



class MVCNN(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max', gumbel=True, random_select=False):
        super().__init__(dataset, aggregation, )

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
        self.cam_pred = CamPredModule(dataset.num_cam, base_dim, 1, gumbel, random_select)
        pass

    def get_feat(self, imgs, M, visualize=False):
        B, N, C, H, W = imgs.shape
        imgs_feat = self.base(imgs.flatten(0, 1))
        imgs_feat = self.avgpool(imgs_feat)
        _, C, H, W = imgs_feat.shape
        return imgs_feat.unflatten(0, [B, N]), None

    def get_output(self, overall_feat, visualize=False):
        overall_result = self.classifier(torch.flatten(overall_feat, 1))
        return overall_result


if __name__ == '__main__':
    from src.datasets import imgDataset
    from torch.utils.data import DataLoader
    from thop import profile

    dataset = imgDataset('/home/houyz/Data/modelnet/modelnet40v2png_ori4', 20)
    dataloader = DataLoader(dataset, 2, False, num_workers=0)
    imgs, tgt, keep_cams = next(iter(dataloader))
    model = MVCNN(dataset).cuda()
    init_prob = F.one_hot(torch.tensor([0, 1]), num_classes=dataset.num_cam)
    keep_cams[0, 3] = 0
    # model.train()
    # res = model(imgs.cuda(), None, init_prob, keep_cams)
    # model.eval()
    # res = model(imgs.cuda(), None, init_prob, override=5)
    with torch.no_grad():
        cam_combination_results = model.forward_override_combination(imgs.cuda(), None, 1)
    # macs, params = profile(model, inputs=(imgs[:, ],))
    #
    # print(f'{macs}')
    # print(f'{params}')
    pass
