import os

os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
from src.models.resnet import resnet18
from src.models.shufflenetv2 import shufflenet_v2_x0_5
from src.models.mvselect import CamSelect, aggregate_feat, setup_args
from src.models.multiview_base import MultiviewBase
from src.utils.image_utils import img_color_denormalize, array2heatmap
from src.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
import matplotlib.pyplot as plt


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, 1, 1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class MVDet(nn.Module):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 use_bottleneck=True, hidden_dim=128, outfeat_dim=0, z=0):
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = np.array(dataset.Rimg_shape), np.array(dataset.Rworld_shape)
        self.img_reduce = dataset.img_reduce

        # world grid change to xy indexing
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam],
                                                                          dataset.base.extrinsic_matrices[cam],
                                                                          z / dataset.base.worldcoord_unit)
                                         for cam in range(dataset.num_cam)]
        # Rworldgrid(xy)_from_imgcoord(xy)
        self.proj_mats = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord_mat @
                                                       worldcoord_from_imgcoord_mats[cam])
                                      for cam in range(dataset.num_cam)]).float()

        if arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        elif arch == 'shufflenet0.5':
            self.base = nn.Sequential(*list(shufflenet_v2_x0_5(pretrained=True,
                                                               replace_stride_with_dilation=[False, True, True]
                                                               ).children())[:-2])
            base_dim = 192
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        if use_bottleneck:
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 1), nn.ReLU())
            base_dim = hidden_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)
        # self.img_id = output_head(base_dim, outfeat_dim, len(dataset.pid_dict))

        # world feat
        self.world_feat = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(), )

        # select camera based on initialization
        self.select_module = CamSelect(dataset.num_cam, hidden_dim, 3, aggregation)

        # world heads
        self.world_heatmap = output_head(hidden_dim, outfeat_dim, 1)
        self.world_offset = output_head(hidden_dim, outfeat_dim, 2)
        # self.world_id = output_head(hidden_dim, outfeat_dim, len(dataset.pid_dict))

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)
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

        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = torch.diag(torch.tensor([self.img_reduce, self.img_reduce, 1])
                                                ).unsqueeze(0).repeat(B, 1, 1).float()
        proj_mats = self.proj_mats.unsqueeze(0).repeat(B, 1, 1, 1)[batch_id, view_id] @ imgcoord_from_Rimggrid_mat

        imgs_feat = self.base(imgs[batch_id, view_id])
        imgs_feat = self.bottleneck(imgs_feat)

        # world feat
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape)

        return world_feat

    def get_output(self, world_feat):

        # world heads
        world_feat = self.world_feat(world_feat)
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)

        return world_heatmap, world_offset


if __name__ == '__main__':
    from src.datasets.frameDataset import frameDataset
    from src.datasets.Wildtrack import Wildtrack
    from src.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from src.utils.decode import ctdet_decode
    from thop import profile
    import tqdm

    dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='test')
    dataloader = DataLoader(dataset, 1, num_workers=0)
    imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = next(iter(dataloader))

    torch.backends.cudnn.benchmark = False
    model = MVDet(dataset).cuda()
    model.eval()
    # init_cam, step = 0, 2
    init_cam, step = None, None
    t0 = time.time()
    # avoid bottleneck @ dataloader
    for _ in tqdm.tqdm(range(1000)):
        model(imgs.cuda(), init_cam, step)
    print(time.time() - t0)
    pass
