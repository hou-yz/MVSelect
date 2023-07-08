import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
from src.models.resnet import resnet18
from src.models.shufflenetv2 import shufflenet_v2_x0_5
from src.models.mvselect import CamSelect
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


class MVDet(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 use_bottleneck=True, hidden_dim=128, outfeat_dim=0, z=0):
        super().__init__(dataset, aggregation)
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

    def get_feat(self, imgs, M, down=1, visualize=False):
        B, N, _, H, W = imgs.shape
        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)

        inverse_affine_mats = torch.inverse(M.view([B * N, 3, 3]))
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ \
                                     torch.diag(torch.tensor([self.img_reduce * down, self.img_reduce * down, 1])
                                                ).unsqueeze(0).repeat(B * N, 1, 1).float()
        # Rworldgrid(xy)_from_Rimggrid(xy)
        # proj_mats = torch.diag(torch.tensor([1 / down, 1 / down, 1])).unsqueeze(0).repeat(B * N, 1, 1).float() @ \
        #             self.proj_mats.unsqueeze(0).repeat(B, 1, 1, 1).flatten(0, 1) @ imgcoord_from_Rimggrid_mat
        proj_mats = self.proj_mats[:N].unsqueeze(0).repeat(B, 1, 1, 1).flatten(0, 1) @ imgcoord_from_Rimggrid_mat

        if visualize:
            denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            proj_imgs = warp_perspective(F.interpolate(imgs, scale_factor=1 / 8), proj_mats.to(imgs.device),
                                         (self.Rworld_shape / down).astype(int)).unflatten(0, [B, N])
            for cam in range(N):
                visualize_img = T.ToPILImage()(denorm(imgs.detach())[cam * B])
                # visualize_img.save(f'../../imgs/augimg{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
                visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                plt.imshow(visualize_img)
                plt.show()

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat)

        # img heads
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
                # visualize_img.save(f'../../imgs/augimgfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # world feat
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape).unflatten(0, [B, N])

        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                # visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # world_feat = self.world_feat_pre(world_feat) * keep_cams.view(B * N, 1, 1, 1).to(imgs.device)
        return world_feat, (F.interpolate(imgs_heatmap, tuple(self.Rimg_shape)),
                            F.interpolate(imgs_offset, tuple(self.Rimg_shape)),
                            F.interpolate(imgs_wh, tuple(self.Rimg_shape)))

    def get_output(self, world_feat, visualize=False):

        # world heads
        world_feat = self.world_feat(world_feat)
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)
        # world_id = self.world_id(world_feat)

        if visualize:
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            # visualize_img.save(f'../../imgs/worldfeatall.png')
            plt.imshow(visualize_img)
            plt.show()
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            # visualize_img.save(f'../../imgs/worldres.png')
            plt.imshow(visualize_img)
            plt.show()

        return world_heatmap, world_offset



if __name__ == '__main__':
    from src.datasets.frameDataset import frameDataset
    from src.datasets.wildtrack import Wildtrack
    from src.datasets.multiviewx import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from src.utils.decode import ctdet_decode
    from thop import profile

    dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train', augmentation=True)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)

    model = MVDet(dataset).cuda()
    imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = next(iter(dataloader))
    keep_cams[0, 3] = 0
    init_cam = 0
    model.train()
    (world_heatmap, world_offset), _, cam_train = model(imgs.cuda(), affine_mats, 2, init_cam, 3)
    xysc_train = ctdet_decode(world_heatmap, world_offset)
    # macs, params = profile(model, inputs=(imgs[:, :3].cuda(), affine_mats[:, :3].contiguous()))
    # macs, params = profile(model.select_module, inputs=(torch.randn([1, 128, 160, 250]).cuda(),
    #                                                     F.one_hot(torch.tensor([1]), num_classes=6).cuda()))
    # macs, params = profile(model, inputs=(torch.rand([1, 128, 160, 250]).cuda(),))
    pass
