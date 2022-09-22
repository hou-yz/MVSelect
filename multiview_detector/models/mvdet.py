import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.shufflenetv2 import shufflenet_v2_x0_5
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap
from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
import matplotlib.pyplot as plt


def masked_softmax(input, mask=None, dim=1, epsilon=1e-5):
    if mask is None:
        mask = torch.ones_like(input, dtype=torch.bool)
    masked_exp = torch.exp(input) * mask.float()
    masked_sum = masked_exp.sum(dim, keepdim=True) + epsilon
    softmax = masked_exp / masked_sum
    return softmax


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1,
                   mask: torch.Tensor = None) -> torch.Tensor:
    # ~Gumbel(0,1)
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())
    # ~Gumbel(logits,tau)
    gumbels = (logits + gumbels) / tau
    y_soft = masked_softmax(gumbels, mask, dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


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


def create_coord_map(img_size, with_r=False):
    H, W = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        grid_r = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, grid_r], dim=1)
    return ret


class CamPredModule(nn.Module):
    def __init__(self, num_cam, hidden_dim):
        super().__init__()
        self.cam_feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU())
        # self.cam_emb = nn.Embedding(num_cam, hidden_dim)
        self.cam_pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, num_cam, bias=False))

    def forward(self, init_cam, init_cam_feat, cam_candidate):
        cam_feat = self.cam_feat(init_cam_feat).amax(dim=[2, 3])
        # cam_emb = self.cam_emb(init_cam)
        # cam_pred = self.cam_pred(torch.cat([cam_feat, cam_emb], dim=1))
        cam_prob = self.cam_pred(cam_feat)
        if self.training:
            # gumbel softmax trick
            cam_prob = gumbel_softmax(cam_prob, dim=1, hard=False, mask=cam_candidate)
        else:
            cam_prob = masked_softmax(cam_prob, cam_candidate, dim=1)
        return cam_prob


class MVDet(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, use_bottleneck=True, hidden_dim=128, outfeat_dim=0):
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.img_reduce = dataset.img_reduce
        self.num_cam = dataset.num_cam

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
                                      for cam in range(dataset.num_cam)])

        if arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        elif arch == 'shufflenet_x0.5':
            self.base = nn.Sequential(*list(shufflenet_v2_x0_5(pretrained=True,
                                                               replace_stride_with_dilation=[False, True, True]
                                                               ).children())[:-2])
            base_dim = 192
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        if use_bottleneck:
            self.bottleneck = nn.Conv2d(base_dim, hidden_dim, 1)
            base_dim = hidden_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)
        # self.img_id = output_head(base_dim, outfeat_dim, len(dataset.pid_dict))

        # world feat
        self.coord_map = create_coord_map(np.array(dataset.Rworld_shape))
        self.world_feat = nn.Sequential(nn.Conv2d(base_dim + 2, hidden_dim, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(), )

        # select camera based on initialization
        self.cam_pred = CamPredModule(dataset.num_cam, hidden_dim)

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

    def forward(self, imgs, M, init_cam=None, keep_cams=None, override=None, visualize=False):
        B, N, C, H, W = imgs.shape
        if keep_cams is None:
            keep_cams = torch.ones([B, N], dtype=torch.bool)
        # B = init_cam.shape
        imgs = imgs.view(B * N, C, H, W)

        inverse_affine_mats = torch.inverse(M.view([B * N, 3, 3]))
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ \
                                     torch.from_numpy(np.diag([self.img_reduce, self.img_reduce, 1])
                                                      ).view(1, 3, 3).repeat(B * N, 1, 1).float()
        # Rworldgrid(xy)_from_Rimggrid(xy)
        proj_mats = self.proj_mats.repeat(B, 1, 1, 1).view(B * N, 3, 3).float() @ imgcoord_from_Rimggrid_mat

        if visualize:
            denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            proj_imgs = warp_perspective(T.Resize(self.Rimg_shape)(imgs), proj_mats.to(imgs.device),
                                         self.Rworld_shape, align_corners=False). \
                view(B, N, 3, self.Rworld_shape[0], self.Rworld_shape[1])
            for cam in range(N):
                visualize_img = T.ToPILImage()(denorm(imgs.detach())[cam * B])
                visualize_img.save(f'../../imgs/augimg{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
                visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                plt.imshow(visualize_img)
                plt.show()

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat) * keep_cams.view(B * N, 1, 1, 1).to(imgs.device)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/augimgfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # img heads
        _, C, H, W = imgs_feat.shape
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        # world feat
        H, W = self.Rworld_shape
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape, align_corners=False)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat.view(B, N, C, H, W)[0, cam].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # world_feat = self.world_feat(world_feat) * keep_cams.view(B * N, 1, 1, 1).to(imgs.device)

        if init_cam is not None:
            if self.training:
                init_cam = keep_cams.to(imgs.device).nonzero()
                init_cam_feat = world_feat[init_cam[:, 0] * N + init_cam[:, 1]]
                cam_candidate = keep_cams.to(imgs.device).view(B * N, 1)[init_cam[:, 0] * N + init_cam[:, 1]
                                                                         ].repeat(1, N).scatter(1, init_cam[:, [1]], 0)
                cam_prob = self.cam_pred(init_cam[:, 1], init_cam_feat, cam_candidate)
                cam_selection = cam_prob.argmax(dim=1)
                # # mask out the original camera
                # y_soft = F.gumbel_softmax(cam_prob, dim=1) * cam_candidate
                # # F.gumbel_softmax(cam_prob, hard=True)
                # cam_selection = y_soft.argmax(dim=1)
                # y_hard = torch.zeros_like(y_soft).scatter_(-1, cam_selection[:, None], 1.0)
                # # cam_prob = y_hard - y_soft.detach() + y_soft
                # world_feat = torch.stack([(world_feat.view([B, N, C, H, W]).repeat_interleave(N, 0) *
                #                            cam_prob.view([B * N, N, 1, 1, 1])).sum(dim=1) / 2,
                #                           init_cam_feat], dim=1).mean(dim=1)
                # world_feat = torch.cat([(world_feat.view([B, N, C, H, W]).repeat_interleave(N, 0) *
                #                          cam_prob.view([B * N, N, 1, 1, 1])),
                #                         init_cam_feat[:, None]], dim=1).max(dim=1)[0]
                # world_feat = torch.cat([(world_feat * cam_prob.view([B * N, 1, 1, 1])).view(B, N, C, H, W),
                #                         init_cam_feat[:, None]], dim=1).max(dim=1)[0]
                world_feat = (world_feat.view(B, N, C, H, W)[init_cam[:, 0]] *
                              cam_prob.scatter(1, init_cam[:, [1]], 1)[:, :, None, None, None]).max(dim=1)[0]
                # world_feat = (init_cam_feat +
                #               (world_feat * cam_prob.view([B * N, 1, 1, 1])).view(B, N, C, H, W).sum(dim=1)) / 2
            else:
                init_cam_feat = world_feat[init_cam + torch.arange(B, device=imgs.device) * N]
                cam_candidate = keep_cams.to(imgs.device).scatter(1, init_cam[:, None], 0)
                cam_prob = self.cam_pred(init_cam, init_cam_feat, cam_candidate)
                # cam_prob = cam_prob > 0
                # distribution = torch.distributions.Categorical(cam_prob.view([B, N]))
                # cam_selection = distribution.sample()
                cam_selection = cam_prob.argmax(dim=1)

                if override is not None:
                    cam_selection = torch.as_tensor([override]).long().cuda()

                world_feat = torch.stack([world_feat[cam_selection + torch.arange(B, device=imgs.device) * N],
                                          init_cam_feat], dim=1).max(dim=1)[0]
                # world_feat = (init_cam_feat + world_feat[cam_selection + torch.arange(B).cuda() * N]) / 2
        else:
            world_feat = world_feat.view(B, N, C, H, W).max(dim=1)[0]
            cam_selection = None
        world_feat = torch.cat([world_feat, self.coord_map.repeat([world_feat.shape[0], 1, 1, 1]).to(imgs.device)], 1)
        world_feat = self.world_feat(world_feat)

        # world heads
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)
        # world_id = self.world_id(world_feat)

        if visualize:
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            visualize_img.save(f'../../imgs/worldfeatall.png')
            plt.imshow(visualize_img)
            plt.show()
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            visualize_img.save(f'../../imgs/worldres.png')
            plt.imshow(visualize_img)
            plt.show()
        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_selection


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.decode import ctdet_decode

    gumbel_softmax(torch.randn(2, 7))
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=False)
    dataloader = DataLoader(dataset, 2, False, num_workers=0)
    model = MVDet(dataset)
    imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = next(iter(dataloader))
    init_cam = torch.tensor([4, 0], dtype=torch.long)
    keep_cams[0, 3] = 0
    model.train()
    (world_heatmap, world_offset), _, cam_train = model(imgs, affine_mats, init_cam, keep_cams)
    xysc_train = ctdet_decode(world_heatmap, world_offset)
    model.eval()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_eval = \
        model(imgs, affine_mats, init_cam, keep_cams)
    xysc_eval = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
