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


def create_pos_embedding(L, hidden_dim=128, temperature=10000, ):
    position = torch.arange(L).unsqueeze(1) / L * 2 * np.pi
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) / hidden_dim * (-np.log(temperature)))
    pe = torch.zeros(L, hidden_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CamPredModule(nn.Module):
    def __init__(self, num_cam, hidden_dim, kernel_size=1, gumbel=False, random_select=False):
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
        self.cam_emb = nn.Parameter(torch.zeros([num_cam, num_cam]))
        self.cam_pred = nn.Linear(hidden_dim, num_cam, bias=False)
        self.cam_pred.weight.data.fill_(0)
        self.gumbel = gumbel
        self.random_select = random_select

    def forward(self, init_cam, world_feat, keep_cams, hard=True, override=None):
        B, N, C, H, W = world_feat.shape
        world_feat = world_feat.view([B * N, C, H, W])
        if isinstance(init_cam, int):
            init_cam = torch.cat([torch.arange(B, device=world_feat.device)[:, None],
                                  torch.ones([B, 1], dtype=torch.long, device=world_feat.device) * init_cam], dim=1)
        else:
            init_cam = init_cam.to(world_feat.device)
        cam_candidate = keep_cams[init_cam[:, 0]].scatter(1, init_cam[:, [1]], 0)
        init_feat = world_feat[init_cam[:, 0] * N + init_cam[:, 1]]
        if override is None:
            if not self.random_select:
                cam_emb = F.layer_norm(self.cam_emb[init_cam[:, 1]], [N])
                # cam_feat = self.cam_feat(init_feat[:, :, None, None] if len(init_feat.shape) == 2 else init_feat)
                cam_feat = self.cam_feat(init_feat.amax(dim=[2, 3]))
                cam_pred = F.layer_norm(self.cam_pred(cam_feat), [N]) / 10
                logits = cam_pred + cam_emb
                # cam_feat = self.cam_feat(init_feat.amax(dim=[2, 3])) + self.cam_emb[init_cam[:, 1]]
                # logits = cam_pred = cam_emb = self.cam_pred(cam_feat)
            else:
                logits = cam_pred = cam_emb = torch.randn([init_cam.shape[0], N], device=world_feat.device)
            if self.training:
                assert hard is True or hard is False, 'plz provide bool type {hard}'
                # gumbel softmax trick
                if self.gumbel:
                    cam_prob = gumbel_softmax(logits, dim=1, mask=cam_candidate)
                else:
                    cam_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
                cam_prob_hard = softmax_to_hard(cam_prob)
            else:
                cam_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
                selected_cam = torch.argmax(cam_prob, dim=1)
                cam_prob_hard = torch.zeros_like(cam_prob).scatter_(1, selected_cam[:, None], 1.)
        else:
            logits = cam_pred = cam_emb = torch.randn([init_cam.shape[0], N], device=world_feat.device)
            selected_cam = torch.ones(init_cam.shape[0], device=world_feat.device).long() * override
            cam_prob = cam_prob_hard = torch.zeros_like(logits).scatter_(1, selected_cam[:, None], 1.)

        select_feat = world_feat.view(B, N, C, H, W)[init_cam[:, 0]] * \
                      (cam_prob_hard if hard is True or not self.training else cam_prob)[:, :, None, None, None]
        world_feat = torch.stack([init_feat, select_feat.sum(dim=1)], dim=1)
        return world_feat, (cam_emb, cam_pred, cam_prob_hard if hard is True or not self.training else cam_prob)


class MVDet(nn.Module):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 use_bottleneck=True, hidden_dim=128, outfeat_dim=0, z=0,
                 gumbel=False, random_select=False):
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
        elif arch == 'shufflenet0.5':
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

        self.aggregation = aggregation

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
        self.cam_pred = CamPredModule(dataset.num_cam, hidden_dim, 3, gumbel, random_select)

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

    def forward(self, imgs, M, init_cam=None, keep_cams=None, hard=None, override=None, visualize=False):
        B, N, _, H, W = imgs.shape
        if keep_cams is None:
            keep_cams = torch.ones([B, N], dtype=torch.bool)
        keep_cams = keep_cams.to(imgs.device)
        if self.training and init_cam is not None and hard is None:
            hard = True
        # B = init_cam.shape
        imgs = imgs.view(B * N, -1, H, W)

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
        imgs_feat = self.bottleneck(imgs_feat)
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
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape,
                                      align_corners=False).view(B, N, -1, H, W)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # world_feat = self.world_feat_pre(world_feat) * keep_cams.view(B * N, 1, 1, 1).to(imgs.device)

        if init_cam is not None:
            world_feat, (cam_emb, cam_pred, cam_prob) = self.cam_pred(init_cam, world_feat, keep_cams, hard, override)
        else:
            cam_emb, cam_pred, cam_prob = None, None, None
        world_feat = world_feat.mean(dim=1) if self.aggregation == 'mean' else world_feat.max(dim=1)[0]
        # world_feat = torch.cat([world_feat, self.coord_map.repeat([world_feat.shape[0], 1, 1, 1]).to(imgs.device)], 1)
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
        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), (cam_emb, cam_pred, cam_prob)


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.decode import ctdet_decode

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='train')
    dataloader = DataLoader(dataset, 2, False, num_workers=0)

    model = MVDet(dataset)
    imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = next(iter(dataloader))
    keep_cams[0, 3] = 0
    model.train()
    (world_heatmap, world_offset), _, cam_train = model(imgs, affine_mats, keep_cams.nonzero(), keep_cams)
    xysc_train = ctdet_decode(world_heatmap, world_offset)
    model.eval()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_eval = \
        model(imgs, affine_mats, 2, override=5)
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_eval = \
        model(imgs, affine_mats, keep_cams.nonzero(), override=5)
    xysc_eval = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
