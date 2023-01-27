import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
from src.models.resnet import resnet18
from src.models.shufflenetv2 import shufflenet_v2_x0_5
from src.utils.image_utils import img_color_denormalize, array2heatmap
from src.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
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
        self.cam_emb = nn.Embedding(num_cam, num_cam)
        self.cam_emb.weight.data.fill_(0)
        self.cam_pred = nn.Linear(hidden_dim, num_cam, bias=False)
        self.cam_pred.weight.data.fill_(0)
        self.gumbel = gumbel
        self.random_select = random_select
        self.aggregation = aggregation

    def forward(self, feat, init_cam, keep_cams, hard=True, override=None):
        B, N, C, H, W = feat.shape
        # init_cam should be of shape [B, N] in binary form
        if init_cam is None:
            overall_feat, (cam_emb, cam_pred, overall_prob) = feat, (None, None, None)
            overall_feat = overall_feat.mean(dim=1) if self.aggregation == 'mean' else overall_feat.max(dim=1)[0]
            return overall_feat, (cam_emb, cam_pred, overall_prob)
        elif isinstance(init_cam, int):
            init_cam = F.one_hot(torch.tensor(init_cam).repeat(B), num_classes=N)
        elif isinstance(init_cam, np.ndarray):
            init_cam = F.one_hot(torch.tensor(init_cam), num_classes=N)
        init_cam = init_cam.bool().to(feat.device)
        if keep_cams is None:
            keep_cams = torch.ones([B, N], dtype=torch.bool)
        keep_cams = keep_cams.to(feat.device)
        if self.training and hard is None:
            hard = True
        cam_candidate = ~init_cam & keep_cams
        init_feat = feat * init_cam[:, :, None, None, None]
        init_feat = init_feat.sum(dim=1) / init_cam.sum(dim=1) if self.aggregation == 'mean' \
            else init_feat.max(dim=1)[0]
        if override is None:
            # cam_emb = F.layer_norm(init_cam.float() @ self.cam_emb.weight, [N])
            cam_emb = F.layer_norm(self.cam_emb(init_cam.nonzero()[:, 1]), [N])
            # cam_feat = self.cam_feat(init_feat[:, :, None, None] if len(init_feat.shape) == 2 else init_feat)
            cam_feat = self.cam_feat(init_feat.amax(dim=[2, 3]))
            cam_pred = F.layer_norm(self.cam_pred(cam_feat), [N]) / 10
            logits = cam_pred + cam_emb
            # cam_feat = self.cam_feat(init_feat.amax(dim=[2, 3])) + self.cam_emb[init_cam[:, 1]]
            # logits = cam_pred = cam_emb = self.cam_pred(cam_feat)
            if self.random_select:
                logits = cam_pred = cam_emb = torch.randn([B, N], device=feat.device)
            if self.training:
                assert hard is True or hard is False, 'plz provide bool type {hard}'
                # gumbel softmax trick
                if self.gumbel:
                    overall_prob = gumbel_softmax(logits, dim=1, mask=cam_candidate)
                else:
                    overall_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
                overall_prob_hard = softmax_to_hard(overall_prob)
            else:
                overall_prob = masked_softmax(logits, dim=1, mask=cam_candidate)
                selected_cam = torch.argmax(overall_prob, dim=1)
                overall_prob_hard = F.one_hot(selected_cam, num_classes=N)
        else:
            cam_pred = cam_emb = None
            selected_cam = torch.ones([B], device=feat.device).long() * override
            overall_prob = overall_prob_hard = F.one_hot(selected_cam, num_classes=N)

        overall_prob = overall_prob_hard if hard is True or not self.training else overall_prob
        if self.aggregation == 'mean':
            overall_feat = (init_feat * init_cam.sum(1) + (feat * overall_prob[:, :, None, None, None]).sum(1)) / (
                    init_cam + overall_prob).sum(1)
        else:
            overall_feat = torch.stack([init_feat, (feat * overall_prob[:, :, None, None, None]).sum(1)], 1).max(1)[0]
        return overall_feat, (cam_emb, cam_pred, overall_prob)


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
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 1), nn.ReLU())
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
        imgs_feat, world_feat = self.get_feat(imgs, M, visualize)
        world_feat, selection_res = self.cam_pred(world_feat, init_cam, keep_cams, hard, override)
        world_res, imgs_res = self.get_output(imgs_feat, world_feat)
        return world_res, imgs_res, selection_res

    def get_feat(self, imgs, M, visualize=False):
        B, N, _, H, W = imgs.shape
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

        # world feat
        H, W = self.Rworld_shape
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), (H, W),
                                      align_corners=False).view(B, N, -1, H, W)

        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # world_feat = self.world_feat_pre(world_feat) * keep_cams.view(B * N, 1, 1, 1).to(imgs.device)

        return imgs_feat, world_feat

    def get_output(self, imgs_feat, world_feat, visualize=False):
        # img heads
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        # world heads
        world_feat = self.world_feat(world_feat)
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

        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)


def test():
    from src.datasets.frameDataset import frameDataset
    from src.datasets.Wildtrack import Wildtrack
    from src.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from src.utils.decode import ctdet_decode
    from thop import profile

    dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train')
    dataloader = DataLoader(dataset, 1, False, num_workers=0)

    model = MVDet(dataset)
    imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = next(iter(dataloader))
    keep_cams[0, 3] = 0
    init_cam = 5
    model.train()
    (world_heatmap, world_offset), _, cam_train = model(imgs, affine_mats, init_cam, keep_cams)
    xysc_train = ctdet_decode(world_heatmap, world_offset)
    model.eval()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_eval = \
        model(imgs, affine_mats, init_cam, override=5)
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh), cam_eval = \
        model(imgs, affine_mats, init_cam, override=5)
    xysc_eval = ctdet_decode(world_heatmap, world_offset)
    # macs, params = profile(model, inputs=(imgs[:, :2], affine_mats))
    # macs, params = profile(model, inputs=(torch.rand(1, 6, 128, 160, 250), affine_mats))
    #
    # print(f'{macs}')
    # print(f'{params}')
    pass


if __name__ == '__main__':
    test()
