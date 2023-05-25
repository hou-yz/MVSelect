import os
import re
import json
import time
from operator import itemgetter
from PIL import Image
from kornia.geometry import warp_perspective
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from src.utils.projection import *
from src.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


def read_pom(root):
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r'(\d+) (\d+)')
    cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
    with open(os.path.join(root, 'rectangles.pom'), 'r') as fp:
        for line in fp:
            if 'RECTANGLE' in line:
                cam, pos = map(int, cam_pos_pattern.search(line).groups())
                if pos not in bbox_by_pos_cam:
                    bbox_by_pos_cam[pos] = {}
                if 'notvisible' in line:
                    bbox_by_pos_cam[pos][cam] = None
                else:
                    cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                    bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                 min(right, 1920 - 1), min(bottom, 1080 - 1)]
    return bbox_by_pos_cam


class frameDataset(VisionDataset):
    def __init__(self, base, split='train', reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 split_ratio=(0.8, 0.1, 0.1), top_k=100, force_download=True, dropout=0.0, augmentation=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.dropout = dropout
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        # split = ('train', 'val', 'test'), split_ratio=(0.8, 0.1, 0.1)
        split_ratio = tuple(sum(split_ratio[:i + 1]) for i in range(len(split_ratio)))
        assert split_ratio[-1] == 1
        self.split = split
        if split == 'train':
            frame_range = range(0, int(self.num_frame * split_ratio[0]))
        elif split == 'val':
            frame_range = range(int(self.num_frame * split_ratio[0]), int(self.num_frame * split_ratio[1]))
        elif split == 'trainval':
            frame_range = range(0, int(self.num_frame * split_ratio[1]))
        elif split == 'test':
            frame_range = range(int(self.num_frame * split_ratio[1]), self.num_frame)
        else:
            raise Exception

        self.world_from_img, self.img_from_world = self.get_world_imgs_trans()
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        self.imgs_region = warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest')
        self.Rworld_coverage = self.get_world_coverage().bool()

        self.img_fpaths = self.get_image_fpaths(frame_range)
        self.world_gt, self.imgs_gt, self.pid_dict, self.frames = self.get_gt_targets(
            split if split == 'trainval' else f'{split} \t', frame_range)
        # gt in mot format for evaluation
        self.gt_fname = f'{self.root}/gt'
        if not os.path.exists(f'{self.gt_fname}.txt') or force_download:
            self.prepare_gt()
        pass

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_gt_targets(self, split, frame_range):
        num_world_bbox, num_imgs_bbox = 0, 0
        world_gt = {}
        imgs_gt = {}
        pid_dict = {}
        frames = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                frames.append(frame)
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in pid_dict:
                        pid_dict[pedestrian['personID']] = len(pid_dict)
                    num_world_bbox += 1
                    if self.base.indexing == 'xy':
                        world_pts.append((grid_x, grid_y))
                    else:
                        world_pts.append((grid_y, grid_x))
                    world_pids.append(pid_dict[pedestrian['personID']])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pid_dict[pedestrian['personID']])
                            num_imgs_bbox += 1
                world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))

        print(f'{split}:\t pid: {len(pid_dict)}, frame: {len(frames)}, '
              f'world bbox: {num_world_bbox / len(frames):.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / len(frames) / self.num_cam:.1f}')
        return world_gt, imgs_gt, pid_dict, frames

    def get_world_coverage(self):
        # world grid change to xy indexing
        world_zoom_mat = np.diag([self.world_reduce, self.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            self.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam], )
                                         for cam in range(self.num_cam)]
        # Rworldgrid(xy)_from_imgcoord(xy)
        proj_mats = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord_mat @
                                                  worldcoord_from_imgcoord_mats[cam])
                                 for cam in range(self.num_cam)]).float()

        imgs = torch.ones([self.num_cam, 1, self.base.img_shape[0], self.base.img_shape[1]])
        coverage = warp_perspective(imgs, proj_mats, self.Rworld_shape)
        return coverage

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        # world grid change to xy indexing
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                       self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam],
                                                                          z / self.base.worldcoord_unit)
                                         for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                     for cam in range(self.num_cam)]
        world_from_img = torch.tensor(np.stack(proj_mats))
        # img(xy)_from_worldgrid(xy)
        img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
        return world_from_img.float(), img_from_world.float()

    def prepare_gt(self):
        og_gt = [[] for _ in range(self.num_cam)]
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam, grid_x, grid_y):
                    visible = not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                   single_pedestrian['views'][cam]['xmax'] == -1 and
                                   single_pedestrian['views'][cam]['ymin'] == -1 and
                                   single_pedestrian['views'][cam]['ymax'] == -1)
                    in_view = (single_pedestrian['views'][cam]['xmin'] > 0 and
                               single_pedestrian['views'][cam]['xmax'] < 1920 and
                               single_pedestrian['views'][cam]['ymin'] > 0 and
                               single_pedestrian['views'][cam]['ymax'] < 1080)

                    # Rgrid_x, Rgrid_y = grid_x // self.world_reduce, grid_y // self.world_reduce
                    # in_map = Rgrid_x < self.Rworld_shape[0] and Rgrid_y < self.Rworld_shape[1]
                    return visible and in_view

                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                for cam in range(self.num_cam):
                    if is_in_cam(cam, grid_x, grid_y):
                        og_gt[cam].append(np.array([frame, grid_x, grid_y]))
        og_gt = [np.stack(og_gt[cam], axis=0) for cam in range(self.num_cam)]
        np.savetxt(f'{self.gt_fname}.txt', np.unique(np.concatenate(og_gt, axis=0), axis=0), '%d')
        for cam in range(self.num_cam):
            np.savetxt(f'{self.gt_fname}_{cam}.txt', og_gt[cam], '%d')

    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.world_gt.keys())[index]
        # imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        # inverse_M = torch.inverse(
        #     torch.cat([affine_mats, torch.tensor([0, 0, 1]).view(1, 1, 3).repeat(self.num_cam, 1, 1)], dim=1))[:, :2]
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            num_drop = np.random.randint(self.num_cam - 1)
            drop_cams = np.random.choice(self.num_cam, num_drop, replace=False)
            for cam in drop_cams:
                keep_cams[cam] = 0
                for key in imgs_gt:
                    imgs_gt[key][cam] = 0
        # world gt
        world_pt_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        return imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams

    def __len__(self):
        return len(self.world_gt.keys())


def test(test_projection=False):
    from torch.utils.data import DataLoader
    from src.datasets.wildtrack import Wildtrack
    from src.datasets.multiviewx import MultiviewX

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), force_download=True)
    dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), force_download=True)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='train', semi_supervised=.1)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train', semi_supervised=.1)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='train', semi_supervised=0.5)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train', semi_supervised=0.5)
    min_dist = np.inf
    for world_gt in dataset.world_gt.values():
        x, y = world_gt[0][:, 0], world_gt[0][:, 1]
        if x.size and y.size:
            xy_dists = ((x - x[:, None]) ** 2 + (y - y[:, None]) ** 2) ** 0.5
            np.fill_diagonal(xy_dists, np.inf)
            min_dist = min(min_dist, np.min(xy_dists))
            pass
    dataloader = DataLoader(dataset, 2, True, num_workers=0)
    # imgs, world_gt, imgs_gt, M, frame, keep_cams = next(iter(dataloader))
    t0 = time.time()
    for i in range(10):
        imgs, world_gt, imgs_gt, M, frame, keep_cams = dataset.__getitem__(i, visualize=False)
    print(time.time() - t0)

    pass
    if test_projection:
        import matplotlib.pyplot as plt
        from src.utils.projection import get_worldcoord_from_imagecoord
        world_grid_maps = []
        xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
        for cam in range(dataset.num_cam):
            world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(),
                                                          dataset.base.intrinsic_matrices[cam],
                                                          dataset.base.extrinsic_matrices[cam])
            world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
            world_grid_map = np.zeros(dataset.worldgrid_shape)
            for i in range(H):
                for j in range(W):
                    x, y = world_grids[i, j]
                    if dataset.base.indexing == 'xy':
                        if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                            world_grid_map[int(y), int(x)] += 1
                    else:
                        if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                            world_grid_map[int(x), int(y)] += 1
            world_grid_map = world_grid_map != 0
            plt.imshow(world_grid_map)
            plt.show()
            world_grid_maps.append(world_grid_map)
            pass
        plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
        plt.show()
        pass


if __name__ == '__main__':
    test(False)
