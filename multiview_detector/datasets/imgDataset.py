import os
import numpy as np
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import VisionDataset


class imgDataset(VisionDataset):
    classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                  'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                  'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                  'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                  'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(self, root, num_cam, mode='multi', split='train', per_cls_instances=0, dropout=0.0):
        super().__init__(root)
        self.num_cam, self.num_class = num_cam, len(self.classnames)
        assert mode in ['single', 'multi']
        self.mode = mode
        self.transform = T.Compose([T.Resize([224, 224]), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        self.dropout = dropout

        self.img_fpaths = {cam: [] for cam in range(self.num_cam)}
        self.targets = []
        for cls in self.classnames:
            for fname in sorted(glob.glob(f'{root}/{cls}/{split}/*.png')):
                fname = os.path.basename(fname)
                id, cam = map(int, re.findall(r'\d+', fname))
                if id > (per_cls_instances if per_cls_instances else np.inf):
                    break
                self.img_fpaths[cam - 1].append(f'{root}/{cls}/{split}/{fname}')
                if cam == 1:
                    self.targets.append(self.classnames.index(cls))
        assert np.prod([len(i) == len(self.targets) for i in self.img_fpaths.values()]), \
            'plz ensure all models appear {num_cam} times!'
        print(f'{split}: {self.num_class} classes, {num_cam} views, {len(self.targets)} instances')

    def __len__(self):
        return len(self.targets) if self.mode == 'multi' else len(self.targets) * self.num_cam

    def __getitem__(self, idx, visualize=False):
        if self.mode == 'multi':
            imgs = []
            for cam in range(self.num_cam):
                img = Image.open(self.img_fpaths[cam][idx]).convert('RGB')
                if visualize:
                    plt.imshow(img)
                    plt.show()
                imgs.append(self.transform(img))
            imgs = torch.stack(imgs)
            tgt = self.targets[idx]
            # dropout
            drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
            if drop:
                num_drop = np.random.randint(self.num_cam - 1)
                drop_cams = np.random.choice(self.num_cam, num_drop, replace=False)
                for cam in drop_cams:
                    keep_cams[cam] = 0
        elif self.mode == 'single':
            img = Image.open(self.img_fpaths[idx % self.num_cam][idx // self.num_cam]).convert('RGB')
            if visualize:
                plt.imshow(img)
                plt.show()
            imgs = self.transform(img)
            tgt = self.targets[idx // self.num_cam]
            keep_cams = 1
        else:
            raise Exception

        return imgs, tgt, keep_cams


if __name__ == '__main__':
    dataset = imgDataset('/home/houyz/Data/modelnet/modelnet40_images_new_12x', 12, mode='multi')
    dataset.__getitem__(0)
    dataset.__getitem__(len(dataset) - 1, visualize=True)
