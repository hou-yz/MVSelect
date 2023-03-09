import numpy as np
import torch
import os
from src.datasets import frameDataset, MultiviewX, Wildtrack, imgDataset
from src.utils.image_utils import img_color_denormalize
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def set_border(img, width=5, fill=(0, 255, 0)):
    C, H, W = img.shape
    fill = torch.tensor(fill, dtype=img.dtype, device=img.device)[:, None, None] / 255
    img[:, :, :width] = fill
    img[:, :, -width:] = fill
    img[:, :width, :] = fill
    img[:, -width:, :] = fill
    return img


if __name__ == '__main__':
    denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='test', )
    # imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = dataset[0]

    # dataset = imgDataset(os.path.expanduser('~/Data/modelnet/modelnet40v1png'), 12, split='test')
    dataset = imgDataset(os.path.expanduser('~/Data/modelnet/modelnet40v2png_ori4'), 20, split='test')
    for i in range(30):
        # index=np.random.randint(len(dataset))
        index = i
        imgs, tgt, keep_cams = dataset[index+100]
        imgs = denorm(imgs)
        # imgs[0] = set_border(imgs[0])  # , fill=(255, 192, 0)
        # imgs[9] = set_border(imgs[9])  # , fill=(0, 176, 80)
        # imgs[0] = set_border(imgs[0], width=20)  # , fill=(255, 192, 0)
        # imgs[1] = set_border(imgs[1], width=20)  # , fill=(0, 176, 80)
        # imgs[5] = set_border(imgs[5], width=20)  # , fill=(0, 176, 80)

        imgs_grid = make_grid(imgs, nrow=5)
        # save_image(imgs_grid, 'imgs_grid.png')
        plt.imshow(imgs_grid.permute([1, 2, 0]))
        plt.show()
    pass
