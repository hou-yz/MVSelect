import numpy as np
import torch
import os
from src.datasets import frameDataset, MultiviewX, Wildtrack
from src.evaluation.evaluate import evaluate
import matplotlib.pyplot as plt


def show_coverage(base):
    dataset = frameDataset(base, split='test', )
    N = dataset.num_cam
    gts = [np.loadtxt(f'{base.root}/gt_{cam}.txt') for cam in range(N)]
    cover_area = np.zeros([N, N])
    performances = np.zeros([N, N, 4])
    recall, precision, moda, modp = evaluate(f'{base.root}/gt.txt', f'{base.root}/gt.txt')
    for init_cam in range(N):
        for selected_cam in range(N):
            cover_area[init_cam, selected_cam] = (dataset.Rworld_coverage[init_cam] +
                                                  dataset.Rworld_coverage[selected_cam]).bool().float().mean()
            gt_in_cam = np.unique(np.concatenate([gts[init_cam], gts[selected_cam]]), axis=0)
            gt_in_cam = gt_in_cam[gt_in_cam[:, 0] > dataset.num_frame * 0.9]
            np.savetxt('temp.txt', gt_in_cam)
            recall, precision, moda, modp = evaluate('temp.txt', f'{base.root}/gt.txt')
            performances[init_cam, selected_cam] = [moda, modp, precision, recall]

    pass

    plt.figure(figsize=(5, 5))
    plt.imshow(cover_area, cmap='Blues')
    plt.xticks(np.arange(N), np.arange(N) + 1)
    plt.xlabel('second view')
    plt.yticks(np.arange(N), np.arange(N) + 1)
    plt.ylabel('initial view')
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            plt.text(j, i, f'{cover_area[i, j]:.2f}', ha="center", va="center", )
    plt.tight_layout()
    plt.show()

    print(cover_area.max(axis=0).mean())

    plt.figure(figsize=(5, 5))
    plt.imshow(performances[:, :, 0], cmap='Blues')
    plt.xticks(np.arange(N), np.arange(N) + 1)
    plt.xlabel('second view')
    plt.yticks(np.arange(N), np.arange(N) + 1)
    plt.ylabel('initial view')
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            plt.text(j, i, f'{performances[i, j, 0]:.1f}', ha="center", va="center", )
    plt.tight_layout()
    plt.show()

    print(performances[:, :, 0].max(axis=0).mean())


if __name__ == '__main__':
    show_coverage(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    show_coverage(MultiviewX(os.path.expanduser('~/Data/MultiviewX')))
