import numpy as np
from multiview_detector.evaluation.pyeval.CLEAR_MOD_HUN import CLEAR_MOD_HUN


def evaluateDetection_py(res_fpath, gt_fpath):
    """
    This is simply the python translation of a MATLAB　Evaluation tool created by P. Dollar.
    Translated by Zicheng Duan
    Modified by Yunzhong Hou

    The purpose of this API:
    1. To allow the project to run purely in Python without using MATLAB Engine.

    @param res_fpath: detection result file path
    @param gt_fpath: ground truth result file path
    @return: MODP, MODA, recall, precision
    """

    gt_raw = np.loadtxt(gt_fpath)
    det_raw = np.loadtxt(res_fpath)
    if det_raw.shape == (3,):
        det_raw = det_raw[None, :]
    frames = np.unique(det_raw[:, 0]) if det_raw.size else np.zeros(0)

    gt_formatted = []
    det_formatted = []
    if det_raw is None or det_raw.shape[0] == 0:
        MODA, MODP, precision, recall = 0, 0, 0, 0
        return recall, precision, MODA, MODP

    for frame_ctr, t in enumerate(frames):
        gt_in_frame = gt_raw[gt_raw[:, 0] == t, 1:]
        gt_in_frame = np.concatenate([np.ones([len(gt_in_frame), 1]) * frame_ctr,
                                      np.arange(len(gt_in_frame))[:, None],
                                      gt_in_frame], axis=1)
        gt_formatted.append(gt_in_frame)

        det_in_frame = det_raw[det_raw[:, 0] == t, 1:]
        det_in_frame = np.concatenate([np.ones([len(det_in_frame), 1]) * frame_ctr,
                                       np.arange(len(det_in_frame))[:, None],
                                       det_in_frame], axis=1)
        det_formatted.append(det_in_frame)
    gt_formatted = np.concatenate(gt_formatted)
    det_formatted = np.concatenate(det_formatted)
    recall, precision, MODA, MODP = CLEAR_MOD_HUN(gt_formatted, det_formatted)
    return recall, precision, MODA, MODP


if __name__ == "__main__":
    res_fpath = "/home/houyz/Code/MVselect/logs/multiviewx/resnet18_max_lr0.0005_b2_e10_dropcam0.0_2022-11-09_12-01-22/test.txt"
    gt_fpath = "/home/houyz/Data/MultiviewX/gt.txt"
    recall, precision, moda, modp = evaluateDetection_py(res_fpath, gt_fpath)
    print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
