import numpy as np
from src.evaluation.pyeval.CLEAR_MOD_HUN import CLEAR_MOD_HUN


def evaluateDetection_py(det, gt, frames=None):
    """
    This is simply the python translation of a MATLAB　Evaluation tool created by P. Dollar.
    Translated by Zicheng Duan
    Modified by Yunzhong Hou

    The purpose of this API:
    1. To allow the project to run purely in Python without using MATLAB Engine.

    @param det: detection result file path
    @param gt: ground truth result file path
    @return: MODP, MODA, recall, precision
    """

    if isinstance(gt, str):
        gt = np.loadtxt(gt)
    else:
        gt = np.array(gt)
    if isinstance(det, str):
        det = np.loadtxt(det)
    else:
        det = np.array(det)
    if det.shape == (3,):
        det = det[None, :]

    if frames is not None:
        gt = gt[np.isin(gt[:, 0], frames), :]

    MODA, MODP, precision, recall, (tp, fp, fn, gt, dist) = CLEAR_MOD_HUN(gt, det)
    return MODA, MODP, precision, recall, (tp, fp, fn, gt, dist)


if __name__ == "__main__":
    res_fpath = "../test-demo.txt"
    gt_fpath = "../gt-demo.txt"
    moda, modp, precision, recall, stats = evaluateDetection_py(res_fpath, gt_fpath, np.arange(1800, 2000))
    print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
