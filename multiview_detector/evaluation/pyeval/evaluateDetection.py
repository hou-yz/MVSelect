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

    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)
    if detRaw.shape == (3,):
        detRaw = detRaw[None, :]
    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0
    if detRaw is None or detRaw.shape[0] == 0:
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for _ in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for _ in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1
    recall, precision, MODA, MODP = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix)
    return recall, precision, MODA, MODP


if __name__ == "__main__":
    res_fpath = "/home/houyz/Code/MVselect/logs/wildtrack/S_lr0.0005_2022-06-12_01-02-58/test.txt"
    gt_fpath = "/home/houyz/Data/Wildtrack/gt.txt"
    dataset_name = "Wildtrack"
    recall, precision, moda, modp = evaluateDetection_py(res_fpath, gt_fpath)
    print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
