import numpy as np
from scipy.optimize import linear_sum_assignment


def CLEAR_MOD_HUN(gt, det, dist_thres=50 / 2.5):
    frames, num_gt_per_frame = np.unique(gt[:, 0], return_counts=True)
    matches = -np.ones([len(frames), np.max(num_gt_per_frame)])  # matching result for each GT target in each frame
    num_matches = np.zeros([len(frames)])  # c in original code
    fp = np.zeros([len(frames)])
    fn = np.zeros([len(frames)])  # m in original code
    distances = np.inf * np.ones([len(frames), np.max(num_gt_per_frame)])

    for frame_idx, t in enumerate(frames):
        gt_idx, = np.where(gt[:, 0] == t)
        det_idx, = np.where(det[:, 0] == t)

        if gt_idx is not None and det_idx is not None:
            dist = np.linalg.norm(gt[gt_idx, 1:][:, None, :] - det[det_idx, 1:][None, :, :], axis=2)

            # Please notice that the price/distance of are set to 100000 instead of np.inf,
            # since the Hungarian Algorithm implemented in sklearn will be slow if we use np.inf.
            dist[dist > dist_thres] = 1e6
            HUN_res = np.array(linear_sum_assignment(dist))
            # filter out true matches
            HUN_res = HUN_res[:, dist[HUN_res[0], HUN_res[1]] < dist_thres]
            matches[frame_idx, HUN_res[0]] = HUN_res[1]
            distances[frame_idx, HUN_res[0]] = dist[HUN_res[0], HUN_res[1]]

        num_matches[frame_idx] = (matches[frame_idx, :] != -1).sum()
        fp[frame_idx] = len(det_idx) - num_matches[frame_idx]
        fn[frame_idx] = num_gt_per_frame[frame_idx] - num_matches[frame_idx]

    MODA = (1 - ((np.sum(fn) + np.sum(fp)) / np.sum(num_gt_per_frame))) * 100
    MODP = sum(1 - distances[distances < dist_thres] / dist_thres) / (np.sum(num_matches) + 1e-8) * 100
    precision = np.sum(num_matches) / (np.sum(fp) + np.sum(num_matches) + 1e-8) * 100
    recall = np.sum(num_matches) / np.sum(num_gt_per_frame) * 100

    return MODA, MODP, precision, recall, (num_matches, fp, fn, num_gt_per_frame, distances)
