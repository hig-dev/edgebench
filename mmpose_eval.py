# Copyright (c) OpenMMLab. All rights reserved.
# This file contains code from mmpose, which is licensed under the Apache License 2.0.

from typing import Dict, Tuple
import numpy as np
from itertools import product


def decode_heatmap(
    encoded: np.ndarray,
    input_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode keypoint coordinates from heatmaps. The decoded keypoint
    coordinates are in the input image space.

    Args:
        encoded (np.ndarray): Heatmaps in shape (K, H, W)

    Returns:
        tuple:
        - keypoints (np.ndarray): Decoded keypoint coordinates in shape
            (N, K, D)
        - scores (np.ndarray): The keypoint scores in shape (N, K). It
            usually represents the confidence of the keypoint prediction
    """
    heatmaps = encoded.copy()
    K, H, W = heatmaps.shape

    keypoints, scores = get_heatmap_maximum(heatmaps)

    # Unsqueeze the instance dimension for single-instance results
    keypoints, scores = keypoints[None], scores[None]

    keypoints = refine_keypoints(keypoints, heatmaps)

    # Restore the keypoint scale
    scale_factor = (np.array(input_size) / heatmap_size).astype(np.float32)

    keypoints = keypoints * scale_factor

    return keypoints, scores


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f"Invalid shape {heatmaps.shape}"

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints(keypoints: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Refine keypoint predictions by moving from the maximum towards the
    second maximum by 0.25 pixel. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)

        if 1 < x < W - 1 and 0 < y < H:
            dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1]
        else:
            dx = 0.0

        if 1 < y < H - 1 and 0 < x < W:
            dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x]
        else:
            dy = 0.0

        keypoints[n, k] += np.sign([dx, dy], dtype=np.float32) * 0.25

    return keypoints


def _calc_distances(
    preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray
) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1
    )
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    thr: float,
    norm_factor: np.ndarray,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray
) -> Dict[str, float]:
    """Calculates PCK metrics for the predicted keypoints.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (float): Normalization factor.
        thr (float): Threshold of PCK calculation.

    Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
            If `'head'` in `self.norm_item`, the returned results are the pck
            accuracy normalized by `head_size`, which have the following keys:
                - 'Head PCK': The PCK of head
                - 'Shoulder PCK': The PCK of shoulder
                - 'Elbow PCK': The PCK of elbow
                - 'Wrist PCK': The PCK of wrist
                - 'Hip PCK': The PCK of hip
                - 'Knee PCK': The PCK of knee
                - 'Ankle PCK': The PCK of ankle
                - 'PCK': The mean PCK over all keypoints
                - 'PCK@0.1': The mean PCK at threshold 0.1
                - 'PCK-AUC': The area under the curve of PCK

    """
    pck_p, _, _ = keypoint_pck_accuracy(pred, gt, mask, 0.5, norm_factor)

    jnt_count = np.sum(mask, axis=0)
    PCKh = 100.0 * pck_p

    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 16), dtype=np.float32)

    for r, threshold in enumerate(rng):
        _pck, _, _ = keypoint_pck_accuracy(pred, gt, mask, threshold, norm_factor)
        pckAll[r, :] = 100.0 * _pck

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    # dataset_joints_idx:
    #   head 9
    #   lsho 13  rsho 12
    #   lelb 14  relb 11
    #   lwri 15  rwri 10
    #   lhip 3   rhip 2
    #   lkne 4   rkne 1
    #   lank 5   rank 0
    stats = {
        "Head PCK": PCKh[9],
        "Shoulder PCK": 0.5 * (PCKh[13] + PCKh[12]),
        "Elbow PCK": 0.5 * (PCKh[14] + PCKh[11]),
        "Wrist PCK": 0.5 * (PCKh[15] + PCKh[10]),
        "Hip PCK": 0.5 * (PCKh[3] + PCKh[2]),
        "Knee PCK": 0.5 * (PCKh[4] + PCKh[1]),
        "Ankle PCK": 0.5 * (PCKh[5] + PCKh[0]),
        "PCK": np.sum(PCKh * jnt_ratio),
        "PCK@0.1": np.sum(pckAll[10, :] * jnt_ratio),
        "PCK-AUC": np.sum(np.trapz(pckAll, x=rng, axis=0) * jnt_ratio),
    }
    return stats
