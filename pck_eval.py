from typing import Dict
import numpy as np
import json
import os.path as osp

from mmpose_eval import decode_heatmap, compute_metrics


class PckEvaluator:
    def __init__(self, ground_truth_dir: str, model_output_details, limit: int = 0):
        self.model_output_details = model_output_details
        self.limit = limit
        self.gt_keypoints = np.load(osp.join(ground_truth_dir, "gt_keypoints.npy"))
        self.num_keypoints = self.gt_keypoints.shape[1]
        self.gt_keypoints_visible_mask = np.load(
            osp.join(ground_truth_dir, "gt_keypoints_visible_mask.npy")
        )
        self.gt_head_sizes = np.load(osp.join(ground_truth_dir, "gt_head_sizes.npy"))
        with open(osp.join(ground_truth_dir, "meta_infos.json"), "rt") as f:
            self.meta_infos = json.load(f)
        self.pred_keypoints = []
        if limit > 0:
            self.gt_keypoints = self.gt_keypoints[:limit]
            self.gt_keypoints_visible_mask = self.gt_keypoints_visible_mask[:limit]
            self.gt_head_sizes = self.gt_head_sizes[:limit]
            self.meta_infos = self.meta_infos[:limit]

    def process(self, model_output: np.ndarray):
        if model_output.ndim != 4:
            raise ValueError(
                f"Expected 4D output, but got {model_output.ndim}D output."
            )
        if model_output.shape[0] != 1:
            raise ValueError(
                f"Expected batch size of 1, but got {model_output.shape[0]}."
            )
        if model_output.shape[3] == 16:
            model_output = np.transpose(model_output, (0, 3, 1, 2))
        if model_output.shape[1] != self.num_keypoints:
            raise ValueError(
                f"Expected {self.num_keypoints} keypoints, but got {model_output.shape[1]}."
            )
        if model_output.dtype == np.int8:
            output_scale, output_zero_point = self.model_output_details["quantization"]
            model_output = output_scale * (
                model_output.astype(np.float32) - output_zero_point
            )
        self._process(model_output)

    def _process(self, model_output: np.ndarray):
        meta_info = self.meta_infos[len(self.pred_keypoints)]
        pred_heatmap = model_output[0]
        pred_keypoints, pred_scores = decode_heatmap(
            pred_heatmap,
            input_size=(256, 256),
            heatmap_size=(64, 64),
        )
        input_center = np.array(meta_info["input_center"], dtype=np.float32)
        input_scale = np.array(meta_info["input_scale"], dtype=np.float32)
        input_size = np.array(meta_info["input_size"], dtype=np.float32)

        pred_keypoints[..., :2] = (
            pred_keypoints[..., :2] / input_size * input_scale
            + input_center
            - 0.5 * input_scale
        )
        pred_keypoints = pred_keypoints + 1.0  # required for MPII dataset
        self.pred_keypoints.append(pred_keypoints)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model predictions against the ground truth.
        Returns:
            Dict[str: float]: A dictionary containing the evaluation metrics.
        """
        if len(self.pred_keypoints) != len(self.gt_keypoints):
            raise ValueError(
                f"Number of predictions {len(self.pred_keypoints)} does not match number of ground truths {len(self.gt_keypoints)}."
            )

        pred_keypoints = np.concatenate(self.pred_keypoints, axis=0)
        metrics = compute_metrics(
            pred_keypoints,
            self.gt_keypoints,
            self.gt_keypoints_visible_mask,
            self.gt_head_sizes,
        )

        return metrics
