import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric


class SurfaceDiceEmptyHandlingMetric(SurfaceDiceMetric):
    # I think MONAI's class_thresholds argument documentation is wrong; it has units of mm (same as spacing)
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # tensor shapes BCHW[D]
        result_sd = super()._compute_tensor(y_pred, y, **kwargs)
        # handle empty GT case
        empty_gt = y.sum(dim=list(range(2, y_pred.ndim))) == 0
        empty_pred = y_pred.sum(dim=list(range(2, y_pred.ndim))) == 0
        if not self.include_background:
            # remove the first class
            empty_gt = empty_gt[:, 1:]
            empty_pred = empty_pred[:, 1:]
        empty_both = torch.logical_and(empty_gt, empty_pred)
        empty_gt_only = torch.logical_and(empty_gt, torch.logical_not(empty_pred))
        result_sd[empty_both] = 1.0
        result_sd[empty_gt_only] = 0.0
        return result_sd


# same for hausdorff distance
class HausdorffDistanceEmptyHandlingMetric(HausdorffDistanceMetric):
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # tensor shapes BCHW[D]
        result_hd = super()._compute_tensor(y_pred, y, **kwargs)
        # handle empty GT case
        empty_gt = y.sum(dim=list(range(2, y_pred.ndim))) == 0
        empty_pred = y_pred.sum(dim=list(range(2, y_pred.ndim))) == 0
        if not self.include_background:
            # remove the first class
            empty_gt = empty_gt[:, 1:]
            empty_pred = empty_pred[:, 1:]
        empty_both = torch.logical_and(empty_gt, empty_pred)
        empty_gt_only = torch.logical_and(empty_gt, torch.logical_not(empty_pred))
        result_hd[empty_both] = 0.0
        result_hd[empty_gt_only] = 0.5 * np.linalg.norm(y_pred.shape[2:])
        # This is arbitrary: I use half the diagonal as the worst HD value.
        return result_hd
