from typing import Optional, Union

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric as MonaiHDMetric
from monai.metrics.utils import (
    get_mask_edges,
    get_surface_distance,
    ignore_background,
    is_binary_tensor,
)


class HausdorffDistanceMetric(MonaiHDMetric):
    """
    The only difference to MONAI's implementation is that I set fixed values for the empty GT/pred cases.
    """

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
        )


def compute_hausdorff_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
    directed: bool = False,
):
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    if isinstance(y, torch.Tensor):
        y = y.float()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(
            f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}."
        )

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class))
    # This is arbitrary: I use half the diagonal as the worst HD value.
    WORST_VAL = 0.5 * np.linalg.norm(y_pred.shape[2:])
    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        if not np.any(edges_gt) and not np.any(edges_pred):
            # empty gt and empty pred => perfect score 0
            hd[b, c] = 0
        elif not np.any(edges_gt) or not np.any(edges_pred):
            # one of them is zero => worst score.
            hd[b, c] = WORST_VAL
        else:
            distance_1 = compute_percent_hausdorff_distance(
                edges_pred, edges_gt, distance_metric, percentile
            )
            if directed:
                hd[b, c] = distance_1
            else:
                distance_2 = compute_percent_hausdorff_distance(
                    edges_gt, edges_pred, distance_metric, percentile
                )
                hd[b, c] = max(distance_1, distance_2)
    return torch.from_numpy(hd)


def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
):
    """
    This function is used to compute the directed Hausdorff distance.
    """

    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric)

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return np.nan

    if not percentile:
        return surface_distance.max()

    if 0 <= percentile <= 100:
        return np.percentile(surface_distance, percentile)
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
