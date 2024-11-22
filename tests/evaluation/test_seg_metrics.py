# test cases taken from deepmind's repo
import torch

from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics,
)


def test_distance_metrics_cube():
    sd_metric = get_metrics("surface_dice", class_thresholds=[1], include_background=True)[
        "surface_dice"
    ]
    hd95_metric = get_metrics("hausdorff95", include_background=True)["hausdorff95"]
    mask_gt = torch.zeros(100, 100, 100, dtype=float)
    mask_pred = torch.zeros(100, 100, 100, dtype=float)
    mask_gt[0:50, :, :] = 1
    mask_pred[0:51, :, :] = 1
    mask_gt = mask_gt.reshape((1, 1, *mask_gt.shape))
    mask_pred = mask_pred.reshape((1, 1, *mask_pred.shape))
    surf_dice = sd_metric(mask_pred, mask_gt, spacing=(2, 1, 1))
    hd95 = hd95_metric(mask_pred, mask_gt, spacing=(2, 1, 1))
    assert round(surf_dice.item(), 3) == 0.836
    assert hd95 == 2.0


def test_distance_metrics_two_points():
    sd_metric = get_metrics("surface_dice", class_thresholds=[1], include_background=True)[
        "surface_dice"
    ]
    hd95_metric = get_metrics("hausdorff95", include_background=True)["hausdorff95"]
    mask_gt = torch.zeros(100, 100, 100, dtype=float)
    mask_pred = torch.zeros(100, 100, 100, dtype=float)
    mask_gt[50, 60, 70] = 1
    mask_pred[50, 60, 72] = 1
    mask_gt = mask_gt.reshape((1, 1, *mask_gt.shape))
    mask_pred = mask_pred.reshape((1, 1, *mask_pred.shape))
    surf_dice = sd_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    hd95 = hd95_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    assert surf_dice.item() == 0.5
    assert hd95.item() == 2


def test_distance_metrics_empty_gt():
    sd_metric = get_metrics("surface_dice", class_thresholds=[1], include_background=True)[
        "surface_dice"
    ]
    hd95_metric = get_metrics("hausdorff95", include_background=True)["hausdorff95"]
    mask_gt = torch.zeros(100, 100, 100, dtype=float)
    mask_pred = torch.zeros(100, 100, 100, dtype=float)
    # mask_gt[50, 60, 70] = 1  # same happens for empty pred
    mask_pred[50, 60, 72] = 1
    mask_gt = mask_gt.reshape((1, 1, *mask_gt.shape))
    mask_pred = mask_pred.reshape((1, 1, *mask_pred.shape))
    surf_dice = sd_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    hd95 = hd95_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    assert surf_dice.item() == 0.0
    assert not torch.isnan(hd95)


def test_distance_metrics_empty_both():
    sd_metric = get_metrics("surface_dice", class_thresholds=[1], include_background=True)[
        "surface_dice"
    ]
    hd95_metric = get_metrics("hausdorff95", include_background=True)["hausdorff95"]
    mask_gt = torch.zeros(100, 100, 100, dtype=float)
    mask_pred = torch.zeros(100, 100, 100, dtype=float)
    mask_gt = mask_gt.reshape((1, 1, *mask_gt.shape))
    mask_pred = mask_pred.reshape((1, 1, *mask_pred.shape))
    surf_dice = sd_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    hd95 = hd95_metric(mask_pred, mask_gt, spacing=(3, 2, 1))
    assert surf_dice.item() == 1.0
    assert hd95.item() == 0.0
