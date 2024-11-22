from typing import Union

import monai.transforms as trf
import numpy as np
import torch


def convert_to_onehot(
    segmentation: np.ndarray | torch.Tensor, all_labels_or_regions: Union[int, tuple[int, ...]]
) -> np.ndarray | torch.Tensor:
    if isinstance(segmentation, np.ndarray):
        return convert_to_onehot_np(segmentation, all_labels_or_regions)
    else:
        return convert_to_onehot_torch(segmentation, all_labels_or_regions)


def convert_to_onehot_np(
    segmentation: np.ndarray, all_labels_or_regions: Union[int, tuple[int, ...]]
) -> np.ndarray:
    # assume shape HW[D]
    if isinstance(all_labels_or_regions, int):
        all_labels_or_regions = list(range(all_labels_or_regions))
    result = np.zeros((len(all_labels_or_regions), *segmentation.shape), dtype=np.uint8)
    for i, l in enumerate(all_labels_or_regions):
        if np.isscalar(l):
            result[i] = segmentation == l
        else:
            result[i] = np.isin(segmentation, l)
    return result


def convert_to_onehot_torch(
    segmentation: torch.Tensor, all_labels_or_regions: Union[int, tuple[int, ...]]
) -> torch.Tensor:
    return convert_to_onehot_batch(segmentation.unsqueeze(0), all_labels_or_regions).squeeze(0)


def convert_to_onehot_batch(
    segmentation: torch.Tensor, all_labels_or_regions: Union[int, tuple[int, ...]]
) -> torch.Tensor:
    # assume shape B1HW[D]
    batch_size = segmentation.shape[0]
    assert segmentation.shape[1] == 1
    if isinstance(all_labels_or_regions, int):
        all_labels_or_regions = list(range(all_labels_or_regions))
    result = torch.zeros(
        (batch_size, len(all_labels_or_regions), *segmentation.shape[2:]),
        dtype=segmentation.dtype,
        device=segmentation.device,
    )
    for i, l in enumerate(all_labels_or_regions):
        if np.isscalar(l):
            result[:, i] = segmentation[:, 0] == l
        else:
            result[:, i] = torch.isin(
                segmentation[:, 0], torch.tensor(l, device=segmentation.device)
            )
    return result


def convert_nnunet_regions_to_labels(region_map, region_class_order: list):
    # assume shape B,C,*spatial
    if isinstance(region_map, np.ndarray):
        assert region_map.dtype == bool
        label_map = np.zeros((region_map.shape[0], *region_map.shape[2:]), dtype=np.uint16)
    else:
        assert region_map.dtype == torch.bool
        # no uint16 in torch
        label_map = torch.zeros(
            (region_map.shape[0], *region_map.shape[2:]),
            dtype=torch.int16,
            device=region_map.device,
        )
    for i, c in enumerate(region_class_order):
        label_map[region_map[:, i]] = c
    return label_map


def discretize_softmax(probs: np.ndarray, overlapping_classes: bool) -> np.ndarray:
    """
    This is a helper function to convert logits to a discrete segmentation.

    Args:
        logits: (C, H, W[, D]) array of logits
        overlapping_classes: whether the logits are for overlapping classes or not
    """
    if overlapping_classes:
        return (probs > 0.5).astype(np.uint8)
    else:
        return np.argmax(probs, axis=0).astype(np.uint8)


class ConvertSegToRegions(trf.MapTransform):
    def __init__(self, keys, class_or_regions_defs, include_background=True):
        super().__init__(keys)
        self.class_or_regions_defs = class_or_regions_defs
        if not include_background:
            self.class_or_regions_defs = {
                k: v
                for k, v in class_or_regions_defs.items()
                if k.lower() not in ["background", "bg"]
            }

    def __call__(self, data):
        for key in self.keys:
            # assume shape 1HW[D]
            assert data[key].shape[0] == 1
            data[key] = convert_to_onehot(data[key], list(self.class_or_regions_defs.values()))
        return data
