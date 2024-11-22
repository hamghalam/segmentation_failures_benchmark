import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from segmentation_failures.utils.io import load_json, save_json


def get_padding(input_shape, output_shape):
    padding = []
    for in_dim, out_dim in zip(input_shape, output_shape):
        curr_pad = out_dim - in_dim
        if curr_pad > 0:
            padding.extend([curr_pad // 2, curr_pad // 2 + curr_pad % 2])
        else:
            padding.extend([0, 0])
    return padding


def get_dataset_dir(dataset_id, data_root_dir: str = None) -> Path:
    if data_root_dir is None:
        data_root_dir = os.environ["TESTDATA_ROOT_DIR"]
    candidates = list(Path(data_root_dir).glob(f"Dataset{dataset_id}_*"))
    if len(candidates) == 0:
        raise ValueError(f"Could not find dataset {dataset_id} in {data_root_dir}")
    elif len(candidates) > 1:
        raise ValueError(
            f"Found multiple datasets matching {dataset_id} in {data_root_dir}: {candidates}"
        )
    return candidates[0]


def load_dataset_json(dataset_id, data_root_dir: str = None):
    dataset_dir = get_dataset_dir(dataset_id, data_root_dir)
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Could not find dataset.json in {dataset_dir}")
    return load_json(dataset_json_path)


def make_centered_fg_bbox(mask_arr: torch.Tensor, bbox_size: list[int], center_of_mass=False):
    """
    Generate a bounding box mask centered around the foreground in a given mask array.

    Args:
        mask_arr (torch.Tensor): The input mask array.
        bbox_size (list[int]): The size of the bounding box in each dimension.
        center_of_mass (bool, optional): Whether to compute the center of mass of the foreground.
            If False, the center of the bounding box will be used. Defaults to False.

    Returns:
        torch.Tensor: The generated bounding box mask.

    Note:
        - If there is no foreground in the mask, the function returns a mask with all ones (the whole image).
        - The bounding box is computed based on the center of the foreground or the center of mass (if specified).
        - The function ensures that the bounding box is fully inside the image.
        - bbox_size is disregarded if the mask is larger than bbox_size in any dimension.

    """
    spatial_size = mask_arr.shape[1:]
    nonzero_list = torch.argwhere(torch.any(mask_arr > 0, dim=0))
    # compute the bbox around the center of the foreground
    # (not center of mass, just the center of the bounding box)
    if len(nonzero_list) == 0:
        # if there is no foreground, just return the whole image
        return torch.ones_like(mask_arr)
    mask_lb = torch.amin(nonzero_list, dim=0).to(int)
    mask_ub = torch.amax(nonzero_list, dim=0).to(int)
    if center_of_mass:
        # compute the center of mass of the foreground
        center_point = nonzero_list.to(float).mean(dim=0).to(int)
    else:
        center_point = (0.5 * (mask_lb + mask_ub)).to(int)
    bbox_slices = []
    for dim_idx in range(len(center_point)):
        if mask_ub[dim_idx] + 1 - mask_lb[dim_idx] < bbox_size[dim_idx]:
            # if the mask is smaller than the bbox, center the bbox around the mask
            lb_proposal = int(center_point[dim_idx] - bbox_size[dim_idx] // 2)
            ub_proposal = int(
                center_point[dim_idx] + bbox_size[dim_idx] // 2 + (bbox_size[dim_idx] % 2)
            )
            # make sure the bbox is inside the image
            if ub_proposal > spatial_size[dim_idx]:
                # shift to the left
                lb_proposal = max(lb_proposal - (ub_proposal - spatial_size[dim_idx]), 0)
                ub_proposal = spatial_size[dim_idx]
            if lb_proposal < 0:
                # shift to the right
                ub_proposal = min(spatial_size[dim_idx], ub_proposal - lb_proposal)
                lb_proposal = 0
        else:
            lb_proposal = mask_lb[dim_idx]
            ub_proposal = mask_ub[dim_idx] + 1
        bbox_slices.append(slice(lb_proposal, ub_proposal))
    # generate a mask that is 1 inside the bbox and 0 outside
    bbox_slices = (slice(None), *bbox_slices)
    bbox_mask = torch.zeros_like(mask_arr)
    bbox_mask[bbox_slices] = 1
    # check if the mask is fully inside the bbox
    return bbox_mask


def kfold_split(data_dir: str, num_folds: int = 5, output_path: str = None, seed=42):
    data_path = Path(data_dir)
    assert data_path.exists(), data_path
    if output_path is not None:
        output_path = Path(output_path)

    # read list of cases
    case_list = set()
    for img_path in (data_path / "imagesTr").iterdir():
        case_id = "_".join(img_path.name.split("_")[:-1])
        case_list.add(case_id)
    case_list = list(case_list)
    # perform case-wise split
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_splits = []
    for k, (train_idx, valid_idx) in enumerate(kfold.split(case_list)):
        all_splits.append(
            {
                "train": np.array(case_list)[train_idx].tolist(),
                "val": np.array(case_list)[valid_idx].tolist(),
            }
        )
        if output_path is not None:
            save_json(output_path / f"fold_{k}.json")
    return all_splits


def invert_existing_split(split_file_path: str):
    original_splits = load_json(split_file_path)
    inverted_splits = []
    for split in original_splits:
        inverted_splits.append({"train": split["val"], "val": split["train"]})
    split_file_path = Path(split_file_path)
    new_name = split_file_path.stem + "_inverted.json"
    save_json(inverted_splits, split_file_path.parent / new_name)


def check_splits(splits_path: str):
    val_splits = []
    for split in load_json(splits_path):
        val_splits.append(set(split["val"]))
    print("Size of validation splits:")
    for i in range(len(val_splits)):
        print(len(val_splits[i]))
        assert len(val_splits[i - 1].intersection(val_splits[i])) == 0
