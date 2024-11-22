"""
Standard nnunet conversion:
- Copy all images to the training directory
- Convert naming
- splits are generated automatically by nnunet
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset514_MVSeg23"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    parser.add_argument(
        "--use_default_splits", action="store_true", help="Use default train/test split."
    )
    args = parser.parse_args()
    source_dir = Path(args.raw_data_dir)
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir()
    default_split_path = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"

    # copy all images and labels to correct locations
    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    # split into train and test
    case_dict = {}
    for lab_file in (source_dir / "val").glob("*-label.nii.gz"):
        case_id = lab_file.name.removesuffix("-label.nii.gz")
        img_file = source_dir / "val" / f"{case_id}-US.nii.gz"
        assert case_id not in case_dict
        case_dict[case_id] = {"img": img_file, "lab": lab_file}
    for lab_file in (source_dir / "train").glob("*-label.nii.gz"):
        case_id = lab_file.name.removesuffix("-label.nii.gz")
        img_file = source_dir / "train" / f"{case_id}-US.nii.gz"
        assert case_id not in case_dict
        case_dict[case_id] = {"img": img_file, "lab": lab_file}
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        test_cases = [x for x in case_dict if x not in train_cases]
    else:
        random.seed(420000)
        test_cases = random.sample(list(case_dict.keys()), k=55)
        train_cases = list(set(case_dict.keys()) - set(test_cases))
    # copying
    print(f"Copying {len(train_cases)} training cases and {len(test_cases)} test cases...")
    for case_id in tqdm(train_cases):
        shutil.copy(case_dict[case_id]["img"], images_train_dir / f"{case_id}_0000.nii.gz")
        shutil.copy(case_dict[case_id]["lab"], labels_train_dir / f"{case_id}.nii.gz")
    for case_id in tqdm(test_cases):
        shutil.copy(case_dict[case_id]["img"], images_test_dir / f"{case_id}_0000.nii.gz")
        shutil.copy(case_dict[case_id]["lab"], labels_test_dir / f"{case_id}.nii.gz")
    # save only domains of test cases
    domain_mapping = {p: "ID" for p in test_cases}
    save_json(domain_mapping, target_root_dir / "domain_mapping_00.json")

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "3D-US"},
        labels={
            "background": 0,
            "Posterior leaflet": 1,
            "Anterior leaflet": 2,
        },
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        # order has to be this because tumor is the last region (== label 2)
        dataset_name=TASK_NAME,
        description="MVSeg2023",
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
