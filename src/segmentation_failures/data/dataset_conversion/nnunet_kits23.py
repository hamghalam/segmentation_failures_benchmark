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

TASK_NAME = "Dataset515_KiTS23"


def copy_case(
    case_dir: Path,
    image_target_dir: Path,
    label_target_dir: Path,
):
    assert case_dir.is_dir()
    case_id = case_dir.name
    shutil.copy(case_dir / "imaging.nii.gz", image_target_dir / f"{case_id}_0000.nii.gz")
    shutil.copy(case_dir / "segmentation.nii.gz", label_target_dir / f"{case_id}.nii.gz")


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
    print("Copying cases...")
    # random split into train and test
    case_ids = [x.name for x in source_dir.iterdir() if x.is_dir()]
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        test_cases = [x for x in case_ids if x not in train_cases]
    else:
        random.seed(420000)
        test_cases = random.sample(case_ids, k=int(len(case_ids) * 0.25))
        train_cases = list(set(case_ids) - set(test_cases))
    for case in tqdm(train_cases):
        case_dir = source_dir / case
        if case_dir.is_dir():
            copy_case(case_dir, images_train_dir, labels_train_dir)
    for case in tqdm(test_cases):
        case_dir = source_dir / case
        if case_dir.is_dir():
            copy_case(case_dir, images_test_dir, labels_test_dir)
    # save only domains of test cases
    domain_mapping = {p: "ID" for p in test_cases}
    save_json(domain_mapping, target_root_dir / "domain_mapping_00.json")

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "CT"},
        labels={"background": 0, "kidney_and_masses": (1, 2, 3), "masses": (2, 3), "tumor": 2},
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        regions_class_order=(1, 3, 2),
        # order has to be this because tumor is the last region (== label 2)
        dataset_name=TASK_NAME,
        overwrite_image_reader_writer="NibabelIOWithReorient",
        description="KiTS2023",
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
