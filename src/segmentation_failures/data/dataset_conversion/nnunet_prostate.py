"""
What to do here:

Assumes testing data obtained from the webpage: https://liuquande.github.io/SAML/
Training data used from nnunet raw data (MSD prostate)
- copy data to correct locations
- maybe make labels consistent
- Create dataset.json
- Create domain mapping file for test set images
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset521_ProstateGonzalez"


def convert_labels(label_file_list: list[Path]):
    for label_file in label_file_list:
        if not label_file.name.endswith("nii.gz"):
            continue
        # load mask
        seg_itk = sitk.ReadImage(str(label_file))
        seg_npy = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)  # shape (z, y, x)
        new_seg = 1 * (seg_npy > 0)
        # save mask
        new_seg_itk = sitk.GetImageFromArray(new_seg)
        new_seg_itk.CopyInformation(seg_itk)
        sitk.WriteImage(new_seg_itk, str(label_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir_msd", type=str, help="Path to raw data directory (MSD)")
    parser.add_argument("raw_data_dir_saml", type=str, help="Path to raw data directory (MSD)")
    parser.add_argument(
        "--use_default_splits", action="store_true", help="Use default train/test split."
    )
    args = parser.parse_args()
    num_id_test_cases = 6
    source_dir_msd = Path(args.raw_data_dir_msd)
    source_dir_saml = Path(args.raw_data_dir_saml)
    if not (source_dir_msd.exists() and source_dir_saml.exists()):
        raise FileNotFoundError("One of the specified directories does not exist")
    default_split_path = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir()

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"

    # Copy training images/labels from MSD for training
    images_train_dir.mkdir()
    for img_file in (source_dir_msd / "imagesTr").iterdir():
        if img_file.name.endswith("0000.nii.gz"):
            # only copy T2 modality
            shutil.copy(img_file, images_train_dir / img_file.name)
    shutil.copytree(source_dir_msd / "labelsTr", labels_train_dir)
    # Move some of the training cases into the test set for ID testing
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    patient_files = defaultdict(list)
    for x in images_train_dir.iterdir():
        if x.name.endswith(".nii.gz"):
            patient_id = x.name.removesuffix("_0000.nii.gz")
            patient_files[patient_id].append(x)
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        id_test_cases = [x for x in patient_files if x not in train_cases]
    else:
        id_test_cases = random.sample(list(patient_files), k=num_id_test_cases)
    for patient_id in id_test_cases:
        for image_file in patient_files[patient_id]:
            label_file = labels_train_dir / image_file.name.replace("_0000.nii.gz", ".nii.gz")
            shutil.move(image_file, images_test_dir / f"ID_{image_file.name}")
            shutil.move(label_file, labels_test_dir / f"ID_{label_file.name}")
    num_train_cases = len(patient_files) - len(id_test_cases)

    # Copy training images/labels from the SAML collection for testing
    for site_dir in source_dir_saml.iterdir():
        if not site_dir.is_dir():
            continue
        site_prefix = site_dir.name
        if site_prefix == "RUNMC":
            # This is the same institution as in the decathlon
            # and there are overlaps between the datasets
            continue
        print(site_prefix)
        segmentation_paths = [
            path for path in site_dir.iterdir() if "segmentation" in path.name.lower()
        ]

        for seg_path in tqdm(segmentation_paths):
            # check if corresponding image exists
            case_id = seg_path.name.split("_")[0]
            img_path = seg_path.parent / (case_id + ".nii.gz")
            if not img_path.exists():
                raise FileNotFoundError(f"Image {img_path} does not exist")
            shutil.copy(img_path, images_test_dir / f"{site_prefix}_{case_id}_0000.nii.gz")
            shutil.copy(seg_path, labels_test_dir / f"{site_prefix}_{case_id}.nii.gz")

    # Here I extract the domains from the file names (I set them earlier as a prefix)
    case_to_domain_map = {}
    for label_file in labels_test_dir.iterdir():
        if not label_file.name.endswith("nii.gz"):
            continue
        case_id = label_file.name.removesuffix(".nii.gz")
        case_to_domain_map[case_id] = case_id.split("_")[0]
    save_json(case_to_domain_map, target_root_dir / "domain_mapping_00.json")

    # convert labels
    all_labels = list(labels_test_dir.iterdir()) + list(labels_train_dir.iterdir())
    convert_labels(all_labels)

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "MRI-T2"},
        labels={"background": 0, "whole_prostate": 1},
        num_training_cases=num_train_cases,
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
