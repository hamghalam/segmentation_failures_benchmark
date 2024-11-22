import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from segmentation_failures.utils.io import save_json

TASK_NAME = "Dataset502_FeTS22"


def convert_labels_to_nnunet(
    orig_label_file: Path, target_label_file: Path, mapping: dict[int, int]
):
    # convert labels from brats format to nnunet format
    # nnUNet has a different label convention than BraTS; convert back here
    seg_sitk = sitk.ReadImage(str(orig_label_file))
    seg = sitk.GetArrayFromImage(seg_sitk)
    new_seg = seg.copy()
    for orig_label, new_label in mapping.items():
        new_seg[seg == orig_label] = new_label
    new_seg_sitk = sitk.GetImageFromArray(new_seg)
    new_seg_sitk.CopyInformation(seg_sitk)
    sitk.WriteImage(new_seg_sitk, str(target_label_file))


def default_split(split_csv: str, seed: int = 0, test_size_inst1=0.2):
    split_df = pd.read_csv(split_csv)
    all_cases = split_df.Subject_ID.tolist()
    inst1_cases = split_df.loc[split_df["Partition_ID"] == 1, "Subject_ID"].tolist()
    train_cases, _ = train_test_split(inst1_cases, test_size=test_size_inst1, random_state=seed)
    test_cases = list(set(all_cases).difference(train_cases))
    domain_dict = split_df.set_index("Subject_ID")["Partition_ID"].to_dict()
    return train_cases, test_cases, domain_dict


def copy_case(
    case_dir: Path,
    image_target_dir: Path,
    label_target_dir: Path,
    modalities: dict[int, str],
    seg_suffix: str = "seg",
):
    assert case_dir.is_dir()
    case_id = case_dir.name
    for modality_id, modality_name in modalities.items():
        image_file = case_dir / f"{case_id}_{modality_name}.nii.gz"
        shutil.copy(image_file, image_target_dir / f"{case_id}_{modality_id:04d}.nii.gz")
    label_file = case_dir / f"{case_id}_{seg_suffix}.nii.gz"
    shutil.copy(label_file, label_target_dir / f"{case_id}.nii.gz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    args = parser.parse_args()
    seed = 42
    MODALITIES = {
        0: "t1",
        1: "t1ce",
        2: "t2",
        3: "flair",
    }
    perc_id_test_cases = 0.15
    source_dir = Path(args.raw_data_dir)
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir()

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"

    # split cases into train/test
    # default split: Use the partition1.csv from the data directory.
    # - site 1 (largest) is split into training/testing
    # - other sites are 100% testing.
    train_cases, test_cases, domain_mapping = default_split(
        source_dir / "partitioning_1.csv",
        seed=seed,
        test_size_inst1=perc_id_test_cases,
    )
    # copy all images and labels to correct locations
    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    print("Copying cases...")
    for case_id in tqdm(train_cases):
        copy_case(source_dir / case_id, images_train_dir, labels_train_dir, MODALITIES)
    for case_id in tqdm(test_cases):
        copy_case(source_dir / case_id, images_test_dir, labels_test_dir, MODALITIES)
    # save only domains of test cases
    domain_mapping = {k: v for k, v in domain_mapping.items() if k in test_cases}
    save_json(domain_mapping, target_root_dir / "domain_mapping_00.json")

    # map labels to nnunet format
    print("Converting labels to nnUNet format...")
    all_label_files = list(labels_train_dir.glob("*.nii.gz")) + list(
        labels_test_dir.glob("*.nii.gz")
    )
    for label_file in tqdm(all_label_files):
        convert_labels_to_nnunet(
            label_file,
            label_file,
            mapping={
                1: 2,  # necrosis
                2: 1,  # edema
                4: 3,  # enhancing
            },
        )
    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names=MODALITIES,
        labels={
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [2, 3],
            "enhancing_tumor": 3,
        },
        regions_class_order=[1, 2, 3],
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
