import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from segmentation_failures.data.dataset_conversion.nnunet_fets22 import (
    convert_labels_to_nnunet,
    copy_case,
)
from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset503_BraTS19"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    parser.add_argument(
        "--use_default_splits", action="store_true", help="Use default train/test split."
    )
    args = parser.parse_args()

    seed = 42
    MODALITIES = {
        0: "t1",
        1: "t1ce",
        2: "t2",
        3: "flair",
    }
    num_test_per_grade = 50
    num_lgg_train = 26
    # NOTE if you change this, make sure to use another TASK_NAME!
    # validation splits would be a more flexible solution, but I don't want to select checkpoints based on OOD performance
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

    case_info_csv = pd.read_csv(source_dir / "name_mapping.csv")
    if num_test_per_grade + num_lgg_train > len(case_info_csv[case_info_csv.Grade == "LGG"]):
        raise ValueError(
            "Not enough LGG cases for this split. Please adjust num_test_per_grade and num_lgg_train."
        )
    domain_mapping = case_info_csv.set_index("BraTS_2019_subject_ID")["Grade"].to_dict()
    # split cases into train/test as specified by num_test_per_grade and num_lgg_train
    lgg_cases = case_info_csv[case_info_csv.Grade == "LGG"].BraTS_2019_subject_ID.tolist()
    hgg_cases = case_info_csv[case_info_csv.Grade == "HGG"].BraTS_2019_subject_ID.tolist()
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        test_cases = [x for x in lgg_cases + hgg_cases if x not in train_cases]
    else:
        random.seed(seed)
        test_cases = random.sample(lgg_cases, num_test_per_grade) + random.sample(
            hgg_cases, num_test_per_grade
        )
        train_cases = list(set(lgg_cases) - set(test_cases))[:num_lgg_train] + list(
            set(hgg_cases) - set(test_cases)
        )

    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    print("Copying cases to nnunet format...")
    for case_id in tqdm(train_cases):
        case_dir = source_dir / domain_mapping[case_id] / case_id
        copy_case(case_dir, images_train_dir, labels_train_dir, MODALITIES)
    for case_id in tqdm(test_cases):
        case_dir = source_dir / domain_mapping[case_id] / case_id
        copy_case(case_dir, images_test_dir, labels_test_dir, MODALITIES)
    # save only domains of test cases
    save_json(domain_mapping, target_root_dir / "all_tumor_grades.json")
    domain_mapping = {k: v for k, v in domain_mapping.items() if k in test_cases}
    save_json(domain_mapping, target_root_dir / "domain_mapping_00.json")

    # map labels to nnunet format
    all_label_files = list(labels_train_dir.glob("*.nii.gz")) + list(
        labels_test_dir.glob("*.nii.gz")
    )
    print("Converting labels to nnunet format...")
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


def trainval_split(seed=904187):
    raw_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    preproc_root_dir = Path(os.environ["nnUNet_preprocessed"]) / TASK_NAME
    case2grade = load_json(raw_root_dir / "all_tumor_grades.json")
    test_cases = list(load_json(raw_root_dir / "domain_mapping_00.json"))
    mapping = {
        "LGG": 0,
        "HGG": 1,
    }
    # stratified train-val split by tumor grade
    train_cases = [x for x in list(case2grade) if x not in test_cases]
    train_grades = [mapping[case2grade[x]] for x in train_cases]
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in strat_kfold.split(train_cases, train_grades):
        folds.append(
            {
                "train": np.array(train_cases)[train_idx].tolist(),
                "val": np.array(train_cases)[val_idx].tolist(),
            }
        )
    save_json(folds, preproc_root_dir / "splits_final.json")


if __name__ == "__main__":
    main()
    # trainval_split()
