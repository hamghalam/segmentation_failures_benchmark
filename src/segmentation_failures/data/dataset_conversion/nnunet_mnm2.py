"""
What to do:
1. Create a train/test split by pathologies
2. Convert the data to nnUNet format
3. Create a dataset.json file

"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from segmentation_failures.utils.io import save_json

TASK_NAME = "Dataset512_MnM2_Pathology"


def copy_case(case_id: int, source_dir: Path, image_target_dir: Path, label_target_dir: Path):
    # Copy only short-axis MRI and segmentation, two frames (ED/ES) per patient
    # expected format: {case_id}_{SA,LA}_{ED,ES,CINE}{,_gt}.nii.gz
    for frame in ["ED", "ES"]:
        image_file = source_dir / f"{case_id:03d}_SA_{frame}.nii.gz"
        shutil.copy(
            image_file, image_target_dir / image_file.name.replace(".nii.gz", "_0000.nii.gz")
        )
        label_file = source_dir / f"{case_id:03d}_SA_{frame}_gt.nii.gz"
        shutil.copy(
            label_file, label_target_dir / label_file.name.replace("_gt.nii.gz", ".nii.gz")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    args = parser.parse_args()
    source_dir = Path(args.raw_data_dir)
    MODALITIES = {
        0: "MRI-SA",
    }
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir()

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"

    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    # generate train/test split
    data_info = (
        pd.read_csv(source_dir / "dataset_information.csv", dtype=str)
        .dropna(axis=0, how="all")
        .astype({"SUBJECT_CODE": int})
    )
    # save the dataset information file for reference
    data_info.to_csv(target_root_dir / "mnm2_dataset_information.csv", index=False)
    # Use all cases for training, no test cases. This makes it easier to define splits via train/val
    # NOTE since the focus of this experiment is on pathology shift, I think it is ok to use the image geometry
    # information from ood cases during preprocessing.
    train_cases = data_info.SUBJECT_CODE.unique()
    test_cases = []

    print("Copying cases...")
    for case_id in tqdm(train_cases):
        copy_case(case_id, source_dir, images_train_dir, labels_train_dir)
    for case_id in tqdm(test_cases):
        copy_case(case_id, source_dir, images_test_dir, labels_test_dir)

    domain_mapping = {}
    data_info = data_info.set_index("SUBJECT_CODE")
    # I save train cases here, too, because I want to use validation splits for OOD testing
    for seg_file in list(labels_train_dir.glob("*.nii.gz")) + list(
        labels_test_dir.glob("*.nii.gz")
    ):
        file_id = seg_file.name.removesuffix(".nii.gz")
        case_id = int(seg_file.name.split("_")[0])
        domain_mapping[file_id] = data_info.at[case_id, "DISEASE"]
    save_json(domain_mapping, target_root_dir / "domain_mapping_00.json")

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names=MODALITIES,
        labels={
            "background": 0,
            "left_ventricle": 1,
            "left_ventricular_myocardium": 2,
            "right_ventricle": 3,
        },
        num_training_cases=len(train_cases) * 2,
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        overwrite_image_reader_writer="NibabelIO",
        dim=3,
    )


def trainval_splits(num_folds=3, seed=12346):
    raw_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    preproc_root_dir = Path(os.environ["nnUNet_preprocessed"]) / TASK_NAME
    # Split into folds, but keep the two frames of each patient together
    # (i.e. if patient 1 has ED and ES, then both frames will be in the same fold)
    metadata = pd.read_csv(raw_root_dir / "mnm2_dataset_information.csv")

    def stupid_renaming(name):
        if name.lower().startswith("philips"):
            return 0
        elif name.lower().startswith("siemens"):
            return 1
        elif name.lower().startswith("ge"):
            return 2
        else:
            raise ValueError

    metadata["VENDOR"] = metadata.VENDOR.map(stupid_renaming)

    # SPLIT ACCODING TO DISEASE [fold 0-2]
    # add all cases != NOR to the validation set
    diseased_cases = metadata[metadata.DISEASE != "NOR"].SUBJECT_CODE.tolist()
    nor_subdf = metadata[metadata.DISEASE == "NOR"]

    vendor_labels = nor_subdf.VENDOR
    # three folds because there are 75 training images with normal condition
    strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in strat_kfold.split(nor_subdf, vendor_labels):
        train_cases = nor_subdf.iloc[train_idx].SUBJECT_CODE.tolist()
        val_cases = nor_subdf.iloc[test_idx].SUBJECT_CODE.tolist()
        folds.append({"train": train_cases, "val": val_cases + diseased_cases})

    # SPLIT ACCODING TO VENDOR [fold 3-5], just a single training fold per vendor
    num_train = 50
    for id_scanner in metadata.VENDOR.unique():
        if id_scanner == 2:
            # ge has not enough cases
            continue
        # generate folds with 50 training cases and the rest validation cases
        vendor_subdf = metadata[metadata.VENDOR == id_scanner]
        train_cases = vendor_subdf.sample(n=num_train, random_state=seed).SUBJECT_CODE.tolist()
        val_cases = metadata[~metadata.SUBJECT_CODE.isin(train_cases)].SUBJECT_CODE.tolist()
        folds.append({"train": train_cases, "val": val_cases})

    # add _ED and _ES to the case IDs
    for fold in folds:
        for split in ["train", "val"]:
            fold[split] = [f"{case_id:03d}_SA_ED" for case_id in fold[split]] + [
                f"{case_id:03d}_SA_ES" for case_id in fold[split]
            ]
    save_json(folds, preproc_root_dir / "splits_final.json")


if __name__ == "__main__":
    main()
    # IMPORTANT: Do the custom train-val split below after preprocessing!
    # trainval_splits()
