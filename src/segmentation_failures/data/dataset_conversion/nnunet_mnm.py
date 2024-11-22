"""
What to do:
1. Create a train/test split by vendor (train on vendor B, test on others)
2. Convert the data to nnUNet format
3. Create a dataset.json file

"""

import argparse
import os
import random
from pathlib import Path

import nibabel as nib
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import KFold
from tqdm import tqdm

from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset511_MnM_VendorB_train"


def extract_select_ed_es_frames(nifti_path: str, edi_idx: int, esy_idx: int, target_dir: str):
    nifti_path = Path(nifti_path)
    # select ED/ES frames.
    # Load the 4D nifti file using nibabel and save the selected frames as 3D nifti files
    img4d = nib.load(nifti_path)
    data = img4d.get_fdata()
    edi_data = data[..., edi_idx]
    esy_data = data[..., esy_idx]
    edi_img = nib.Nifti1Image(edi_data, affine=img4d.affine)
    esy_img = nib.Nifti1Image(esy_data, affine=img4d.affine)
    if nifti_path.name.endswith("_sa_gt.nii.gz"):
        target_name = Path(nifti_path).name.removesuffix("_sa_gt.nii.gz")
        nib.save(edi_img, Path(target_dir) / (target_name + "_ED.nii.gz"))
        nib.save(esy_img, Path(target_dir) / (target_name + "_ES.nii.gz"))
    else:
        assert nifti_path.name.endswith("_sa.nii.gz")
        target_name = Path(nifti_path).name.removesuffix("_sa.nii.gz")
        nib.save(edi_img, Path(target_dir) / (target_name + "_ED_0000.nii.gz"))
        nib.save(esy_img, Path(target_dir) / (target_name + "_ES_0000.nii.gz"))


def copy_case(
    case_id: int,
    source_dir: Path,
    image_target_dir: Path,
    label_target_dir: Path,
    ed_idx: int,
    es_idx: int,
):
    for subdir in ["Training/Labeled", "Training/Labeled", "Validation", "Testing"]:
        case_dir = source_dir / subdir / f"{case_id}"
        if case_dir.exists():
            extract_select_ed_es_frames(
                case_dir / f"{case_id}_sa.nii.gz", ed_idx, es_idx, image_target_dir
            )
            extract_select_ed_es_frames(
                case_dir / f"{case_id}_sa_gt.nii.gz", ed_idx, es_idx, label_target_dir
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    parser.add_argument(
        "--use_default_splits", action="store_true", help="Use default train/test split."
    )
    args = parser.parse_args()
    source_dir = Path(args.raw_data_dir)
    default_split_path = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
    seed = 47382
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
    data_info = pd.read_csv(
        source_dir / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        index_col=0,
    )
    # save the dataset information file for reference
    data_info.to_csv(target_root_dir / "mnm_dataset_information.csv", index=False)
    data_info = data_info.set_index("External code")

    vendorB_cases = data_info.loc[data_info.Vendor == "B"].index.tolist()
    other_cases = data_info.loc[data_info.Vendor != "B"].index.tolist()
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        # remove ED/ES suffixes
        train_cases = list(set([x.removesuffix("_ED").removesuffix("_ES") for x in train_cases]))
        test_cases = [x for x in vendorB_cases + other_cases if x not in train_cases]
        test_cases = list(set([x.removesuffix("_ED").removesuffix("_ES") for x in test_cases]))
    else:
        random.seed(seed)
        random.shuffle(vendorB_cases)
        random.shuffle(other_cases)
        # use 30 vendor-B cases for testing
        train_cases = vendorB_cases[30:]
        test_cases = vendorB_cases[:30] + other_cases

    print("Copying cases...")
    for case_id in tqdm(train_cases):
        copy_case(
            case_id,
            source_dir,
            images_train_dir,
            labels_train_dir,
            data_info.at[case_id, "ED"],
            data_info.at[case_id, "ES"],
        )
    for case_id in tqdm(test_cases):
        copy_case(
            case_id,
            source_dir,
            images_test_dir,
            labels_test_dir,
            data_info.at[case_id, "ED"],
            data_info.at[case_id, "ES"],
        )

    domain_mapping = {}
    # I save train cases here, too, because I want to use validation splits for OOD testing
    for seg_file in list(labels_test_dir.glob("*.nii.gz")):
        # expected format: {case_id}_{ED,ES}.nii.gz
        file_id = seg_file.name.removesuffix(".nii.gz")
        case_id = seg_file.name.split("_")[0]
        domain_mapping[file_id] = data_info.at[case_id, "Vendor"]
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


def trainval_splits(num_folds=5, seed=12346):
    train_img_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME / "imagesTr"
    preproc_root_dir = Path(os.environ["nnUNet_preprocessed"]) / TASK_NAME
    # Split into 5 folds, but keep the two frames of each patient together
    # (i.e. if patient 1 has ED and ES, then both frames will be in the same fold)
    patient_ids = set()
    for x in train_img_dir.iterdir():
        if x.name.endswith(".nii.gz"):
            # I know that they are named patientXXX_{ED,ES}_0000.nii.gz
            curr_id = x.name.split("_")[0]
            patient_ids.add(curr_id)
    patient_ids = list(patient_ids)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_splits = []
    for train_idx, val_idx in kfold.split(patient_ids):
        split_dict = {"train": [], "val": []}
        for i in train_idx:
            split_dict["train"].extend([f"{patient_ids[i]}_{frame}" for frame in ["ED", "ES"]])
        for i in val_idx:
            split_dict["val"].extend([f"{patient_ids[i]}_{frame}" for frame in ["ED", "ES"]])
        all_splits.append(split_dict)
    save_json(all_splits, preproc_root_dir / "splits_final.json")


if __name__ == "__main__":
    main()
    # Make a custom train-val split below after preprocessing:
    # trainval_splits()
