"""
What to do here:

Assumes data is already copied (ACDC as train, M&Ms as test)
- Need to convert the segmentation convention
- Create dataset.json
- Create domain mapping file for test set images
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import KFold

from segmentation_failures.utils.io import save_json

TASK_NAME = "Dataset510_ACDCtoMMs"


def convert_mnm_to_acdc_labels(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 3
    new_seg[seg == 2] = 2
    new_seg[seg == 3] = 1
    return new_seg


def main():
    num_id_test_cases = 20  # will result in 2x the number moved (ED and ES)
    source_dir_acdc = Path(os.environ["nnUNet_raw_data_base"]) / "nnUNet_raw_data" / "Task027_ACDC"
    source_dir_mnms = Path(os.environ["nnUNet_raw_data_base"]) / "nnUNet_raw_data" / "Task101_mms"
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"

    target_root_dir.mkdir()
    # Copy training images/labels from acdc for training
    shutil.copytree(source_dir_acdc / "imagesTr", images_train_dir)
    shutil.copytree(source_dir_acdc / "labelsTr", labels_train_dir)
    # Copy training images/labels from M&Ms for testing
    shutil.copytree(source_dir_mnms / "imagesTr", images_test_dir)
    shutil.copytree(source_dir_mnms / "labelsTr", labels_test_dir)

    # Move some of the ACDC training cases into the test set for ID testing
    # TODO if we're unlucky there could also be a pathology shift here.
    # Ideally, use the ACDC meta-data to select cases from each pathology group
    patient_files = defaultdict(list)
    for x in images_train_dir.iterdir():
        if x.name.endswith(".nii.gz"):
            # I know that they are named patientXXX_frameYY_0000.nii.gz
            patient_id = x.name.split("_")[0]
            patient_files[patient_id].append(x)
    for patient_id in random.sample(list(patient_files), k=num_id_test_cases):
        for image_file in patient_files[patient_id]:
            # image
            shutil.move(image_file, images_test_dir / image_file.name)
            # label
            label_file = labels_train_dir / (
                image_file.name.removesuffix("_0000.nii.gz") + ".nii.gz"
            )
            shutil.move(label_file, labels_test_dir / label_file.name)
    num_train_cases = 2 * (len(patient_files) - num_id_test_cases)  # ED and ES

    # Here I extract the domains from the label file name, which Peter chose like this:
    # filename = "{}_{}_{}_{}.{}".format(pat_id, str(ts).zfill(4), vendor, centre, data_format)
    case_to_domain_map0 = {}
    case_to_domain_map1 = {}
    for label_file in labels_test_dir.iterdir():
        if not label_file.name.endswith("nii.gz"):
            continue
        case_id = label_file.name.removesuffix(".nii.gz")
        if case_id.startswith("patient"):
            # ACDC case
            case_to_domain_map0[case_id] = "ID"
            case_to_domain_map1[case_id] = "ID"
        else:
            # M&Ms case
            _, _, vendor, centre = case_id.split("_")
            case_to_domain_map0[case_id] = centre
            case_to_domain_map1[case_id] = vendor
    save_json(case_to_domain_map0, target_root_dir / "domain_mapping_00.json")
    save_json(case_to_domain_map1, target_root_dir / "domain_mapping_01.json")

    # convert labels
    for label_file in labels_test_dir.iterdir():
        if not label_file.name.endswith("nii.gz") or label_file.name.startswith("patient"):
            # not an image OR ACDC subject, which should not be converted
            continue
        # load mask
        seg_itk = sitk.ReadImage(str(label_file))
        seg_npy = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)  # shape (z, y, x)
        new_seg = convert_mnm_to_acdc_labels(seg_npy)
        # save mask
        new_seg_itk = sitk.GetImageFromArray(new_seg)
        new_seg_itk.CopyInformation(seg_itk)
        sitk.WriteImage(new_seg_itk, str(label_file))

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "MRI"},
        labels={"background": 0, "RV": 1, "MLV": 2, "LVC": 3},
        num_training_cases=num_train_cases,
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        dim=3,
    )


def trainval_splits(num_folds=5, seed=12346):
    train_img_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME / "imagesTr"
    preproc_root_dir = Path(os.environ["nnUNet_preprocessed"]) / TASK_NAME
    # Split into 5 folds, but keep the two frames of each patient together
    # (i.e. if patient 1 has ED and ES, then both frames will be in the same fold)
    patient_files = defaultdict(list)
    for x in train_img_dir.iterdir():
        if x.name.endswith(".nii.gz"):
            # I know that they are named patientXXX_frameYY_0000.nii.gz
            patient_id = x.name.split("_")[0]
            patient_files[patient_id].append("_".join(x.name.split("_")[:2]))
    patient_ids = list(patient_files)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_splits = []
    for train_idx, val_idx in kfold.split(patient_ids):
        split_dict = {"train": [], "val": []}
        for i in train_idx:
            for case_frame_id in patient_files[patient_ids[i]]:
                split_dict["train"].append(case_frame_id)
        for i in val_idx:
            for case_frame_id in patient_files[patient_ids[i]]:
                split_dict["val"].append(case_frame_id)
        all_splits.append(split_dict)
    save_json(all_splits, preproc_root_dir / "splits_final.json")


if __name__ == "__main__":
    main()
    # IMPORTANT: Do the custom train-val split below after preprocessing!
    # trainval_splits()
