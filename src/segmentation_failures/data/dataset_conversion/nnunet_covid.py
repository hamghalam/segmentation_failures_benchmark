"""
What to do here:
- copy and rename the training cases (challenge folder)
- copy and rename the testing cases (mosmed and radiopedia)
- create a random split of training images for ID testing
- create a domain mapping file
- create a dataset.json
"""

import argparse
import os
import shutil
from pathlib import Path

import nibabel
import numpy
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset520_CovidLungCT"


def copy_challenge(case_id, source_dir, target_dir_img, target_dir_lab):
    img_path = source_dir / f"{case_id}_ct.nii.gz"
    seg_path = source_dir / f"{case_id}_seg.nii.gz"
    shutil.copy(img_path, target_dir_img / f"challenge_{case_id}_0000.nii.gz")
    shutil.copy(seg_path, target_dir_lab / f"challenge_{case_id}.nii.gz")


def rescale_gray_level_values_to_hu(img_file, hu_min=-2048, hu_max=1900, out_path=None):
    if out_path is None:
        out_path = img_file
    # load image using nibabel
    img = nibabel.load(img_file)
    img_data = img.get_fdata()
    # assert img_data.min() >= 0 and img_data.max() <= 255
    # rescale to HU
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = img_data * (hu_max - hu_min) + hu_min
    # save as nifti. The other datasets use int16, so we do the same here
    new_img = nibabel.Nifti1Image(img_data, affine=img.affine, dtype=numpy.int16)
    nibabel.save(new_img, out_path)


def only_rescale_radiopaedia():
    # radiopaedia cases are not in HU scale but in [0, 1] (uint8)
    target_data_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    tmp_save_dir = target_data_dir / "tmp_rescaled_radiopaedia"
    tmp_save_dir.mkdir(exist_ok=True)
    images_ts_dir = target_data_dir / "imagesTs"
    for img_path in tqdm(images_ts_dir.iterdir()):
        case_id = img_path.name.removesuffix(".nii.gz")
        datasource = case_id.split("_")[0]
        if datasource == "radiopaedia":
            print(case_id)
            rescale_gray_level_values_to_hu(img_path, out_path=tmp_save_dir / img_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_data_dir_challenge", type=str, help="Path to raw data directory (challenge part)"
    )
    parser.add_argument(
        "raw_data_dir_radiopaedia", type=str, help="Path to raw data directory (radiopaedia part)"
    )
    parser.add_argument(
        "raw_data_dir_mosmed", type=str, help="Path to raw data directory (mosmed part)"
    )
    parser.add_argument(
        "--use_default_splits", action="store_true", help="Use default train/test split."
    )
    args = parser.parse_args()
    source_dir_challenge = Path(args.raw_data_dir_challenge) / "Train"
    source_dir_radiopaedia = Path(args.raw_data_dir_radiopaedia)
    source_dir_mosmed = Path(args.raw_data_dir_mosmed)
    if not (
        source_dir_challenge.exists()
        and source_dir_radiopaedia.exists()
        and source_dir_mosmed.exists()
    ):
        raise FileNotFoundError("One of the specified directories does not exist")
    default_split_path = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
    num_id_testing = 39
    seed = 42
    target_data_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    images_tr_dir = target_data_dir / "imagesTr"
    labels_tr_dir = target_data_dir / "labelsTr"
    images_ts_dir = target_data_dir / "imagesTs"
    labels_ts_dir = target_data_dir / "labelsTs"
    target_data_dir.mkdir(exist_ok=True)
    images_tr_dir.mkdir()
    labels_tr_dir.mkdir()
    images_ts_dir.mkdir()
    labels_ts_dir.mkdir()

    # training cases
    case_id_list = [
        x.name.removesuffix("_ct.nii.gz")
        for x in source_dir_challenge.iterdir()
        if x.name.endswith("_ct.nii.gz")
    ]
    if default_split_path is not None:
        all_splits = load_json(default_split_path)
        train_cases = all_splits[0]["train"] + all_splits[0]["val"]
        # remove the challenge_ prefix
        train_cases = [x.removeprefix("challenge_") for x in train_cases]
        id_test_cases = [x for x in case_id_list if x not in train_cases]
    else:
        train_cases, id_test_cases = train_test_split(
            case_id_list, test_size=num_id_testing, random_state=seed
        )
    print("Copying 'Challenge' cases...")
    # copy the images whose names contain _ct or _seg to the imagesTr folder and rename them according to nnunet convention
    for case_id in tqdm(train_cases):
        copy_challenge(case_id, source_dir_challenge, images_tr_dir, labels_tr_dir)
    for case_id in tqdm(id_test_cases):
        copy_challenge(case_id, source_dir_challenge, images_ts_dir, labels_ts_dir)

    # testing cases
    mosmed_images = source_dir_mosmed / "studies/CT-1"  # masks only for these available
    mosmed_labels = source_dir_mosmed / "masks"

    print("Copying 'MosMed' cases...")
    # mosmed samples
    for lab_path in tqdm(mosmed_labels.iterdir()):
        if lab_path.name.endswith(".nii.gz"):
            case_id = lab_path.name.removesuffix("_mask.nii.gz")
            img_path = mosmed_images / f"{case_id}.nii.gz"
            if not img_path.exists():
                raise FileNotFoundError(img_path)
            shutil.copy(lab_path, labels_ts_dir / f"mosmed_{case_id}.nii.gz")
            shutil.copy(img_path, images_ts_dir / f"mosmed_{case_id}_0000.nii.gz")
    print("Copying 'Radiopedia' cases...")
    # radiopedia samples
    radiop_images = source_dir_radiopaedia / "COVID-19-CT-Seg_20cases"
    radiop_labels = source_dir_radiopaedia / "Infection_Mask"
    for lab_path in tqdm(radiop_labels.iterdir()):
        if lab_path.name.endswith(".nii.gz"):
            case_id = lab_path.name.removesuffix(".nii.gz")
            img_path = radiop_images / f"{case_id}.nii.gz"
            if not img_path.exists():
                raise FileNotFoundError(img_path)
            # I don't prepend anything because the case ids already start with the data source
            nnunet_img_path = images_ts_dir / f"{case_id}_0000.nii.gz"
            shutil.copy(lab_path, labels_ts_dir / f"{case_id}.nii.gz")
            shutil.copy(img_path, nnunet_img_path)
            if nnunet_img_path.name.startswith("radiopedia"):
                # Background: Radiopaedia cases are not in HU scale (they use [0, 255]).
                # ALTERNATIVELY use z-normalization during training
                rescale_gray_level_values_to_hu(nnunet_img_path)

    # iterate over test cases to create domain mapping file
    domain_mapping_gonzalez = {}
    domain_mapping_natural = {}
    for lab_path in labels_ts_dir.iterdir():
        case_id = lab_path.name.removesuffix(".nii.gz")
        datasource = case_id.split("_")[0]
        if datasource == "challenge":
            domain_mapping_gonzalez[case_id] = "ID"
            domain_mapping_natural[case_id] = "ID"
        elif datasource in ["radiopaedia", "coronacases"]:
            domain_mapping_natural[case_id] = datasource
            domain_mapping_gonzalez[case_id] = "radiopaedia"
        else:
            domain_mapping_natural[case_id] = datasource
            domain_mapping_gonzalez[case_id] = datasource
    save_json(domain_mapping_gonzalez, target_data_dir / "domain_mapping_00.json")
    save_json(domain_mapping_natural, target_data_dir / "domain_mapping_01.json")

    # NOTE: I do it here (presumable) like Gonzalez et al. and use nnunet's CT normalization
    # HOWEVER, this is not appropriate for some Radiopaedia cases, which do not use HU scale
    # The alternative would be to use a modality != CT or transform the Radiopaedia cases to HU.
    generate_dataset_json(
        output_folder=str(target_data_dir),
        channel_names={0: "CT"},
        labels={"background": 0, "covid": 1},
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        # order has to be this because tumor is the last region (== label 2)
        dataset_name=TASK_NAME,
        description="Collection of COVID-19 lung CT scans from different sources; used originally in Gonzalez et al.",
        dim=3,
    )


if __name__ == "__main__":
    main()
