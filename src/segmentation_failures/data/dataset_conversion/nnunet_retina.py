import argparse
import os
from pathlib import Path

import numpy as np
import skimage.io as io
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from skimage.transform import resize
from tqdm import tqdm

from segmentation_failures.utils.io import save_json


def convert_case(img_path, lab_path, target_img_path, target_lab_path, resize_to=None):
    seg = io.imread(lab_path)
    img = io.imread(img_path)
    if resize_to is not None:
        img = resize(img, resize_to, anti_aliasing=True).astype(np.float32)
        if target_img_path.suffix == ".png":
            img = img / 255.0
        seg = resize(seg, resize_to, anti_aliasing=False, order=0)
    # Need to convert labels from rgb ints to consecutive ints
    new_seg = np.zeros_like(seg, dtype=np.uint8)
    new_seg[seg == 255] = 0  # background
    new_seg[seg == 0] = 1  # optic cup
    new_seg[seg == 128] = 2  # optic disk/ring
    # remove redundant channels
    new_seg = new_seg[..., 0]
    io.imsave(target_lab_path, new_seg, check_contrast=False)
    io.imsave(target_img_path, img, check_contrast=False)


def convert_domain(
    domain_dir,
    images_tr_dir,
    images_ts_dir,
    labels_tr_dir,
    labels_ts_dir,
    ood=False,
    use_roi=True,
    resize_to=None,
    output_suffix=".tiff",
):
    # tiff is used to avoid compression (if also resizing)
    domain_dir = Path(domain_dir)
    images_tr_dir = Path(images_tr_dir)
    images_ts_dir = Path(images_ts_dir)
    labels_tr_dir = Path(labels_tr_dir)
    labels_ts_dir = Path(labels_ts_dir)
    if not use_roi:
        # This is in principle possible, but I would need special preprocessing
        # (for Domain 2 there are two scans per image)
        raise NotImplementedError
    src_train_dir = domain_dir / "train/ROIs/"
    src_test_dir = domain_dir / "test/ROIs/"
    if ood:
        # move all to test set
        images_tr_dir = images_ts_dir
        labels_tr_dir = labels_ts_dir
    all_paths = []
    for img_file in (src_train_dir / "image").iterdir():
        case_id = img_file.stem
        assert img_file.suffix == ".png"
        lab_file = src_train_dir / "mask" / (case_id + img_file.suffix)
        dest_img_file = images_tr_dir / f"{domain_dir.name}_{case_id}_0000{output_suffix}"
        dest_lab_file = labels_tr_dir / f"{domain_dir.name}_{case_id}{output_suffix}"
        # convert_case(img_file, lab_file, dest_img_file, dest_lab_file)
        all_paths.append((img_file, lab_file, dest_img_file, dest_lab_file))
    for img_file in (src_test_dir / "image").iterdir():
        case_id = img_file.stem
        assert img_file.suffix == ".png"
        lab_file = src_test_dir / "mask" / (case_id + img_file.suffix)
        dest_img_file = images_ts_dir / f"{domain_dir.name}_{case_id}_0000{output_suffix}"
        dest_lab_file = labels_ts_dir / f"{domain_dir.name}_{case_id}{output_suffix}"
        # convert_case(img_file, lab_file, dest_img_file, dest_lab_file)
        all_paths.append((img_file, lab_file, dest_img_file, dest_lab_file))
    for path_tuple in tqdm(all_paths):
        convert_case(*path_tuple, resize_to=resize_to)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory (MSD)")
    parser.add_argument(
        "--resize", type=int, help="Resize images to this size", required=False, default=None
    )
    args = parser.parse_args()

    source_dir = Path(args.raw_data_dir)
    TASK_NAME = "Dataset530_DoFEOpticDiscCup"
    if args.resize:
        print(f"Resizing images to {args.resize}x{args.resize}")
        TASK_NAME = f"Dataset531_DoFEOpticDiscCup_{args.resize}"
    filetype = ".png" if args.resize is None else ".tiff"

    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir(exist_ok=True)
    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"
    # mkdir
    images_train_dir.mkdir(exist_ok=True)
    images_test_dir.mkdir(exist_ok=True)
    labels_train_dir.mkdir(exist_ok=True)
    labels_test_dir.mkdir(exist_ok=True)

    # use REFUGE (Domain 3/4) for training and others for testing
    domain_mapping = {
        "Domain1": "dristhi",
        "Domain2": "rimone3",
        "Domain3": "refuge1",
        "Domain4": "refuge2",
    }
    id_domains = ["Domain3", "Domain4"]
    for d in domain_mapping:
        print(f"Converting domain {d}...")
        convert_domain(
            source_dir / d,
            images_train_dir,
            images_test_dir,
            labels_train_dir,
            labels_test_dir,
            ood=d not in id_domains,
            resize_to=(args.resize, args.resize) if args.resize else None,
            output_suffix=filetype,
        )

    # Here I extract the domains from the file names (I set them earlier as a prefix)
    case_to_domain_map = {}
    for label_file in labels_test_dir.iterdir():
        if not label_file.suffix == filetype:
            print("WARNING: Skipping file", label_file)
            continue
        case_id = label_file.stem
        d = case_id.split("_")[0]
        if d in id_domains:
            d = "ID"
        case_to_domain_map[case_id] = d
    save_json(case_to_domain_map, target_root_dir / "domain_mapping_00.json")
    num_train_cases = len(list(images_train_dir.iterdir()))

    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "R", 1: "G", 2: "B"},
        # note: this does not work with nnunet's overlay plot function (needs one modality)
        labels={
            "background": 0,
            "optic disc": [1, 2],
            "optic cup": 1,
        },
        regions_class_order=[2, 1],
        num_training_cases=num_train_cases,
        overwrite_image_reader_writer="NaturalImage2DIO",
        file_ending=filetype,
        dataset_name=TASK_NAME,
        dim=2,
        domain_mapping=domain_mapping,
    )


if __name__ == "__main__":
    main()
