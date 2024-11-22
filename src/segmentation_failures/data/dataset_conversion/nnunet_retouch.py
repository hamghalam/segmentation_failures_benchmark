import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from segmentation_failures.utils.io import load_json, save_json


def process_case(
    case_dir, img_out_dir, label_out_dir, scanner, only_stats=False, merge_labels=False
):
    # load with simpleITK
    img = sitk.ReadImage(str(case_dir / "oct.mhd"))
    seg = sitk.ReadImage(str(case_dir / "reference.mhd"))
    # for the image, we need to normalize the intensities
    # Cirrus: 0-255, Spectralis: 0-2**16, Topcon: 0-255
    img_arr = sitk.GetArrayFromImage(img)
    image_stats = {
        "min": img_arr.min(),
        "max": img_arr.max(),
        "dtype": str(img_arr.dtype),
    }
    if img_arr.dtype == np.uint8:
        img_arr = img_arr.astype(np.float32) / 255
    elif img_arr.dtype == np.uint16:
        img_arr = img_arr.astype(np.float32) / 2**16
    if merge_labels:
        seg_old = seg
        seg_arr = sitk.GetArrayFromImage(seg)
        seg_arr[seg_arr != 0] = 1
        seg = sitk.GetImageFromArray(seg_arr)
        seg.CopyInformation(seg_old)
    # save in new location
    if not only_stats:
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(img)
        sitk.WriteImage(img_out, str(img_out_dir / f"{case_dir.name}_0000.nii.gz"))
        sitk.WriteImage(seg, str(label_out_dir / f"{case_dir.name}.nii.gz"))
    return image_stats


def main():
    subsets = ["Cirrus", "Spectralis", "Topcon"]
    random.seed(420000)
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory (MSD)")
    parser.add_argument(
        "--ood_subset", default="Cirrus", choices=subsets, help="OOD subset to use."
    )
    parser.add_argument(
        "--use_default_splits",
        action="store_true",
        help="Use default train/test split.",
    )
    parser.add_argument(
        "--merge_labels",
        action="store_true",
        help="Merge labels to a single fluid class.",
    )
    args = parser.parse_args()
    source_dir = Path(args.raw_data_dir)
    if not source_dir.exists():
        raise FileNotFoundError("One of the specified directories does not exist")
    TASK_NAME = f"Dataset5{4 + args.merge_labels}{subsets.index(args.ood_subset)}_RETOUCH_ood={args.ood_subset}"
    print(f"Processing data for task {TASK_NAME}")
    num_id_test_cases_per_scanner = 6
    nnunet_split = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
        nnunet_split = load_json(default_split_path)
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir(exist_ok=True)

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"
    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    # get a list of all patients for each scanner
    # and do the train/test split
    case_dict = {}
    all_cases = []
    for curr_scanner in subsets:
        subset_dir = source_dir / f"RETOUCH-TrainingSet-{curr_scanner}"
        case_dict[curr_scanner] = [x.name for x in subset_dir.iterdir() if x.is_dir()]
        all_cases.extend(case_dict[curr_scanner])
    if nnunet_split is not None:
        train_cases = []
        for fold in nnunet_split:
            train_cases.extend(fold["train"])
        test_cases = list(set(all_cases) - set(train_cases))
    else:
        train_cases = []
        test_cases = []
        for scanner, case_list in case_dict.items():
            if scanner == args.ood_subset:
                test_cases.extend(case_list)
            else:
                selected_cases = random.sample(case_list, num_id_test_cases_per_scanner)
                train_cases.extend([x for x in case_list if x not in selected_cases])
                test_cases.extend(selected_cases)

    # process all cases
    all_stats = {k: [] for k in subsets}
    case_to_domain_map = {}
    results = []
    print(f"Processing {len(train_cases)} training cases and {len(test_cases)} test cases")
    with ProcessPoolExecutor(max_workers=6) as executor:
        for scanner, case_list in case_dict.items():
            for case_id in case_list:
                if case_id in test_cases:
                    img_out_dir = images_test_dir
                    label_out_dir = labels_test_dir
                    case_to_domain_map[case_id] = scanner
                else:
                    img_out_dir = images_train_dir
                    label_out_dir = labels_train_dir
                results.append(
                    (
                        scanner,
                        executor.submit(
                            process_case,
                            source_dir / f"RETOUCH-TrainingSet-{scanner}" / case_id,
                            img_out_dir,
                            label_out_dir,
                            scanner=scanner,
                            merge_labels=args.merge_labels,
                        ),
                    )
                )
    for scanner, r in results:
        stats = r.result()
        all_stats[scanner].append(stats)
    for scanner, stats in all_stats.items():
        print(f"=== Stats for {scanner} ===")
        overall_min = min([x["min"] for x in stats])
        overall_max = max([x["max"] for x in stats])
        all_dtypes = set([x["dtype"] for x in stats])
        print(f"Overall min: {overall_min}, max: {overall_max}")
        print(f"Overall dtypes: {all_dtypes}")
    save_json(case_to_domain_map, target_root_dir / "domain_mapping_00.json")
    label_dict = {
        "background": 0,
        "Intraretinal Fluid": 1,
        "Subretinal Fluid": 2,
        "Pigment Epithelium Detachments": 3,
    }
    if args.merge_labels:
        label_dict = {"background": 0, "fluid": 1}
    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "OCT"},
        labels=label_dict,
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
