import argparse
import os
import shutil
from pathlib import Path

import dotenv

from segmentation_failures.utils.data import get_dataset_dir
from segmentation_failures.utils.io import load_json

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False, verbose=True)


def check_trainval_split(dataset_name: str, ref_split_path: Path, dry_run=False):
    # check if the splits are the same
    split_path = Path(os.environ["nnUNet_preprocessed"]) / dataset_name / "splits_final.json"
    if dataset_name.startswith("Dataset500"):
        split_path = Path(os.environ["nnUNet_raw"]) / dataset_name / "splits_final.json"
    if not split_path.exists():
        print("No split found. Copying reference split...")
        if not dry_run:
            shutil.copy(ref_split_path, split_path)
        return
    ref_split = load_json(split_path)
    curr_split = load_json(ref_split_path)
    # compare the two splits. Each should be a list of dictionaries with keys "train" and "val"
    for fold_idx in range(len(ref_split)):
        ref_fold = ref_split[fold_idx]
        curr_fold = curr_split[fold_idx]
        assert ref_fold.keys() == curr_fold.keys()
        for key in ref_fold:
            # should be lists
            assert isinstance(ref_fold[key], list)
            assert isinstance(curr_fold[key], list)
            if set(ref_fold[key]) != set(curr_fold[key]):
                raise ValueError(f"TRAIN/VAL: Splits are different for fold {fold_idx}.")
    print("TRAIN/VAL: Splits are OK.")


def check_test_cases(dataset_name: str, ref_split_path: Path, dry_run=False):
    test_data_dir = Path(os.environ["TESTDATA_ROOT_DIR"]) / dataset_name
    # check if the domain mapping is the same
    domain_mapping_path = test_data_dir / "domain_mapping_00.json"
    if not domain_mapping_path.exists():
        print("No domain mapping found. Copying reference domain mapping...")
        if not dry_run:
            shutil.copy(ref_split_path, domain_mapping_path)
        return
    ref_domain_mapping = load_json(ref_split_path)
    curr_domain_mapping = load_json(domain_mapping_path)
    if ref_domain_mapping.keys() != curr_domain_mapping.keys():
        raise ValueError("TEST: Test cases are different.")
    else:
        print("TEST: Split is OK.")
    wrong_entries = []
    for key in ref_domain_mapping:
        if ref_domain_mapping[key] != curr_domain_mapping[key]:
            wrong_entries.append(key)
    if len(wrong_entries) > 0:
        raise ValueError(f"TEST: Domain mapping is different for {wrong_entries}.")
    print("TEST: Domain mapping is OK.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", nargs="+", help="Dataset IDs to check.", default=None)
    parser.add_argument(
        "--dry_run", action="store_true", help="Do not copy train/val split files."
    )
    args = parser.parse_args()
    split_dir = Path(__file__).resolve().parents[3] / "dataset_splits"
    if args.dataset_ids is not None:
        datasets_ids = args.dataset_ids
    else:
        datasets_ids = [x.name for x in split_dir.iterdir() if x.is_dir()]
    for ds in datasets_ids:
        dataset_dir = get_dataset_dir(ds, split_dir)
        print("=" * 10)
        print(dataset_dir.name)
        dataset_name = dataset_dir.name
        check_trainval_split(dataset_name, dataset_dir / "splits_final.json")
        check_test_cases(dataset_name, dataset_dir / "domain_mapping_00.json")


if __name__ == "__main__":
    main()
