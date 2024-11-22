"""
This script can be used for preparing the training data of quality regression methods.
It will look for prediction and confidence map files in the provided input folders and copy them to the output folder.
It will also create a csv file with the ground truth quality scores for each image.
For now, I support only the CV-style data preparation, i.e. use the validation set predictions for all available folds.

What to do here:
Input: Dataset ID, experiment name, output data folder name, [optional] which seeds/folds to use
Output: A folder with the data for the regression (i.e. prediction files and confidence maps if available)
"""

import argparse
import multiprocessing
import os
import shutil
from collections import defaultdict
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
from loguru import logger
from nnunetv2.imageio.reader_writer_registry import (
    determine_reader_writer_from_dataset_json,
)
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from tqdm import tqdm

from segmentation_failures.evaluation.segmentation.compute_seg_metrics import (
    compute_metrics_for_file,
)
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics,
)
from segmentation_failures.utils import GLOBAL_SEEDS
from segmentation_failures.utils.data import get_dataset_dir, load_dataset_json
from segmentation_failures.utils.io import load_expt_config, load_json, save_json

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


def extend_args(parser):
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID")
    parser.add_argument(
        "--experiment_root",
        type=str,
        required=True,
        help="Experiment path. Should be a folder with one subfolder per experiment to include.",
    )
    parser.add_argument(
        "--expt_group",
        type=str,
        default=None,
        help="Name of the experiment group. Will save results to a subfolder with this name.",
    )
    parser.add_argument(
        "--pred_suffix_sep",
        type=str,
        default=None,
        help=(
            "Prediction suffix separator character. Can be used if there are multiple predictions "
            "per case in the experiment folder, which differ by some suffix that starts with the separator."
        ),
    )
    parser.add_argument(
        "--imagecsf_name",
        type=str,
        default="quality_regression",
        help="Name of the imagecsf targeted with this data.",
    )
    parser.add_argument("--output_name", type=str, default=None, help="Output folder name")
    parser.add_argument("--confid", type=str, nargs="+", default=None, help="Pixel CSF names")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds to use")
    parser.add_argument("--seed", type=int, default=None, help="Seed to use")
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Label directory. If provided, use it for computing segmentation metrics.",
    )
    parser.add_argument(
        "--no_preprocessing",
        action="store_true",
        help="If set, do NOT preprocess the prediction files like nnunet does.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, don't copy any files; only compute metrics.",
    )


def main():
    parser = argparse.ArgumentParser()
    extend_args(parser)
    args = parser.parse_args()
    dataset = args.dataset
    expt_root = Path(args.experiment_root)
    expt_group = args.expt_group
    confid_name = args.confid
    pred_suffix_sep = args.pred_suffix_sep
    folds = args.folds
    seed = 0
    if args.seed is not None:
        if args.seed not in GLOBAL_SEEDS:
            raise ValueError(f"Unknown seed {seed}")
        seed = args.seed
    output_name = args.output_name
    if output_name is None:
        # Assume experiment root is $expt_name/{train_seg,validate_pixel_csf}
        output_name = expt_root.parent.name

    label_dir = None if args.label_dir is None else Path(args.label_dir)
    try:
        output_dir = get_dataset_dir(dataset, os.environ["SEGFAIL_AUXDATA"])
    except ValueError as e:
        if not str(e).startswith("Could not find"):
            raise e
        # make the directory
        output_dir = Path(os.environ["SEGFAIL_AUXDATA"]) / get_dataset_dir(dataset).name

    if expt_group is not None:
        output_dir = output_dir / expt_group
    output_dir = (
        output_dir
        / args.imagecsf_name
        / output_name
        / f"folds{'-'.join([str(f) for f in folds])}_seed_{seed}"
    )
    output_dir.mkdir(exist_ok=args.dry_run, parents=True)
    logger.info(f"Output directory: {output_dir}")
    dataset_json = load_dataset_json(dataset)
    suffix = dataset_json.get("file_ending", ".nii.gz")
    # save arguments for logging
    save_json(vars(args), output_dir / "args.json")

    # Get the experiment folder
    if not expt_root.exists():
        raise FileNotFoundError(f"Experiment folder not found: {expt_root}")
    relevant_runs = get_relevant_runs(expt_root, folds, [seed])
    # Copy the prediction files
    src_target_to_case_id = prepare_data(
        list(relevant_runs),
        output_dir,
        confid_names=confid_name,
        dataset_id=dataset,
        pred_suffix_sep=pred_suffix_sep,
        dry_run=args.dry_run,
        skip_preprocessing=args.no_preprocessing,
        file_type=suffix,
    )

    if label_dir is not None and not args.dry_run:
        logger.info("Computing metrics...")
        with multiprocessing.Pool(4) as pool:
            args_list = []
            for (src_file, target_file), case_id in src_target_to_case_id.items():
                lab_file = label_dir / f"{case_id}{suffix}"
                args_list.append((src_file, target_file, lab_file, case_id, dataset))
            results = list(pool.starmap(compute_metrics_worker, args_list))
        metrics_df = pd.concat(results, ignore_index=True)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    logger.info("Done")


def compute_metrics_worker(src_file, target_file, lab_file, case_id, dataset):
    curr_metrics = compute_metrics(
        src_file,
        lab_file,
        dataset_id=dataset,
        metric_names=["dice", "generalized_dice"],
    )
    curr_metrics = pd.DataFrame(curr_metrics)
    curr_metrics["orig_pred_file"] = str(src_file)
    curr_metrics["final_pred_file"] = target_file.name
    curr_metrics["case_id"] = case_id
    return curr_metrics


def get_relevant_runs(expt_root, folds=None, seeds=None):
    relevant_runs = {}
    if seeds is not None:
        actual_seeds = [GLOBAL_SEEDS[seed] for seed in seeds]
    for rundir in expt_root.iterdir():
        # need to load the config to get the seed and fold
        seg_config = load_expt_config(rundir)
        curr_fold = seg_config.datamodule.fold
        curr_seed = seg_config.seed
        if folds is not None and curr_fold not in folds:
            continue
        if seeds is not None and curr_seed not in actual_seeds:
            continue
        relevant_runs[rundir] = (seg_config.datamodule.fold, seg_config.seed)
    if len(relevant_runs) == 0:
        raise ValueError("No matching experiments found")
    seeds_for_fold = {}
    for fold, seed in relevant_runs.values():
        if fold not in seeds_for_fold:
            seeds_for_fold[fold] = []
        seeds_for_fold[fold].append(seed)
    logger.info(
        f"Found {len(relevant_runs)} matching experiments (seeds for each fold: {seeds_for_fold})"
    )
    for fold, seeds in seeds_for_fold.items():
        if len(seeds) > 1:
            logger.warning(f"Found multiple seeds for fold {fold}: {seeds}")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Found duplicate seeds for fold {fold}: {seeds}")
    return relevant_runs


def prepare_data(
    rundirs: list[Path],
    output_dir: Path,
    dataset_id: str,
    confid_names: list[str] = None,
    pred_suffix_sep: str = None,
    dry_run=False,
    skip_preprocessing=False,
    file_type=".nii.gz",
):
    if confid_names is None:
        confid_names = []
    target_dir = output_dir / "predictions"
    # mapping is necessary because it might happen that there are multiple predictions for the same case
    # TODO a datatframe would be easier for keeping track of paths/case ids
    src_target_to_case_id = {}
    case_id_to_src_target = defaultdict(list)  # just for convenience
    target_dir = output_dir / "predictions"
    target_dir.mkdir(exist_ok=True)
    case_id_confid_file_list = []
    target_dirs_confid = []
    for cname in confid_names:
        target_dirs_confid.append(output_dir / "confidence_maps" / cname)
        target_dirs_confid[-1].mkdir(parents=True)
    for rundir in rundirs:
        pred_dir = rundir / "predictions"
        confid_dir = rundir / "confidence_maps"
        if not pred_dir.exists():
            raise FileNotFoundError(f"Predictions folder not found: {pred_dir}")
        if confid_names is not None and not confid_dir.exists():
            raise FileNotFoundError(f"Confidence masks folder not found: {confid_dir}")
        for pred_file in pred_dir.iterdir():
            if not pred_file.name.endswith(file_type):
                logger.warning(f"Skipping file {pred_file} because it is not a {file_type} file.")
                continue
            case_id = pred_file.name.split(".")[0]
            if pred_suffix_sep is not None:
                # remove the suffix from the case id (only from the last occurence)
                case_id = pred_file.name.rsplit(pred_suffix_sep, 1)[0]
            curr_counter = len(case_id_to_src_target.get(case_id, []))
            file_ending = ".npy" if dataset_id != "500" and not skip_preprocessing else file_type
            target_file = target_dir / f"{case_id}_{curr_counter:03d}{file_ending}"
            # update mappings
            src_target_to_case_id[(pred_file, target_file)] = case_id
            case_id_to_src_target[case_id].append(target_file.name)
            # confidence stuff
            for cidx, cname in enumerate(confid_names):
                if pred_suffix_sep is not None:
                    raise NotImplementedError
                confid_src = confid_dir / cname / f"{case_id}{file_type}"
                confid_target = target_dirs_confid[cidx] / target_file.name
                if not confid_src.exists():
                    raise FileNotFoundError(f"Confidence mask not found: {confid_src}")
                case_id_confid_file_list.append((case_id, confid_src, confid_target))

    if not dry_run:
        logger.info("Copying data...")
        preprocess_preds(dataset_id, src_target_to_case_id, skip_preprocessing=skip_preprocessing)
        preprocess_confids(
            dataset_id, case_id_confid_file_list, skip_preprocessing=skip_preprocessing
        )
        # save only the mapping from the target file name to case id
        mapping_file_to_case_id = {
            src_target_paths[1].name: case_id
            for src_target_paths, case_id in src_target_to_case_id.items()
        }
        save_json(mapping_file_to_case_id, target_dir / "prediction_to_case_id.json")
    return src_target_to_case_id


def preprocess_confids(dataset_id, case_id_confid_file_list, skip_preprocessing=False):
    # Copy the prediction files and preprocess them like nnunet
    preprocessor = None
    # for dataset 500 just copy the niftis
    if dataset_id != "500":
        # hard-coding plans and configuration is not very stable, but sufficient for now
        plans_file = (
            get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"]) / "nnUNetPlans.json"
        )
        dataset_json = load_json(plans_file.parent / "dataset.json")
        plans_manager = PlansManager(load_json(plans_file))
        nnunet_config = "3d_fullres"
        if dataset_json.get("dim", 3) == 2:
            nnunet_config = "2d"
        config_manager = plans_manager.get_configuration(nnunet_config)
        preprocessor: DefaultPreprocessor = config_manager.preprocessor_class(verbose=True)
    for case_id, src_file, target_file in tqdm(case_id_confid_file_list):
        if preprocessor is None or skip_preprocessing:
            shutil.copy(src_file, target_file)
        else:
            # HACK Manipulate the configuration here such that the confidence map is resampled like an image.
            tmp_config_mngr = ConfigurationManager(config_manager.configuration)
            tmp_config_mngr.configuration["resampling_fn_seg_kwargs"] = (
                tmp_config_mngr.configuration["resampling_fn_data_kwargs"]
            )
            raw_train_img_dir = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"]) / "imagesTr"
            img_files = [x for x in raw_train_img_dir.iterdir() if x.name.startswith(case_id)]
            if len(img_files) != len(dataset_json["channel_names"]):
                raise ValueError(
                    f"Found {len(img_files)} images for case {case_id}, expected {len(dataset_json['channel_names'])}"
                )
            confid = preprocess_confidence_map(
                img_files, src_file, plans_manager, tmp_config_mngr, dataset_json
            )
            np.save(target_file, confid)


def preprocess_preds(dataset_id, src_target_to_case_id, skip_preprocessing=False):
    # Copy the prediction files and preprocess them like nnunet
    preprocessor = None
    # for dataset 500 just copy the niftis
    if dataset_id != "500":
        # hard-coding plans and configuration is not very stable, but sufficient for now
        plans_file = (
            get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"]) / "nnUNetPlans.json"
        )
        dataset_json = load_json(plans_file.parent / "dataset.json")
        plans_manager = PlansManager(load_json(plans_file))
        nnunet_config = "3d_fullres"
        if dataset_json.get("dim", 3) == 2:
            nnunet_config = "2d"
        config_manager = plans_manager.get_configuration(nnunet_config)
        preprocessor: DefaultPreprocessor = config_manager.preprocessor_class(verbose=True)
    for (src_file, target_file), case_id in tqdm(src_target_to_case_id.items()):
        if preprocessor is None or skip_preprocessing:
            shutil.copy(src_file, target_file)
        else:
            # run nnunet preprocessing on the prediction file and save it as .npy
            # unfortunately, the preprocessor also needs the image files, so I construct the paths here
            raw_train_img_dir = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"]) / "imagesTr"
            img_files = [x for x in raw_train_img_dir.iterdir() if x.name.startswith(case_id)]
            if len(img_files) != len(dataset_json["channel_names"]):
                # special case: RGB
                if not list(dataset_json["channel_names"].values()) == ["R", "G", "B"]:
                    raise ValueError(
                        f"Found {len(img_files)} images for case {case_id}, expected {len(dataset_json['channel_names'])}"
                    )
            _, seg, _ = preprocessor.run_case(
                img_files, src_file, plans_manager, config_manager, dataset_json
            )
            np.save(target_file, seg)


def compute_metrics(pred_file, lab_file, dataset_id, metric_names=None):
    results = []
    # try to get the dataset.json file
    if dataset_id == "500":
        dataset_json_path = (
            get_dataset_dir(dataset_id, os.environ["TESTDATA_ROOT_DIR"]) / "dataset.json"
        )
    else:
        dataset_json_path = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"]) / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(
            f"Dataset.json not found: {dataset_json_path}. Cannot determine segmentation reader function without."
        )
    dataset_json = load_json(dataset_json_path)
    # I just use the nnunet reader always because it works for all datasets I have so far
    nnunet_reader = determine_reader_writer_from_dataset_json(dataset_json)()
    reader_fn = nnunet_reader.read_seg
    all_labels = dataset_json["labels"]
    include_background = any([isinstance(x, (tuple, list)) for x in all_labels])
    kept_labels = []
    for lab in all_labels:
        if include_background or lab.lower() not in ["bg", "background"]:
            kept_labels.append(lab)
    metric_objs = get_metrics(metric_names, include_background=include_background)
    if not lab_file.exists():
        raise FileNotFoundError(f"Label file not found: {lab_file}")
    metrics, metrics_multi = compute_metrics_for_file(
        metric_objs,
        label_file=lab_file,
        pred_file=pred_file,
        all_labels=list(all_labels.values()),
        seg_reader_fn=reader_fn,
    )
    for k in metrics.keys():
        assert len(metrics[k]) == 1
        curr_result = {
            "metric_name": k,
            "metric_value": metrics[k].item(),
            "metric_type": "single",
            "class": None,
        }
        results.append(curr_result)
    for k in metrics_multi.keys():
        for i in range(len(metrics_multi[k])):
            curr_result = {
                "metric_name": k,
                "metric_value": metrics_multi[k][i],
                "metric_type": "multi",
                "class": kept_labels[i],
            }
            results.append(curr_result)
    return results


# I can't use the nnunet preprocessor directly (cannot pass confidence as segmentation, but need image for cropping) -.-
def preprocess_confidence_map(
    image_files, confid_file, plans_manager, configuration_manager, dataset_json, verbose=False
):
    # This is adapted from nnunet's default preprocessor
    if isinstance(dataset_json, str):
        dataset_json = load_json(dataset_json)

    rw = plans_manager.image_reader_writer_class()

    # load image(s)
    data, properties = rw.read_images(image_files)
    confid, _ = rw.read_images([confid_file])

    assert data.shape[1:] == confid.shape[1:], "Shape mismatch between image and confidence."

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    confid = confid.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    original_spacing = [properties["spacing"][i] for i in plans_manager.transpose_forward]

    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    properties["shape_before_cropping"] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, confid, bbox = crop_to_nonzero(data, confid)
    properties["bbox_used_for_cropping"] = bbox
    # print(data.shape, seg.shape)
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

    # resample
    target_spacing = configuration_manager.spacing  # this should already be transposed

    if len(target_spacing) < len(data.shape[1:]):
        # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
        # in 2d configuration we do not change the spacing between slices
        target_spacing = [original_spacing[0]] + target_spacing
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    # # normalize NOT NEEDED AS DATA IS NOT USED
    # # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # # longer fitting the images perfectly!
    # data = self._normalize(data, confid, configuration_manager,
    #                         plans_manager.foreground_intensity_properties_per_channel)

    # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
    #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
    old_shape = data.shape[1:]
    # data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
    confid = configuration_manager.resampling_fn_seg(
        confid, new_shape, original_spacing, target_spacing
    )
    if verbose:
        print(
            f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
            f"new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}"
        )

    return confid


if __name__ == "__main__":
    main()
