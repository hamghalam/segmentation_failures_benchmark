"""
Trying out napari with the FeTS data. I like it :)
"""

# from argparse import ArgumentParser
from argparse import ArgumentParser
from pathlib import Path

import napari
import numpy as np
import SimpleITK as sitk
from loguru import logger

from segmentation_failures.utils.io import load_json


def main(data_root: Path, train=True, shift=None):
    # read info file
    dataset_dict = load_json(data_root / "dataset.json")
    if train:
        image_path = data_root / "imagesTr"
        label_path = data_root / "labelsTr"
    else:
        if shift:
            shift = "_" + shift.lower()
        else:
            shift = ""
        image_path = data_root / f"imagesTs{shift}"
        label_path = data_root / f"labelsTs{shift}"

    color_list = ["blue", "green", "red", "cyan", "yellow"]
    label_colors = {int(k): color_list[i] for i, k in enumerate(dataset_dict["labels"])}
    label_colors.update({0: None})
    #     0: None,
    #     1: "royalblue",
    #     2: "orange",
    # }
    for case in label_path.iterdir():
        case_name = case.name.split(".")[0]
        logger.info(case_name)
        # load data (flair)
        img_npy = []
        for mod in dataset_dict["modalities"]:
            file_path = str(image_path / f"{case_name}_{int(mod):04d}.nii.gz")
            logger.debug(f"Loading image {file_path}")
            img = sitk.ReadImage(file_path)
            img_npy.append(sitk.GetArrayFromImage(img))
        img_npy = np.stack(img_npy)

        img_viewer = napari.view_image(
            img_npy,
            rgb=False,
            channel_axis=0,
            name=list(dataset_dict["modalities"].values()),
            visible=[i == 0 for i in range(len(img_npy))],
            colormap="gray",
        )

        seg = sitk.ReadImage(str(case))
        seg_npy = sitk.GetArrayFromImage(seg).astype(int)
        img_viewer.add_labels(seg_npy, name="segmentation", opacity=0.5, color=label_colors)
        try:
            napari.run()
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--testshift", type=str, default=None, required=False)
    args = parser.parse_args()
    data_root = Path(
        # "/Users/e290-mb003-wl/tmp_sshfs/Datasets/segmentation_failures/Task000D2_Example"
        "/Users/e290-mb003-wl/datasets/Task001_simple_brats"
    )
    # shift = "biasfield"
    shift = None
    main(data_root, train=not args.testset, shift=args.testshift)
