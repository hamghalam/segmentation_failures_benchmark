import multiprocessing
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

# import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from loguru import logger
from nnunetv2.imageio.natural_image_reager_writer import NaturalImage2DIO
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from pytorch_lightning.callbacks import Callback
from skimage import io


class PixelConfidenceWriter(Callback):
    """Save confidence maps for pixel CSFs."""

    def __init__(
        self,
        output_dir: str,
        confid_name: str = None,
        num_export_workers: int = 1,
        keep_preprocessed=False,
        precision=32,
    ):
        self.output_path = Path(output_dir)
        self.output_path.mkdir()
        self.confid_name = confid_name
        self.keep_preprocessed = keep_preprocessed
        self.mp_pool = None
        self.precision = precision
        if num_export_workers > 1:
            self.mp_pool = multiprocessing.get_context("spawn").Pool(num_export_workers)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        confid_dict = outputs.get("confidence_pixel", None)
        if confid_dict is None:
            if batch_idx == 0:
                logger.warning(
                    "Couldn't find key pixel confidence in batch output. No confidence masks will be saved."
                )
            return
        if isinstance(confid_dict, dict) and self.confid_name is not None:
            confid_dict = {self.confid_name: confid_dict[self.confid_name]}
        if hasattr(trainer.datamodule, "nnunet_trainer"):
            for confid_name in confid_dict:
                confid_dict[confid_name] = trainer.datamodule.post_inference_process(
                    confid_dict[confid_name], batch
                )
            self._export_confid_maps_nnunet(confid_dict, batch, trainer.datamodule.nnunet_trainer)
        else:
            self._export_confid_maps_simple(confid_dict, batch)
        if self.keep_preprocessed:
            # save confidence maps in preprocessed space
            # TODO
            raise NotImplementedError

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _export_confid_maps_nnunet(self, confid_dict, batch, nnunet_trainer: nnUNetTrainer):
        # TODO nnunet has some precautions against too many exports at once. Maybe add later
        export_args = []
        for confid_name, confid_arr in confid_dict.items():
            # TODO make more efficient by saving multiple confid maps to one file (channels)
            curr_output_dir = self.output_path / confid_name
            curr_output_dir.mkdir(exist_ok=True)
            for idx, case_id in enumerate(batch["keys"]):
                export_args.append(
                    (
                        confid_arr[idx],
                        nnunet_trainer.plans_manager,
                        nnunet_trainer.configuration_manager,
                        batch["properties"][idx],
                        str(curr_output_dir / case_id),
                        4,
                        self.precision,
                    )
                )
        if self.mp_pool is None:
            for args in export_args:
                resample_uncrop_save_confidence_map(*args)
        else:
            self.mp_pool.starmap(resample_uncrop_save_confidence_map, export_args)

    def _export_confid_maps_simple(self, confid_dict, batch):
        if not isinstance(confid_dict, dict):
            confid_dict = {"unnamed": confid_dict}
        affine = np.eye(4)
        if hasattr(batch["data"], "meta"):
            affine = batch["data"].meta["affine"][0].cpu().numpy()
        for confid_name, confid_arr in confid_dict.items():
            curr_output_dir = self.output_path / confid_name
            curr_output_dir.mkdir(exist_ok=True)
            # assume all have the same affine matrix in batch
            for idx, case_id in enumerate(batch["keys"]):
                img_shape = batch["data"].shape[2:]  # remove batch and channel dimension
                dtype_mapping = {16: torch.float16, 32: torch.float32, 64: torch.float64}
                curr_confid = confid_arr[idx].to(dtype=dtype_mapping[self.precision])
                if curr_confid.shape != img_shape:
                    raise ValueError(
                        f"Expected that confidence mask shape == label mask shape, but got {curr_confid.shape} and {img_shape}."
                    )
                nib.save(
                    nib.Nifti1Image(curr_confid.cpu().numpy(), affine=affine),
                    curr_output_dir / f"{case_id}.nii.gz",
                )

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self.mp_pool is not None:
            self.mp_pool.close()


# slightly modified from nnunet
def resample_uncrop_save_confidence_map(
    confid_map: Union[torch.Tensor, np.ndarray],
    plans_manager,
    configuration_manager,
    properties_dict: dict,
    output_file_truncated: str,
    num_threads_torch: int = 4,
    fp_precision=32,
):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing)
        == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [properties_dict["spacing"][0], *configuration_manager.spacing]
    )
    # This will result in interpolation order == 3. Not sure if ideal but let's keep it for now
    confid_map = configuration_manager.resampling_fn_data(
        torch.unsqueeze(confid_map, 0),  # add channel dimension
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        properties_dict["spacing"],
    )
    # tensor may be torch.Tensor but we continue with numpy
    if isinstance(confid_map, torch.Tensor):
        confid_map = confid_map.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    max_confidence = np.max(confid_map)
    # This is the easiest way here; it would be better if I knew the maximum *possible* confidence value
    confid_map_postprocessed = max_confidence * np.ones(
        properties_dict["shape_before_cropping"],
        dtype=confid_map.dtype,
    )
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    confid_map_postprocessed[slicer] = confid_map
    del confid_map

    # revert transpose. This assumes confidence map has no channel dimension
    confid_map_postprocessed = confid_map_postprocessed.transpose(plans_manager.transpose_backward)
    # casting
    if fp_precision == 16:
        confid_map_postprocessed = confid_map_postprocessed.astype(np.float16)
    elif fp_precision == 32:
        confid_map_postprocessed = confid_map_postprocessed.astype(np.float32)
    elif fp_precision == 64:
        confid_map_postprocessed = confid_map_postprocessed.astype(np.float64)
    else:
        raise ValueError(f"Unsupported floating point precision {fp_precision}")
    torch.set_num_threads(old_threads)
    # Save as nifti for compatibility with nnunet -> nibabel_stuff/original_affine
    assert confid_map_postprocessed.ndim == 3
    file_ending = ".nii.gz"
    # I cannot use the nnunet readerwriter class because it only has a write_seg method.
    if len(configuration_manager.spacing) == 2:
        assert confid_map_postprocessed.shape[0] == 1
        confid_map_postprocessed = confid_map_postprocessed[0]
        file_ending = ".tiff"
    write_confid_map(
        confid_map_postprocessed,
        output_file_truncated + file_ending,
        properties_dict,
        plans_manager,
    )
    return confid_map_postprocessed


def write_confid_map(img, output_fname, properties, plans_manager):
    io_cls = plans_manager.image_reader_writer_class
    if io_cls is NibabelIO:
        write_map_nibabelio(img, output_fname, properties)
    elif io_cls is NibabelIOWithReorient:
        write_map_nibabelio_reorient(img, output_fname, properties)
    elif io_cls is SimpleITKIO:
        write_map_sitk(img, output_fname, properties)
    elif io_cls is NaturalImage2DIO:
        write_map_skimage(img, output_fname)
    else:
        raise ValueError(f"Unsupported image reader writer class {io_cls}")


def write_map_nibabelio(img: np.ndarray, output_fname: str, properties: dict) -> None:
    # revert transpose
    img = img.transpose((2, 1, 0))
    img_nib = nib.Nifti1Image(img, affine=properties["nibabel_stuff"]["original_affine"])
    nib.save(img_nib, output_fname)


def write_map_nibabelio_reorient(img: np.ndarray, output_fname: str, properties: dict) -> None:
    # revert transpose
    img = img.transpose((2, 1, 0))

    img_nib = nib.Nifti1Image(img, affine=properties["nibabel_stuff"]["reoriented_affine"])
    img_nib_reoriented = img_nib.as_reoriented(
        nib.io_orientation(properties["nibabel_stuff"]["original_affine"])
    )
    assert np.allclose(
        properties["nibabel_stuff"]["original_affine"], img_nib_reoriented.affine
    ), "restored affine does not match original affine"
    nib.save(img_nib_reoriented, output_fname)


def write_map_sitk(img: np.ndarray, output_fname: str, properties: dict) -> None:
    assert img.ndim == 3, "If you are exporting a 2d map, please provide it as shape 1,x,y"
    output_dimension = len(properties["sitk_stuff"]["spacing"])
    assert 1 < output_dimension < 4
    if output_dimension == 2:
        img = img[0]

    itk_image = sitk.GetImageFromArray(img)
    itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
    itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
    itk_image.SetDirection(properties["sitk_stuff"]["direction"])

    sitk.WriteImage(itk_image, output_fname, True)


def write_map_skimage(img: np.ndarray, output_fname: str) -> None:
    io.imsave(output_fname, img)
