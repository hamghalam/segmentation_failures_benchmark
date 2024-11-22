from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import ensure_tuple
from tifffile import TiffFile


class TiffReader(ImageReader):
    def __init__(self, rgb: bool, **kwargs):
        self.rgb = rgb  # assume 2d rgb image
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return is_supported_format(filename, ["tif", "tiff"])

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_: list[TiffFile] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img_.append(TiffFile(name, **kwargs_))  # type: ignore
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: an image object loaded from an image file or a list of image objects.

        """
        img_array: list[np.ndarray] = []
        meta_dict = {"original_channel_dim": float("nan")}

        for curr_img in ensure_tuple(img):
            curr_arr = curr_img.asarray()
            assert len(curr_arr.shape) == 2
            img_array.append(curr_arr)
            curr_img.close()
            # not sure if necessary, but TifFile is used as a context manager in the examples
        if len(img_array) == 1:
            img_array = img_array[0]
        else:
            img_array = np.stack(img_array, axis=0)
            meta_dict["original_channel_dim"] = 0
        return img_array, meta_dict
