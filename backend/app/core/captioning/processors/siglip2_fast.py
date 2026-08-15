"""
Bundled Siglip2ImageProcessorFast for Youtu-VL.

Source: https://huggingface.co/tencent/Youtu-VL-4B-Instruct/blob/main/image_processing_siglip2_fast.py

Bundled here so it works regardless of HF cache state and allows
runtime injection with custom max_num_patches values.
"""
from typing import List, Optional, Tuple, Union
import math
import torch

from transformers.image_processing_utils import BatchFeature

# transformers 5.x: `image_processing_utils_fast` is now an alias module with an
# empty import structure. BaseImageProcessorFast survives as a BC alias for
# TorchvisionBackend, SizeDict moved to image_utils, and
# DefaultFastImageProcessorKwargs was deleted - we carry our own copy.
from transformers.image_processing_backends import BaseImageProcessorFast
from transformers.image_utils import SizeDict

from app.core.captioning.compat.transformers5 import DefaultFastImageProcessorKwargs
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


def get_image_size_for_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int
) -> Tuple[int, int]:
    """
    Calculate target image dimensions so the total number of patches
    does not exceed max_num_patches.
    """
    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        patch_size = patch_size * 2
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size
        scaled_size = max(patch_size, scaled_size)
        return int(scaled_size)

    scale = 1.0
    while True:
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches > max_num_patches:
            scale -= 0.02
        else:
            break

    return target_height, target_width


def convert_image_to_patches(
    image: "torch.Tensor", patch_size: int, merge_size: int
) -> "torch.Tensor":
    """
    Converts an input image into flattened patches.

    Args:
        image: Input image tensor of shape (channels, height, width)
        patch_size: Size of each square patch (in pixels)
        merge_size: Number of adjacent patches to merge
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels,
        num_patches_height // merge_size,
        merge_size, patch_size,
        num_patches_width // merge_size,
        merge_size, patch_size,
    )
    patched_image = patched_image.permute(1, 4, 2, 5, 3, 6, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    tensor: "torch.Tensor", target_length: int, pad_value: int = 0
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Pad the input tensor along its first dimension to a target length."""
    current_length = tensor.shape[0]
    padding_length = target_length - current_length
    mask = torch.ones((target_length,), dtype=torch.int32)
    if padding_length > 0:
        padding = [0, 0] * (tensor.ndim - 1) + [0, padding_length]
        tensor = torch.nn.functional.pad(
            tensor, padding, mode="constant", value=pad_value
        )
        mask[-padding_length:] = 0
    return tensor, mask


class Siglip2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs, total=False):
    """`total=False` is required here, not cosmetic: per PEP 589, fields a
    TypedDict subclass declares itself default to required (`total=True`)
    regardless of the parent's totality — only fields inherited unchanged
    stay optional. Without it, `patch_size`/`max_num_patches` become
    *required* keys, and huggingface_hub 1.x's `validate_typed_dict` (called
    from `BaseImageProcessorFast.preprocess`) then rejects the internal
    "not supplied, fall back to the class attribute default" sentinel that
    transformers passes through `kwargs` when the caller omits them —
    raising `StrictDataclassFieldValidationError` instead of falling back to
    `Siglip2ImageProcessorFast.patch_size = 16` / `.max_num_patches`.
    """

    patch_size: Optional[int]
    max_num_patches: Optional[int]


class Siglip2ImageProcessorFast(BaseImageProcessorFast):
    """
    Fast Siglip2 image processor using torchvision GPU ops.
    
    Key parameter: max_num_patches controls the maximum number of vision
    patches (tokens) produced. Lower = faster inference but less detail.
    Default 256 matches the official Youtu-VL config.
    """
    resample = PILImageResampling.BILINEAR
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 16
    max_num_patches = 256
    valid_kwargs = Siglip2FastImageProcessorKwargs
    unused_kwargs = ["size", "do_center_crop", "crop_size"]

    def __init__(self, **kwargs: Unpack[Siglip2FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs) -> tuple:
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Siglip2FastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def get_max_image_patches(self, images):
        return 4096 * 6 * 6

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        pixel_masks = []
        pixel_values = []
        spatial_shapes = []

        for i, image in enumerate(images):
            height, width = get_image_size_for_patches(
                image_height=image.shape[1],
                image_width=image.shape[2],
                patch_size=patch_size,
                max_num_patches=max_num_patches,
            )

            side_dict = SizeDict(height=height, width=width)
            image = self.resize(
                image=image, size=side_dict, interpolation=interpolation
            )
            image = self.rescale_and_normalize(
                image, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            patches = convert_image_to_patches(image, patch_size, 2)
            patches, mask = pad_along_first_dim(patches, len(patches))

            num_patches_height = image.shape[1] // patch_size
            num_patches_width = image.shape[2] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)
            pixel_masks.append(mask)

        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_masks = torch.stack(pixel_masks, dim=0)
        spatial_shapes = torch.tensor(spatial_shapes)

        batch_feature = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_masks,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )
        return batch_feature


__all__ = ["Siglip2ImageProcessorFast"]
