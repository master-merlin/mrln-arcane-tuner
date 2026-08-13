"""Restore the `image_processing_utils_fast` symbols transformers 5.x moved.

WHY: `tencent/Youtu-VL-4B-Instruct` ships remote processor code that does

    from transformers.image_processing_utils_fast import (
        BaseImageProcessorFast, DefaultFastImageProcessorKwargs, SizeDict)

In transformers 5.x that module is no longer a file. It is an alias created by
`_create_module_alias(..., ".image_processing_backends")` with an empty
`_import_structure`, so the import fails with "unknown location". Of the three
names, `BaseImageProcessorFast` survives as a BC alias, `SizeDict` moved to
`transformers.image_utils`, and `DefaultFastImageProcessorKwargs` was deleted
outright.

INVARIANT: this must run before ANY `trust_remote_code=True` load, otherwise
the remote import wins the race and raises. It is installed once from
`CaptionService.__init__`, which every captioning path funnels through.

RETIREMENT: delete this module once tencent's remote code targets transformers
5.x. `test_youtu_vl_shim_is_still_required` fails when that day comes.
"""

from __future__ import annotations

from typing import Optional, TypedDict, Union

import structlog

logger = structlog.get_logger(__name__)

_INSTALLED = False


class DefaultFastImageProcessorKwargs(TypedDict, total=False):
    """Faithful copy of the TypedDict transformers 4.57 exposed.

    Field-for-field identical to the original so remote code that annotates
    with `Unpack[DefaultFastImageProcessorKwargs]` keeps its exact semantics.
    """

    do_resize: Optional[bool]
    size: Optional[dict[str, int]]
    default_to_square: Optional[bool]
    resample: Optional[object]
    do_center_crop: Optional[bool]
    crop_size: Optional[dict[str, int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[Union[int, float]]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_pad: Optional[bool]
    pad_size: Optional[dict[str, int]]
    do_convert_rgb: Optional[bool]
    return_tensors: Optional[object]
    data_format: Optional[object]
    input_data_format: Optional[object]
    device: Optional[object]
    disable_grouping: Optional[bool]


def install_transformers5_compat() -> None:
    """Re-export the moved symbols onto the alias target. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    import transformers.image_processing_backends as backends
    from transformers.image_utils import SizeDict

    if not hasattr(backends, "DefaultFastImageProcessorKwargs"):
        backends.DefaultFastImageProcessorKwargs = DefaultFastImageProcessorKwargs
    if not hasattr(backends, "SizeDict"):
        backends.SizeDict = SizeDict
    # BaseImageProcessorFast is already a BC alias for TorchvisionBackend
    # upstream. Deliberately NOT redefined - shadowing it would silently swap
    # the base class out from under every fast image processor.

    _INSTALLED = True
    logger.debug("transformers5_compat_installed")
