"""Restore transformers 5.x symbols that third-party `trust_remote_code=True`
model code (tencent/Youtu-VL-4B-Instruct) still imports by their pre-5.x name.

WHY (image processing): the remote processor code does

    from transformers.image_processing_utils_fast import (
        BaseImageProcessorFast, DefaultFastImageProcessorKwargs, SizeDict)

In transformers 5.x that module is no longer a file. It is an alias created by
`_create_module_alias(..., ".image_processing_backends")` with an empty
`_import_structure`, so the import fails with "unknown location". Of the three
names, `BaseImageProcessorFast` survives as a BC alias, `SizeDict` moved to
`transformers.image_utils`, and `DefaultFastImageProcessorKwargs` was deleted
outright.

WHY (rope): the remote modeling code does

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    ...
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]  # self.rope_type == "default"

transformers 5.x's rope refactor deleted the `"default"` entry (and its backing
`_compute_default_rope_parameters`) from `ROPE_INIT_FUNCTIONS` outright — native
5.x model classes no longer dict-dispatch the default case. Remote code that
still does raises `KeyError: 'default'` on every model construction, whether or
not `config.rope_scaling` is populated.

INVARIANT: this must run before ANY `trust_remote_code=True` load, otherwise
the remote import wins the race and raises. It is installed once from
`CaptionService.__init__`, which every captioning path funnels through.

RETIREMENT: delete this module once tencent's remote code targets transformers
5.x. `test_youtu_vl_shim_is_still_required` fails when that day comes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, Union

import structlog

if TYPE_CHECKING:
    import torch

logger = structlog.get_logger(__name__)

_INSTALLED = False


def _compute_default_rope_parameters(
    config=None,
    device: "torch.device | None" = None,
    seq_len: int | None = None,
) -> tuple["torch.Tensor", float]:
    """Byte-for-byte port of transformers 4.57's
    `modeling_rope_utils._compute_default_rope_parameters`.

    Ported rather than re-derived so remote code dispatching to `"default"`
    gets numerically identical inverse frequencies to what it got pre-upgrade.
    `seq_len` is accepted (unused) only to match the call signature every
    other entry in `ROPE_INIT_FUNCTIONS` uses.
    """
    import torch

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, attention_factor


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
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if not hasattr(backends, "DefaultFastImageProcessorKwargs"):
        backends.DefaultFastImageProcessorKwargs = DefaultFastImageProcessorKwargs
    if not hasattr(backends, "SizeDict"):
        backends.SizeDict = SizeDict
    # BaseImageProcessorFast is already a BC alias for TorchvisionBackend
    # upstream. Deliberately NOT redefined - shadowing it would silently swap
    # the base class out from under every fast image processor.

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    _INSTALLED = True
    logger.debug("transformers5_compat_installed")
