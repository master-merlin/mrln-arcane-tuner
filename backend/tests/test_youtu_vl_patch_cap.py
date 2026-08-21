"""Regression test for the Youtu-VL captioning VRAM blow-up.

Root cause (see youtu_vl.py::generate for the full trail): tencent's remote
`YoutuVLProcessor.__call__` (processing_youtu_vl.py, cached under
transformers_modules/tencent/Youtu_hyphen_VL_hyphen_4B_hyphen_Instruct)
declares its OWN `max_image_patches: int = 36864` parameter and ALWAYS
forwards it explicitly to
`self.image_processor(images=images, max_num_patches=max_image_patches, ...)`.
Setting `self.processor.image_processor.max_num_patches = 256` (an instance
attribute) is therefore invisible on the `apply_chat_template` route: that
explicit per-call kwarg always wins over the instance default. Without
threading a `max_image_patches` value through
`apply_chat_template(..., processor_kwargs={...})`, a 4000x3000 image
produces ~36520 vision patches -> a single 79.5 GB allocation -> CUDA OOM.

This test reproduces that exact contract -- trimmed to the one call shape
that matters, verified line-for-line against the cached remote code -- using
the REAL (weight-free) `Siglip2ImageProcessorFast`, so it fails if
`YoutuVLModel.generate()` ever stops threading `processor_kwargs` through
`apply_chat_template`. No model weights are loaded; `self.model` /
`self.processor` are pre-set on the plugin so `generate()` never calls
`load()`.
"""

import torch
from PIL import Image

from app.core.captioning.models.youtu_vl import YoutuVLModel
from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast


class _FakeTokenizer:
    """Just enough of a tokenizer for YoutuVLProcessor.__call__'s text path."""

    def __call__(self, text, **kwargs):
        ids = [[1, 2, 3] for _ in text]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(text), 3, dtype=torch.long),
        }

    def batch_decode(self, *args, **kwargs):
        return ["a caption"]


class _InputsResult(dict):
    """Stands in for the BatchFeature apply_chat_template returns: `.to()`
    is a no-op (everything stays on CPU in this test) and `.input_ids` is
    exposed as an attribute the way BatchFeature exposes dict keys."""

    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeYoutuVLProcessor:
    """Trimmed, faithful reproduction of the ONE contract this bug lives in:
    `__call__`'s own `max_image_patches` default silently overrides whatever
    is set on `self.image_processor` unless the caller threads a value
    through explicitly. `apply_chat_template` mirrors
    transformers.ProcessorMixin.apply_chat_template's tokenize branch:
    extract images from message content, then call `self(text=.., images=..,
    **processor_kwargs)` -- NOT **kwargs, which is the exact distinction the
    real fix in youtu_vl.py depends on.

    `__call__` records what it actually received/produced onto `self`
    (`last_max_image_patches`, `last_image_inputs`) instead of relying on a
    monkeypatched spy: patching `__call__` on an *instance* would not
    intercept `self(...)` call syntax anyway (Python resolves dunder methods
    via `type(self).__call__`, bypassing the instance `__dict__`).
    """

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.tokenizer = _FakeTokenizer()
        self.last_max_image_patches = None
        self.last_image_inputs = None

    def __call__(self, text=None, images=None, max_image_patches: int = 36864, **kwargs):
        image_inputs = self.image_processor(
            images=images, max_num_patches=max_image_patches, return_tensors="pt"
        )
        self.last_max_image_patches = max_image_patches
        self.last_image_inputs = image_inputs
        text_inputs = self.tokenizer(text)
        return {**text_inputs, **image_inputs}

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=None,
        **kwargs,
    ):
        processor_kwargs = processor_kwargs or {}
        images = []
        for message in messages:
            for block in message.get("content", []):
                if block.get("type") == "image":
                    images.append(block["image"])
        out = self(text=["<prompt>"], images=[images], **processor_kwargs)
        return _InputsResult(out)

    def batch_decode(self, *args, **kwargs):
        return ["a caption"]


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


def _huge_image() -> Image.Image:
    # Same order of magnitude as the measured-OOM case (4000x3000): large
    # enough that an uncapped call produces tens of thousands of patches.
    return Image.new("RGB", (4000, 3000), color=(80, 120, 200))


def _write_huge_image(tmp_path) -> str:
    """A REAL file on disk, not a dummy nonexistent path: `img_input=` and
    (pre-fix) the chat-template image content both carry `image_path`
    around as a plain string that gets loaded from disk independently of
    the in-memory PIL object passed to `generate()`. A nonexistent path
    would make the pre-fix code crash on the load instead of exercising the
    actual uncapped-patches bug, which would make this test fail for the
    wrong reason."""
    path = tmp_path / "huge.png"
    _huge_image().save(path)
    return str(path)


def _plugin_with_fakes() -> YoutuVLModel:
    plugin = YoutuVLModel(service=None)
    plugin.device = "cpu"
    plugin.model = _FakeModel()
    plugin.processor = _FakeYoutuVLProcessor(Siglip2ImageProcessorFast(max_num_patches=256))
    return plugin


def test_generate_caps_vision_patches_through_apply_chat_template(tmp_path):
    """The production entry point (YoutuVLModel.generate) must bound the
    number of vision patches for a large image, via the real
    Siglip2ImageProcessorFast cap math -- not merely by asking for it."""
    plugin = _plugin_with_fakes()

    image_path = _write_huge_image(tmp_path)
    params = {"image_path": image_path, "max_num_patches": 256}

    caption = plugin.generate(_huge_image(), params)

    assert caption == "a caption"
    # The bug: without processor_kwargs threading the cap through,
    # last_max_image_patches would be the remote code's own 36864 default
    # and spatial_shapes would reflect an unbounded ~36520-patch grid for a
    # 4000x3000 image.
    assert plugin.processor.last_max_image_patches == 256
    num_patches = int(plugin.processor.last_image_inputs["spatial_shapes"][0].prod().item())
    assert num_patches <= 256, (
        f"generate() let {num_patches} vision patches through for a large "
        "image -- the max_num_patches cap is not reaching the image "
        "processor via apply_chat_template. This is the exact VRAM "
        "blow-up regression (4000x3000 -> ~36520 patches -> CUDA OOM)."
    )


def test_generate_reduces_patches_relative_to_uncapped_default(tmp_path):
    """PROVE THE NEGATIVE: a plugin whose generate() does NOT thread
    processor_kwargs through (the pre-fix shape) would leave
    max_image_patches at the remote code's 36864 default and blow the cap
    wide open. This pins that the fixed code path measurably differs from
    that unbounded baseline, not just that it happens to be <= 256."""
    plugin = _plugin_with_fakes()
    image_processor = plugin.processor.image_processor

    # Uncapped baseline: call the real image processor exactly the way the
    # buggy (pre-fix) apply_chat_template route did -- no explicit
    # max_num_patches override reaching it beyond the remote __call__'s own
    # 36864 default.
    uncapped = image_processor(images=[_huge_image()], max_num_patches=36864, return_tensors="pt")
    uncapped_patches = int(uncapped["spatial_shapes"][0].prod().item())
    assert uncapped_patches > 256, "fixture assumption: a 4000x3000 image must exceed the 256 cap uncapped"

    params = {"image_path": _write_huge_image(tmp_path), "max_num_patches": 256}
    plugin.generate(_huge_image(), params)

    capped_patches = int(plugin.processor.last_image_inputs["spatial_shapes"][0].prod().item())
    assert capped_patches < uncapped_patches
    assert capped_patches <= 256
