"""Florence-2 runs on the native transformers implementation, not remote code."""

import pathlib

import pytest
import transformers

_CACHED = pathlib.Path(
    r"D:\AI\huggingface\hub\hub\models--microsoft--Florence-2-large"
).exists()
needs_cache = pytest.mark.skipif(_CACHED is False, reason="Florence-2 not in local HF cache")


def test_native_florence2_classes_exist():
    """5.14.1 ships Florence-2 natively - this is what lets us drop remote code."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "florence2" in CONFIG_MAPPING_NAMES
    assert hasattr(transformers, "Florence2ForConditionalGeneration")
    assert hasattr(transformers, "Florence2Processor")


@needs_cache
def test_processor_produces_correct_tensors():
    """Observable output, not kwargs: the cached repo's RobertaTokenizer predates
    native support and lacks image_token, so the loader must add it. 577 image
    tokens + 14 text tokens = 591 input ids."""
    from app.core.captioning.models.florence2 import Florence2Model
    from PIL import Image

    model = Florence2Model(service=None)
    processor = model._build_native_processor()

    out = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=Image.new("RGB", (512, 512), (128, 128, 128)),
        return_tensors="pt",
    )
    assert tuple(out["pixel_values"].shape) == (1, 3, 768, 768)
    assert out["input_ids"].shape[1] == 591


@needs_cache
def test_image_token_id_is_derived_not_hardcoded():
    """The id depends on the tokenizer's vocab; hardcoding 50265 breaks on any
    repo revision that adds tokens."""
    from app.core.captioning.models.florence2 import Florence2Model

    model = Florence2Model(service=None)
    processor = model._build_native_processor()
    assert processor.image_token == "<image>"
    assert processor.image_token_id == processor.tokenizer.convert_tokens_to_ids("<image>")


def test_no_remote_code_in_florence2_module():
    """Negative test: the whole point of going native is that we stop executing
    code downloaded from the hub. If trust_remote_code comes back, this fails."""
    source = pathlib.Path("app/core/captioning/models/florence2.py").read_text(encoding="utf-8")
    assert "trust_remote_code" not in source
    assert "_patch_florence2_kv_cache" not in source
