"""Florence-2 runs on the native transformers implementation, not remote code."""

import pathlib

import pytest
import transformers
from huggingface_hub.constants import HF_HUB_CACHE

# Resolve the cache dir the way huggingface_hub itself does (respects
# HF_HOME/HF_HUB_CACHE) instead of a hardcoded machine-specific path -- a
# literal path here would silently skip these tests, the only real-code
# evidence for the native Florence-2 processor, on any other machine or CI.
#
# florence-community/Florence-2-large, not microsoft/Florence-2-large: the
# microsoft repo still ships the legacy remote-code weight layout, which the
# native Florence2ForConditionalGeneration class cannot load
# (ignore_mismatched_sizes errors at from_pretrained). florence-community is
# the natively-converted repo -- no auto_map, no remote code.
_CACHED = (pathlib.Path(HF_HUB_CACHE) / "models--florence-community--Florence-2-large").exists()
needs_cache = pytest.mark.skipif(_CACHED is False, reason="Florence-2 not in local HF cache")


def test_native_florence2_classes_exist():
    """5.14.1 ships Florence-2 natively - this is what lets us drop remote code."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "florence2" in CONFIG_MAPPING_NAMES
    assert hasattr(transformers, "Florence2ForConditionalGeneration")
    assert hasattr(transformers, "Florence2Processor")


@needs_cache
def test_processor_produces_correct_tensors():
    """Observable output, not kwargs. Unlike microsoft/Florence-2-large (whose
    RobertaTokenizer predated native support and needed image_token registered
    by hand), florence-community/Florence-2-large's tokenizer already carries
    image_token / image_token_id, so plain AutoProcessor.from_pretrained is
    enough -- no _build_native_processor shim. Shapes are unchanged from the
    old repo: 577 image tokens + 14 text tokens = 591 input ids, verified live
    against this repo."""
    from transformers import AutoProcessor
    from app.core.captioning.models.florence2 import Florence2Model
    from PIL import Image

    processor = AutoProcessor.from_pretrained(Florence2Model.MODEL_PATH)

    out = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=Image.new("RGB", (512, 512), (128, 128, 128)),
        return_tensors="pt",
    )
    assert tuple(out["pixel_values"].shape) == (1, 3, 768, 768)
    assert out["input_ids"].shape[1] == 591


@needs_cache
def test_image_token_id_is_derived_not_hardcoded():
    """The id depends on the tokenizer's vocab. florence-community's vocab
    assigns <image> a different id (51289) than the old hand-registered
    microsoft/Florence-2-large shim did (50265) -- proof this must stay
    derived, never hardcoded, or it silently breaks on exactly this kind of
    repo swap."""
    from transformers import AutoProcessor
    from app.core.captioning.models.florence2 import Florence2Model

    processor = AutoProcessor.from_pretrained(Florence2Model.MODEL_PATH)
    assert processor.image_token == "<image>"
    assert processor.image_token_id == processor.tokenizer.convert_tokens_to_ids("<image>")
    assert processor.image_token_id == 51289


def test_no_remote_code_in_florence2_module():
    """Negative test: the whole point of going native is that we stop executing
    code downloaded from the hub. If trust_remote_code comes back, this fails."""
    # Anchored on this test file's location, not the pytest rootdir/CWD: a
    # CWD-relative path here would raise FileNotFoundError if pytest ever runs
    # from a different directory, rather than silently skipping (see the
    # sibling kwarg-scan test in test_transformers5_compat.py for the more
    # dangerous, vacuously-passing version of this mistake).
    backend_root = pathlib.Path(__file__).resolve().parents[1]
    source = (backend_root / "app/core/captioning/models/florence2.py").read_text(
        encoding="utf-8"
    )
    assert "trust_remote_code" not in source
    assert "_patch_florence2_kv_cache" not in source
