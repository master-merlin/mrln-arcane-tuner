# backend/tests/test_caption_target.py
"""Unit tests for caption_target.resolve_caption_target."""

import pytest

from app.engine.core.caption_target import CaptionTarget, resolve_caption_target
from app.engine.models.registry import registry


@pytest.fixture(autouse=True)
def _ensure_registry_loaded():
    """Populate the model registry so family lookups resolve.

    The registry singleton does not auto-discover at import; the app calls
    ``initialize()`` at startup. ``initialize()`` is idempotent and loads the
    bundled per-family YAML definitions via ``__file__``-relative paths (no CWD
    dependency), so calling it here makes these tests deterministic.
    """
    registry.initialize()


def _first_definition_id_for_family(family: str) -> str | None:
    for def_id in registry.list_models():
        defn = registry.get_definition(def_id)
        if defn is not None and defn.family == family:
            return def_id
    return None


def test_sdxl_resolves_to_clip_77():
    def_id = _first_definition_id_for_family("sdxl")
    if def_id is None:
        pytest.skip("no sdxl definition registered")
    target = resolve_caption_target(def_id)
    assert isinstance(target, CaptionTarget)
    assert target.family == "sdxl"
    assert target.tokenizer_kind == "clip"
    assert target.tokenizer_id == "openai/clip-vit-large-patch14"
    assert target.raw_max_length == 77
    assert target.usable_limit == 75  # 77 minus BOS/EOS


def test_flux1_resolves_to_t5():
    def_id = _first_definition_id_for_family("flux1")
    if def_id is None:
        pytest.skip("no flux1 definition registered")
    target = resolve_caption_target(def_id)
    assert target.tokenizer_kind == "t5"
    assert target.tokenizer_id == "google/t5-v1_1-xxl"
    assert target.raw_max_length >= 1
    assert target.usable_limit == max(target.raw_max_length - 1, 1)


def test_unknown_family_falls_back_to_heuristic():
    # microsoft_lens / flux2 etc. are not precisely tokenized; verify graceful fallback.
    for fam in ("flux2", "microsoft_lens"):
        def_id = _first_definition_id_for_family(fam)
        if def_id is None:
            continue
        target = resolve_caption_target(def_id)
        assert target.tokenizer_kind == "heuristic"
        assert target.tokenizer_id is None
        assert target.usable_limit == target.raw_max_length
        return
    pytest.skip("no fallback-family definition registered")


def test_unknown_definition_id_raises():
    with pytest.raises(ValueError):
        resolve_caption_target("does-not-exist-xyz")
