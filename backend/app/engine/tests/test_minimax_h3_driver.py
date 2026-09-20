"""minimax_h3 family registration + capability flags + driver non-training surface."""

from __future__ import annotations

import ast
from pathlib import Path

from app.engine.models.registry import ModelRegistry


def _reload_registry() -> type[ModelRegistry]:
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()
    return ModelRegistry


def test_family_is_auto_discovered():
    assert "minimax_h3" in _reload_registry()._families


def test_capability_flags_declare_video_and_audio():
    family_cls = _reload_registry()._families["minimax_h3"]
    caps = family_cls.capability_overrides
    assert caps["is_video"] is True
    assert caps["has_audio"] is True
    assert caps["has_image_encoder"] is True
    # H3 is single-stream: there is no second expert to schedule.
    assert caps["dual_expert"] is False
    # supports_train_te is intentionally NOT asserted here: only sdxl may put
    # that key in capability_overrides (test_only_sdxl_overrides_train_te);
    # minimax_h3 relies on the latent_diffusion archetype's False default.
    # The ~66.7 GB Qwen3-VL TE must be cacheable.
    assert caps["te_cache"] is True


# ---------------------------------------------------------------------------
# Task 6: driver non-training surface
# ---------------------------------------------------------------------------

def _driver(def_id: str = "minimax-h3-t2va"):
    import torch

    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return MiniMaxH3Driver(ModelRegistry._definitions[def_id], torch.device("cpu"))


def test_definition_ships_curated_target_list_matching_driver():
    """The nucleus_image contract: YAML and driver must agree EXACTLY.

    If they drift, the introspector's exhaustive catalog silently overwrites
    the curated list at first model load and the run adapts the wrong modules.
    """
    import pathlib
    import yaml

    defs_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "models" / "families" / "minimax_h3" / "definitions"
    )
    for path in defs_dir.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        driver = _driver(data["id"])
        assert sorted(driver.get_lora_targets()) == sorted(
            data["lora_targetable_modules"]
        ), f"{data['id']}: driver/YAML target-list drift"


def test_block_topology_is_read_from_the_definition():
    topo = _driver().get_block_topology()
    assert sum(entry["count"] for entry in topo) == 52


def test_init_scheduler_still_returns_none():
    """The deliberate exception to "raise, never a plausible default":
    ``None`` IS the real answer (H3 trains with flow matching, no external
    scheduler) — and the ONLY body ``hook_dispatch.TRIVIAL_BODIES`` treats as
    the ``init_scheduler`` no-op baseline. Any other body (including a raise)
    would enroll minimax_h3 in the auto-delegation allowlist and trip
    ``test_autodelegated_family_hook_set_is_exactly_expected``. See
    ``driver.py``'s module docstring."""
    assert _driver().init_scheduler() is None


def test_forward_pass_refuses_loudly_in_pr0():
    import pytest

    with pytest.raises(NotImplementedError, match="PR1"):
        _driver().forward_pass(None, None, None, {})


def test_get_saver_refuses_loudly_in_pr0():
    import pytest

    with pytest.raises(NotImplementedError, match="PR1"):
        _driver().get_saver()


def test_resolve_loading_dtype_is_bf16():
    import torch

    assert _driver().resolve_loading_dtype() is torch.bfloat16


def test_get_te_lora_targets_is_empty_te_never_trains():
    assert _driver().get_te_lora_targets() == []


def test_assign_components_wires_the_five_manifest_keys():
    driver = _driver()
    components = {
        "tokenizer": object(),
        "text_encoder": object(),
        "vae": object(),
        "audio_vae": object(),
        "transformer": object(),
    }
    driver.assign_components(components)
    assert driver.get_components() is components
    assert driver.get_primary_model() is components["transformer"]
    assert driver.get_text_encoders() == {"text_encoder": components["text_encoder"]}


# ---------------------------------------------------------------------------
# Plan row 2.1: text encoding (Qwen3-VL layer tap), the TE cache key, the
# release-before-DiT check, and the INCREMENTAL PR0-refusal guard.
# ---------------------------------------------------------------------------

def _text_driver(def_id: str = "minimax-h3-t2va", *, tokenizer=None, num_layers: int = 64):
    """A driver with a stub Qwen3-VL + processor assigned, no weights."""
    from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL

    driver = _driver(def_id)
    te = StubQwen3VL(num_hidden_layers=num_layers)
    components = {
        "tokenizer": StubProcessor(tokenizer),
        "text_encoder": te,
        "vae": object(),
        "audio_vae": object(),
    }
    driver.assign_components(components)
    return driver, te


def test_encode_text_returns_the_layer_tap_embedding():
    """The embedding IS ``hidden_states[te.hidden_state_tap_index]`` (50 for the
    shipped definitions: MiniMax README "hidden states from its 50th layer";
    ai-toolkit ``text_encoder.py`` "unnormalized hidden_states[50]") — read off
    the returned tensor, whose every element equals the tap index by the stub's
    construction. Raw tokens (no chat template), the caption trimmed to
    ``te.max_length`` (512, ai-toolkit's evidenced default), the empty prompt
    encoded as ONE pad token, per-caption masks re-padded to the batch max."""
    import torch

    driver, te = _text_driver()
    arch = driver.definition.architecture_params
    assert arch["te.hidden_state_tap_index"] == 50
    assert arch["te.max_length"] == 512

    long_caption = " ".join(["word"] * 600)  # 600 tokens under the stub tokenizer
    out = driver.encode_text(["a cat sits", "", long_caption], torch.bfloat16)

    emb, mask = out.embeddings, out.attention_mask
    assert emb.dtype is torch.bfloat16
    assert emb.shape == (3, 512, te.hidden_size), emb.shape
    assert mask.shape == (3, 512)
    # Every real token row carries the tap index, not the last layer (64).
    assert torch.all(emb[mask.bool()].float() == 50.0), "not the layer-50 tap"
    # Lengths: 3 tokens, 1 pad token for the empty prompt, 512 after the trim.
    assert mask.sum(dim=1).tolist() == [3, 1, 512]
    # Padding rows are zero, not the tap value.
    assert torch.all(emb[0, 3:] == 0)
    # Raw, per-caption encodes: 3 ids, 1 pad id, 512 after the trim (never 600).
    assert sorted(t.shape[1] for t in te.seen_input_ids) == [1, 3, 512]


def test_te_cache_key_includes_tap_and_tokenizer():
    """Two encoders that differ ONLY in the tap index, or ONLY in the tokenizer
    vocabulary, must never share a cached embedding (key-collision test)."""
    from app.engine.tests.h3_text_stubs import StubTokenizer

    base, _ = _text_driver()
    same, _ = _text_driver()
    other_tap, _ = _text_driver()
    other_tap.definition = other_tap.definition.model_copy(
        update={
            "architecture_params": {
                **other_tap.definition.architecture_params,
                "te.hidden_state_tap_index": 49,
            }
        }
    )
    other_tok, _ = _text_driver(tokenizer=StubTokenizer(vocab_tag="v2"))

    prompt = "a cat sits"
    assert base.te_cache_key(prompt) == same.te_cache_key(prompt)
    assert base.te_cache_key(prompt) != base.te_cache_key("a dog sits")
    assert base.te_cache_key(prompt) != other_tap.te_cache_key(prompt), (
        "tap index missing from the key: layer-49 and layer-50 embeddings collide"
    )
    assert base.te_cache_key(prompt) != other_tok.te_cache_key(prompt), (
        "tokenizer fingerprint missing from the key"
    )
    assert "tap=50" in base.te_cache_key(prompt)


def test_text_encoder_released_before_dit_load():
    """The DiT never loads while the 63 GB encoder is resident: the check
    names the bytes; after ``release_text_encoders()`` the driver reports no
    encoder and the check passes."""
    import pytest

    driver, te = _text_driver()
    with pytest.raises(RuntimeError, match=r"text encoder still resident \(.* GB\) at DiT load"):
        driver.assert_text_encoder_released()
    assert driver.text_encoder_weight_bytes() == te.weight.numel() * te.weight.element_size()

    driver.release_text_encoders()
    assert driver.get_text_encoders() == {}
    assert driver.text_encoder is None
    driver.assert_text_encoder_released()  # no raise


def test_cache_fingerprint_includes_pixel_adapter_and_vae_identity(monkeypatch):
    """Row 2.3 (D10 "keyed on every input"): the RAW VAE latent depends on the
    pixel convention (`PIXEL_ADAPTER_VERSION`), the VAE class and its config —
    and on nothing else (a sigma shift or the packed-layout version must NOT
    re-encode a dataset)."""
    from app.engine.models.families.minimax_h3 import pixel_adapter
    from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
    from app.engine.tests.h3_text_stubs import StubVisualVAE

    def _with_vae(inner):
        driver = _driver()
        driver.assign_components({"vae": H3PixelAdaptedVAE(inner)})
        return driver

    base = _with_vae(StubVisualVAE())
    same = _with_vae(StubVisualVAE())
    fp = base.latent_cache_fingerprint()
    assert len(fp) == 8 and int(fp, 16) >= 0
    assert fp == same.latent_cache_fingerprint()

    other_config = _with_vae(StubVisualVAE(config={"latent_channels": 64}))
    assert other_config.latent_cache_fingerprint() != fp, "VAE config not in the key"

    class OtherVAE(StubVisualVAE):
        pass

    assert _with_vae(OtherVAE()).latent_cache_fingerprint() != fp, "VAE class not in the key"

    monkeypatch.setattr(pixel_adapter, "PIXEL_ADAPTER_VERSION", pixel_adapter.PIXEL_ADAPTER_VERSION + 1)
    assert _with_vae(StubVisualVAE()).latent_cache_fingerprint() != fp, (
        "two pixel conventions share a key"
    )


# ── The INCREMENTAL PR0-refusal guard ─────────────────────────────────────
#
# Rows 2.4 (forward_pass), 2.5 (_setup_family) and 2.8 (get_saver) each remove
# their entry; 2.8 asserts the set is EMPTY and deletes the constant.
REMAINING_PR0_REFUSALS: set[tuple[str, str]] = {
    ("driver.py", "forward_pass"),
    ("driver.py", "get_saver"),
    ("trainer.py", "_setup_family"),
    ("DRV", "test_forward_pass_refuses_loudly_in_pr0"),
    ("DRV", "test_get_saver_refuses_loudly_in_pr0"),
}

_FAMILY_DIR = Path(__file__).resolve().parents[1] / "models" / "families" / "minimax_h3"
_REFUSAL_CONTROL = Path(__file__).resolve().parent / "fixtures" / "h3_refusal_control.py"


def _refusal_sites(path: Path, label: str) -> set[tuple[str, str]]:
    """Every function in ``path`` that RAISES a ``NotImplementedError`` (directly
    or through the ``_lands_in_pr1`` helper) — code only; docstrings and
    comments are never scanned (the row 1.2 lesson)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            src = ast.unparse(node.exc)
            if "NotImplementedError" in src or "_lands_in_pr1" in src:
                found.add((label, fn.name))
    return found


def _refusal_tests(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        ("DRV", fn.name)
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef) and "refuses_loudly_in_pr0" in fn.name
    }


def test_no_unlisted_pr0_refusal_remains():
    found = (
        _refusal_sites(_FAMILY_DIR / "driver.py", "driver.py")
        | _refusal_sites(_FAMILY_DIR / "trainer.py", "trainer.py")
        | _refusal_tests(Path(__file__))
    )
    unlisted = sorted(found - REMAINING_PR0_REFUSALS)
    gone = sorted(REMAINING_PR0_REFUSALS - found)
    assert not unlisted, "unlisted refusal " + ", ".join(f"{f}:{n}" for f, n in unlisted)
    assert not gone, "retire it by name: " + ", ".join(f"{f}:{n}" for f, n in gone)


def test_refusal_scanner_flags_the_positive_control():
    """The scanner must SEE a refusal: the control fixture raises one directly,
    one through the helper shape, and one inside a method; prose is ignored."""
    found = _refusal_sites(_REFUSAL_CONTROL, "h3_refusal_control.py")
    assert found == {
        ("h3_refusal_control.py", "direct_refusal"),
        ("h3_refusal_control.py", "helper_refusal"),
        ("h3_refusal_control.py", "still_refuses"),
    }, found
