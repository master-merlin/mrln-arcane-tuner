"""MiniMax-H3 trainer — the text-embedding lifecycle (plan row 2.1).

Production caller (DECISION-68 (a)): ``run_trainer.py:159`` calls
``_pre_cache_text_embeddings`` (a base NO-OP) and then ``_offload_text_encoders``;
the base ``encode_text`` (``pipeline_base.py``) returns ``None`` once the driver
reports no encoders. So the three trainer overrides this file pins are what
keeps a job from training on ``None`` embeddings after the 63 GB Qwen3-VL is
released. Stub encoder, no weights; the seams under test (the base offload,
the base caption-hint builder, the disk cache) are REAL.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL


def _definition(def_id: str = "minimax-h3-t2va"):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def _trainer(tmp_path, *, sample_prompt: str = "a [triggerword] clip") -> MiniMaxH3Trainer:
    """A real trainer + real driver + stub encoder, no heavy ``__init__``
    (the ``test_boogu_image_trainer.py::_trainer_shell`` shape)."""
    t = object.__new__(MiniMaxH3Trainer)
    t.device = torch.device("cpu")
    t.definition = _definition()
    t.driver = MiniMaxH3Driver(t.definition, t.device)
    t.logger = MagicMock()
    t._log_writer = None
    t.text_cache = {}
    t._te_unloaded = False
    t.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "global_triggerword": "sks",
        "sample_prompts": [{"prompt": sample_prompt}],
        "datasets": [],
    }
    # One training clip: the base `_build_caption_hints` composes
    # "sks, a cat" (trigger + caption) and "" (dropout) from it.
    t.inventory = [{"id": "1", "path": str(tmp_path / "clip.mp4"), "caption": "a cat"}]
    t._resolve_te_cache_dirs = lambda: [str(tmp_path / "te")]
    t.components = {
        "tokenizer": StubProcessor(),
        "text_encoder": StubQwen3VL(hidden_size=8),
        "vae": object(),
        "audio_vae": object(),
    }
    t._assign_components()
    return t


def _encoder_calls(t: MiniMaxH3Trainer) -> int:
    return t.components["text_encoder"].calls


def test_te_cache_serves_embeddings_after_release(tmp_path):
    t = _trainer(tmp_path)
    te = t.components["text_encoder"]

    # ── warm while the encoder is resident ──
    t._pre_cache_text_embeddings()
    warmed = dict(t.text_cache)
    calls_after_warm = te.calls

    # ── release every encoder reference through the REAL base offload ──
    t._offload_text_encoders()
    assert t.driver.get_text_encoders() == {}
    assert "text_encoder" not in t.components
    assert t.driver.text_encoder is None
    assert t.text_cache, "cache empty after release"
    assert {"sks, a cat", "", "a sks clip"} <= set(warmed), sorted(warmed)
    assert calls_after_warm > 0

    # ── serve: a training caption, the empty prompt, a sample prompt ──
    out = t.encode_text(["sks, a cat", "", "a sks clip"], torch.float32)
    assert out is not None, "encode_text returned None with no encoders"
    assert isinstance(out, TextEncoderOutput)
    assert te.calls == calls_after_warm, "the released encoder was called"
    emb, mask = out.embeddings, out.attention_mask
    assert emb.shape[0] == 3 and mask.shape[0] == 3
    for i, cap in enumerate(["sks, a cat", "", "a sks clip"]):
        w_emb, w_mask = warmed[cap]
        n = int(w_mask.sum())
        assert int(mask[i].sum()) == n
        assert torch.equal(emb[i, :n], w_emb[:n].to(emb.dtype))
        assert torch.all(emb[i, n:] == 0)

    # ── a caption never warmed raises, naming it — never a silent reload ──
    with pytest.raises(RuntimeError, match="never-warmed caption"):
        t.encode_text(["never-warmed caption"], torch.float32)
    assert te.calls == calls_after_warm


def test_pre_cache_round_trips_through_the_disk_cache(tmp_path):
    """A second trainer on the same dataset serves from disk with ZERO
    encoder calls; the disk path carries the driver's cache scope (tap +
    tokenizer), so a different tap is a miss, never a stale hit."""
    first = _trainer(tmp_path)
    first._pre_cache_text_embeddings()
    n_first = _encoder_calls(first)
    assert n_first > 0

    second = _trainer(tmp_path)
    second._pre_cache_text_embeddings()
    assert _encoder_calls(second) == 0, "disk cache not consulted"
    assert set(second.text_cache) == set(first.text_cache)
    for cap, (emb, mask) in first.text_cache.items():
        emb2, mask2 = second.text_cache[cap]
        assert torch.equal(emb, emb2) and torch.equal(mask, mask2)

    third = _trainer(tmp_path)
    third.definition = third.definition.model_copy(
        update={
            "architecture_params": {
                **third.definition.architecture_params,
                "te.hidden_state_tap_index": 49,
            }
        }
    )
    third.driver.definition = third.definition
    third._pre_cache_text_embeddings()
    assert _encoder_calls(third) == n_first, "a different tap reused layer-50 embeddings"
    assert torch.all(third.text_cache["sks, a cat"][0] == 49.0)


def test_caching_off_is_refused_loudly(tmp_path):
    """H3 cannot keep the 63 GB encoder resident beside the DiT, so a config
    that asks for live per-step encoding is refused up front, not at step 1."""
    t = _trainer(tmp_path)
    t.config["cache_text_embeddings"] = False
    with pytest.raises(ValueError, match="cache_text_embeddings"):
        t._pre_cache_text_embeddings()


def test_transformer_is_materialised_only_after_the_release(tmp_path):
    """The loader defers the 62 GB DiT out of Phase A; the trainer materialises
    it through the loader's single-spec path ONLY once the encoder is gone
    (asserted on the driver and the component set), logging
    ``text_encoder released`` before ``transformer loaded``."""
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader

    t = _trainer(tmp_path)
    loader = MiniMaxH3Loader(t.device, defer_transformer=True)
    manifest = {s.key for s in loader.get_component_manifest(t.definition)}
    assert "transformer" not in manifest
    assert {"tokenizer", "text_encoder", "vae", "audio_vae"} == manifest
    eager = {s.key for s in MiniMaxH3Loader(t.device).get_component_manifest(t.definition)}
    assert "transformer" in eager

    dit = torch.nn.Linear(2, 2)
    seen: list = []

    def _single(spec, definition, root_path, dtype, target_device):
        seen.append((spec.key, spec.subfolder, dtype, target_device))
        return dit

    loader._load_single_spec = _single
    loader._root_path = str(tmp_path)
    t.loader = loader

    with pytest.raises(RuntimeError, match="text encoder still resident"):
        t._materialise_transformer()
    assert seen == []

    t._pre_cache_text_embeddings()
    t._offload_text_encoders()
    t._materialise_transformer()

    assert seen == [("transformer", "transformer", torch.bfloat16, "cpu")]
    assert t.components["transformer"] is dit
    assert t.driver.get_primary_model() is dit
    assert t.transformer is dit

    events = [c.args[0] for c in t.logger.info.call_args_list]
    assert "text_encoder released" in events and "transformer loaded" in events
    assert events.index("text_encoder released") < events.index("transformer loaded")
    released = next(c for c in t.logger.info.call_args_list if c.args[0] == "text_encoder released")
    assert released.kwargs["weight_bytes"] == 1024 * 4
    assert "host_peak_bytes" in released.kwargs and "cuda_peak_bytes" in released.kwargs


def test_ref2va_materialises_the_ref_checkpoint(tmp_path):
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader

    loader = MiniMaxH3Loader(torch.device("cpu"), defer_transformer=True)
    seen: list = []
    loader._load_single_spec = lambda spec, *a: seen.append(spec.subfolder)
    loader._root_path = str(tmp_path)
    loader.load_transformer(_definition("minimax-h3-ref2va"), torch.bfloat16)
    assert seen == ["transformer_ref"]
