"""Kandinsky 5.0 trainer — dual-TE cache triple + i2v hooks (no GPU).

The trainer caches ``(te1=Qwen emb, te2=CLIP pooled, te3=cu_seqlens)`` per
caption with te1 written LAST (the LTX-2 commit-marker precedent), trims
per-caption entries to their TRUE length, and reassembles batches with
``padding="longest"`` semantics + rebuilt int32 cu_seqlens. Also pins the i2v
per-step gate and frame-0 loss exclusion.
"""

from __future__ import annotations

import os

import pytest
import structlog
import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.kandinsky5.driver import (
    KANDINSKY5_DEFAULT_NEGATIVE_PROMPT,
    Kandinsky5Driver,
    build_cu_seqlens,
)
from app.engine.models.families.kandinsky5.trainer import Kandinsky5Trainer

_D = 16  # qwen emb dim (tiny)
_P = 8  # pooled dim (tiny)


class _FakeDualTEDriver:
    """Driver stand-in: 1 token per word, padding='longest' batches."""

    def __init__(self) -> None:
        self.text_encoder: object | None = object()  # resident
        self.calls = 0
        self._i2v_active = False

    def encode_text(self, captions, dtype):
        self.calls += 1
        lengths = [max(len(c.split()), 1) for c in captions]
        max_len = max(lengths)
        emb = torch.zeros(len(captions), max_len, _D)
        for i, ln in enumerate(lengths):
            emb[i, :ln] = float(ln)  # content marker: value == true length
        return TextEncoderOutput(
            embeddings=emb,
            attention_mask=build_cu_seqlens(lengths),
            pooled=torch.randn(len(captions), _P),
        )


def _trainer() -> Kandinsky5Trainer:
    t = object.__new__(Kandinsky5Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.text_cache = {}
    t.text_encoder = object()
    t.driver = _FakeDualTEDriver()
    t._log_writer = None
    t._build_caption_hints = lambda: {"a cat runs": "h", "a dog": "h", "": "d"}
    t._resolve_loading_dtype = lambda: torch.float32
    t._resolve_te_cache_dirs = lambda: []
    return t


# ── Warm + serve-after-offload ─────────────────────────────────────────────


def test_warm_populates_triple_cache():
    t = _trainer()
    t._pre_cache_text_embeddings()
    assert set(t.text_cache) == {"a cat runs", "a dog", ""}
    emb, pooled, cu = t.text_cache["a cat runs"]
    assert emb.shape == (1, 3, _D)  # trimmed to TRUE length
    assert pooled.shape == (1, _P)
    assert cu.dtype == torch.int32 and cu.tolist() == [0, 3]


def test_per_caption_entries_are_trimmed_not_padded():
    """Batch-encoding pads to the longest caption; each cached entry must be
    trimmed back to its OWN true length (no foreign padding in the cache)."""
    t = _trainer()
    t._pre_cache_text_embeddings()
    emb_dog, _, cu_dog = t.text_cache["a dog"]
    assert emb_dog.shape == (1, 2, _D)  # NOT the batch max of 3
    assert cu_dog.tolist() == [0, 2]
    assert torch.all(emb_dog == 2.0)  # content marker intact after trim


def test_encode_text_works_after_te_offloaded():
    t = _trainer()
    t._pre_cache_text_embeddings()
    calls_after_warm = t.driver.calls
    t.driver.text_encoder = None
    t.text_encoder = None

    out = t.encode_text(["a cat runs", "a dog"], torch.float32)

    assert isinstance(out, TextEncoderOutput)
    assert out.embeddings.shape == (2, 3, _D)  # padded to batch max
    assert out.pooled.shape == (2, _P)
    assert out.attention_mask.dtype == torch.int32
    assert out.attention_mask.tolist() == [0, 3, 5]  # rebuilt cu_seqlens
    # 'a dog' row: 2 true tokens + 1 zero pad column.
    assert torch.all(out.embeddings[1, :2] == 2.0)
    assert torch.all(out.embeddings[1, 2:] == 0.0)
    assert t.driver.calls == calls_after_warm  # served from cache


def test_uncached_after_offload_raises():
    t = _trainer()
    t.driver.text_encoder = None
    t.text_encoder = None
    with pytest.raises(RuntimeError, match="not pre-cached"):
        t.encode_text(["never seen"], torch.float32)


def test_warm_includes_sample_prompts_and_default_negative():
    """K5 previews run true CFG; the pipeline injects a DEFAULT negative when
    none is configured — the exact string must be pre-warmed."""
    t = _trainer()
    t.config = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a [triggerword] driving"}],
        "global_triggerword": "sks",
    }
    t._build_caption_hints = lambda: {}
    t._pre_cache_text_embeddings()
    assert "a sks driving" in t.text_cache
    assert KANDINSKY5_DEFAULT_NEGATIVE_PROMPT in t.text_cache


def test_warm_honors_configured_negative_prompt():
    t = _trainer()
    t.config = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a cat"}],
        "sample_negative_prompt": "blurry, low quality",
    }
    t._build_caption_hints = lambda: {}
    t._pre_cache_text_embeddings()
    assert "blurry, low quality" in t.text_cache
    assert KANDINSKY5_DEFAULT_NEGATIVE_PROMPT not in t.text_cache


# ── Disk triple (te1 commit marker) ────────────────────────────────────────


def _disk_trainer(tmp: str) -> Kandinsky5Trainer:
    t = _trainer()
    t._resolve_te_cache_dirs = lambda: [tmp]
    return t


def _slot(tmp: str, slot: str) -> str:
    return os.path.join(tmp, "embeddings", "none", slot)


def test_cold_run_writes_te1_te2_te3(tmp_path):
    t = _disk_trainer(str(tmp_path))
    t._pre_cache_text_embeddings()
    for slot in ("te1", "te2", "te3"):
        files = [
            f for f in os.listdir(_slot(str(tmp_path), slot))
            if f.endswith(".safetensors")
        ]
        assert len(files) == 3, f"{slot}: expected 3 caption files"


def test_warm_run_loads_triple_from_disk_without_encoding(tmp_path):
    cold = _disk_trainer(str(tmp_path))
    cold._pre_cache_text_embeddings()

    warm = _disk_trainer(str(tmp_path))
    warm._pre_cache_text_embeddings()

    assert warm.driver.calls == 0
    emb, pooled, cu = warm.text_cache["a cat runs"]
    assert emb.shape == (1, 3, _D)
    assert pooled is not None and cu is not None
    assert cu.to(torch.int64).tolist() == [0, 3]


def test_partial_triple_is_a_miss_and_reencodes(tmp_path):
    """te1 present but te2/te3 missing (crash mid-write) → clean re-encode,
    never a poisoned (emb, None, None) hit."""
    cold = _disk_trainer(str(tmp_path))
    cold._pre_cache_text_embeddings()

    # Simulate a pre-commit-marker crash: wipe te2 + te3.
    import shutil

    shutil.rmtree(_slot(str(tmp_path), "te2"))
    shutil.rmtree(_slot(str(tmp_path), "te3"))
    os.makedirs(_slot(str(tmp_path), "te2"))
    os.makedirs(_slot(str(tmp_path), "te3"))

    warm = _disk_trainer(str(tmp_path))
    warm._pre_cache_text_embeddings()
    assert warm.driver.calls > 0  # re-encoded, not served partially
    _, pooled, cu = warm.text_cache["a dog"]
    assert pooled is not None and cu is not None


# ── I2V gate + frame-0 loss exclusion ──────────────────────────────────────


def _i2v_trainer(prob: float = 1.0) -> Kandinsky5Trainer:
    from unittest.mock import MagicMock

    t = object.__new__(Kandinsky5Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {
        "video_mode": "i2v",
        "first_frame_conditioning_probability": prob,
    }
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test"
    d.lora_targetable_modules = []
    d.architecture_params = {"mode": "i2v"}
    t.driver = Kandinsky5Driver(d, torch.device("cpu"))
    return t


def test_attach_conditioning_engages_and_stashes_for_i2v():
    t = _i2v_trainer(prob=1.0)
    batch: dict = {}
    latents = torch.randn(1, 4, 3, 8, 8)
    t._attach_conditioning(batch, latents)
    assert t.driver._i2v_active is True
    assert Kandinsky5Driver.BATCH_FIRST_FRAME_LATENT in batch


def test_attach_conditioning_never_engages_for_t2v():
    t = _i2v_trainer(prob=1.0)
    t.config["video_mode"] = "t2v"
    batch: dict = {}
    t._attach_conditioning(batch, torch.randn(1, 4, 3, 8, 8))
    assert t.driver._i2v_active is False
    assert Kandinsky5Driver.BATCH_FIRST_FRAME_LATENT not in batch


def test_attach_conditioning_zero_probability_disengages():
    t = _i2v_trainer(prob=0.0)
    batch: dict = {}
    t._attach_conditioning(batch, torch.randn(1, 4, 3, 8, 8))
    assert t.driver._i2v_active is False


def test_i2v_loss_excludes_frame0():
    """Frame 0 is the clean conditioning frame — a huge error there must NOT
    move the loss when i2v is engaged."""
    t = _i2v_trainer(prob=1.0)
    t.driver._i2v_active = True
    t.driver._latent_shape = (3, 8, 8)  # engaged (F > 1)
    t.compute_loss_weight = lambda ts: None

    pred = torch.zeros(1, 3, 8, 8, 4)
    target = torch.zeros(1, 3, 8, 8, 4)
    target[:, 0] = 1000.0  # poison ONLY the conditioning frame

    loss = t._compute_step_loss(pred, target, torch.tensor([500.0]), {}, 1)
    assert loss.item() == pytest.approx(0.0)


def test_t2v_loss_keeps_all_frames():
    t = _i2v_trainer(prob=1.0)
    t.driver._i2v_active = False
    t.driver._latent_shape = (3, 8, 8)
    t.compute_loss_weight = lambda ts: None

    pred = torch.zeros(1, 3, 8, 8, 4)
    target = torch.zeros(1, 3, 8, 8, 4)
    target[:, 0] = 3.0

    loss = t._compute_step_loss(pred, target, torch.tensor([500.0]), {}, 1)
    assert loss.item() > 0.0


# ── Override trio wiring ───────────────────────────────────────────────────


def test_setup_family_wires_loader_driver_saver():
    from app.engine.models.families.kandinsky5.loader import Kandinsky5Loader
    from app.engine.models.families.kandinsky5.saver import Kandinsky5Saver
    from unittest.mock import MagicMock

    t = object.__new__(Kandinsky5Trainer)
    t.device = torch.device("cpu")
    d = MagicMock()
    d.architecture_params = {"mode": "t2v"}
    d.lora_targetable_modules = []
    t.definition = d
    t._setup_family()
    assert isinstance(t.driver, Kandinsky5Driver)
    assert isinstance(t.loader, Kandinsky5Loader)
    assert isinstance(t.saver, Kandinsky5Saver)


def test_update_primary_model_syncs_driver():
    t = _i2v_trainer()
    t.components = {}
    new = torch.nn.Linear(2, 2)
    t._update_primary_model(new)
    assert t.driver.transformer is new
    assert t.components["unet"] is new
    assert t.transformer is new
