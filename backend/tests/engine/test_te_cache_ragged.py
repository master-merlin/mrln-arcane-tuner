"""Cross-family TE-cache RAGGED-BATCH hygiene (Wave 3 item W3-4).

WHAT THIS PINS
--------------
Families whose text encoder produces a **variable-length** output per prompt
(VL / chat-template encoders that ``pad="longest"``, never a fixed crop) cache
one entry per caption and must reassemble a MIXED-LENGTH batch without either

1. crashing on a ragged ``torch.stack`` (entries of different length), or
2. mis-padding — a shorter caption's padded rows must be masked out so the
   padded batch is byte-equivalent to the text encoder's own
   ``pad="longest"`` batch (the model derives per-sample caption lengths from
   ``mask.sum(dim=1)`` / cu_seqlens, so a zero row behind ``mask=0`` is inert).

``boogu_image`` is the reference implementation (trim-at-store + mask-aware pad
at collate). This module proves the SAME invariant holds for every
variable-length family in the W3-4 ownership set, and documents the
fixed-length families that are legitimately exempt.

PROOF PATTERN
-------------
Seed the in-memory cache with two per-caption entries of DIFFERENT true length
(``L1 < L2``), each filled with a content marker equal to its own length, then
collate ``encode_text([short, long])`` and assert:

* no RuntimeError (ragged stack would raise),
* the batch pads to ``L2`` (the batch max) with the short row's tail zeroed,
* the mask marks exactly ``L1`` / ``L2`` valid positions,
* the content markers survive (no foreign padding leaked into the cache).

Only the collate/reassembly seam is exercised (cache pre-seeded), so no real
multi-billion-param text encoder is loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import structlog
import torch

_D = 8  # tiny feature dim


# ── 2-D mask-pad families (entry = (feat[L, D], mask[L])) ──────────────────
#
# qwen_image / boogu_image / ernie_image / ideogram4 all cache a 2-D per-caption
# feature + a 1-D mask and reassemble to (feat[B, Lmax, D], mask[B, Lmax]).


def _make_2d_trainer(trainer_cls, l1: int, l2: int):
    """Bare trainer with a two-entry ragged cache (all-real-token masks)."""
    t = object.__new__(trainer_cls)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.logger = structlog.get_logger("test")
    t.text_encoder = object()  # truthy; the all-cached path never touches it
    t.driver = SimpleNamespace(text_encoder=object())
    t.text_cache = {
        "short": (torch.full((l1, _D), float(l1)), torch.ones(l1, dtype=torch.long)),
        "long": (torch.full((l2, _D), float(l2)), torch.ones(l2, dtype=torch.long)),
    }
    return t


def _import_2d_trainer(family: str):
    if family == "qwen_image":
        from app.engine.models.families.qwen_image.trainer import QwenImageTrainer
        return QwenImageTrainer
    if family == "boogu_image":
        from app.engine.models.families.boogu_image.trainer import BooguImageTrainer
        return BooguImageTrainer
    if family == "ernie_image":
        from app.engine.models.families.ernie_image.trainer import ErnieImageTrainer
        return ErnieImageTrainer
    if family == "ideogram4":
        from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer
        return IdeogramV4Trainer
    raise AssertionError(family)


@pytest.mark.parametrize(
    "family", ["qwen_image", "boogu_image", "ernie_image", "ideogram4"]
)
def test_ragged_2d_collate_pads_and_masks(family):
    """Mixed-length batch collates without crashing and pads mask-equivalently."""
    l1, l2 = 3, 7
    trainer_cls = _import_2d_trainer(family)
    t = _make_2d_trainer(trainer_cls, l1, l2)

    emb, mask = t.encode_text(["short", "long"], torch.float32)

    # Padded to the batch max length, one row per caption.
    assert emb.shape == (2, l2, _D)
    assert mask.shape == (2, l2)

    # Short row: true tokens carry the content marker, tail is zero-padded.
    assert torch.all(emb[0, :l1] == float(l1))
    assert torch.all(emb[0, l1:] == 0.0)
    # Long row: fully real, no padding.
    assert torch.all(emb[1] == float(l2))

    # Mask marks exactly the true lengths (zero == ignored/padding position).
    m = mask.to(torch.bool)
    assert int(m[0].sum()) == l1
    assert int(m[1].sum()) == l2
    assert not torch.any(m[0, l1:])


# ── microsoft_lens (entry = (feat[4, S, D], mask[S])) ──────────────────────


def test_ragged_lens_collate_pads_and_masks():
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )

    l1, l2, n_layers = 3, 7, 4
    t = object.__new__(MicrosoftLensTrainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.logger = structlog.get_logger("test")
    t.text_encoder = object()
    t.driver = SimpleNamespace(text_encoder=object())
    t.text_cache = {
        "short": (
            torch.full((n_layers, l1, _D), float(l1)),
            torch.ones(l1, dtype=torch.long),
        ),
        "long": (
            torch.full((n_layers, l2, _D), float(l2)),
            torch.ones(l2, dtype=torch.long),
        ),
    }

    feats, mask = t.encode_text(["short", "long"], torch.float32)

    assert feats.shape == (2, n_layers, l2, _D)
    assert mask.shape == (2, l2)
    assert torch.all(feats[0, :, :l1] == float(l1))
    assert torch.all(feats[0, :, l1:] == 0.0)
    assert torch.all(feats[1] == float(l2))
    assert int(mask[0].sum()) == l1
    assert int(mask[1].sum()) == l2


# ── zimage (variable-length LIST output — no stack, no mask) ────────────────


def test_ragged_zimage_collate_returns_list_preserving_lengths():
    """Z-Image returns a per-sample list of ``[Li, D]`` tensors, so mixed
    lengths need no padding and can never trip a ragged stack."""
    from app.engine.models.families.zimage.trainer import ZImageTrainer

    l1, l2 = 3, 7
    t = object.__new__(ZImageTrainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.logger = structlog.get_logger("test")
    t.text_encoder = object()
    t.driver = SimpleNamespace(text_encoder=object())
    t.text_cache = {
        "short": torch.full((l1, _D), float(l1)),
        "long": torch.full((l2, _D), float(l2)),
    }

    out = t.encode_text(["short", "long"], torch.float32)

    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape == (l1, _D)
    assert out[1].shape == (l2, _D)
    assert torch.all(out[0] == float(l1))
    assert torch.all(out[1] == float(l2))


# ── kandinsky5 (triple entry with rebuilt cu_seqlens) ──────────────────────


def test_ragged_kandinsky5_collate_pads_and_rebuilds_cu_seqlens():
    from app.engine.models.families.kandinsky5.driver import build_cu_seqlens
    from app.engine.models.families.kandinsky5.trainer import Kandinsky5Trainer

    l1, l2, p = 3, 7, 4
    t = object.__new__(Kandinsky5Trainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.logger = structlog.get_logger("test")
    t.driver = SimpleNamespace(text_encoder=object())
    t.text_cache = {
        "short": (
            torch.full((1, l1, _D), float(l1)),
            torch.full((1, p), 1.0),
            build_cu_seqlens([l1]),
        ),
        "long": (
            torch.full((1, l2, _D), float(l2)),
            torch.full((1, p), 2.0),
            build_cu_seqlens([l2]),
        ),
    }

    out = t.encode_text(["short", "long"], torch.float32)

    assert out.embeddings.shape == (2, l2, _D)
    assert torch.all(out.embeddings[0, :l1] == float(l1))
    assert torch.all(out.embeddings[0, l1:] == 0.0)
    assert out.attention_mask.dtype == torch.int32
    # Rebuilt cu_seqlens = cumulative TRUE lengths: [0, l1, l1+l2].
    assert out.attention_mask.tolist() == [0, l1, l1 + l2]


# ── qwen_image pre-cache: entries stored TRIMMED (no foreign padding) ───────


def test_qwen_precache_stores_trimmed_entries():
    """The batched pre-cache path pads each sub-batch to ITS OWN max length;
    each cached entry must be trimmed back to its true length so a later
    mixed-batch collate never inherits a foreign caption's padding."""
    from app.engine.core.text_encoding import TextEncoderOutput
    from app.engine.models.families.qwen_image.trainer import QwenImageTrainer

    class _FakeVLDriver:
        """1 token per word, ``pad="longest"`` batches, value == true length."""

        def __init__(self) -> None:
            self.text_encoder = object()

        def encode_text(self, captions, dtype):
            lengths = [max(len(c.split()), 1) for c in captions]
            lmax = max(lengths)
            emb = torch.zeros(len(captions), lmax, _D)
            mask = torch.zeros(len(captions), lmax, dtype=torch.long)
            for i, ln in enumerate(lengths):
                emb[i, :ln] = float(ln)
                mask[i, :ln] = 1
            return TextEncoderOutput(embeddings=emb, attention_mask=mask)

    t = object.__new__(QwenImageTrainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.logger = structlog.get_logger("test")
    t.text_encoder = object()
    t.driver = _FakeVLDriver()
    t.text_cache = {}
    t._log_writer = None
    # Two captions of different length encoded in ONE sub-batch of 4 → the
    # short one is padded to the long one's length inside the batch.
    t._build_caption_hints = lambda: {"a b": "h", "a b c d e": "h"}
    t._resolve_loading_dtype = lambda: torch.float32
    t._resolve_te_cache_dirs = lambda: []

    t._pre_cache_text_embeddings()

    short_emb, short_mask = t.text_cache["a b"]
    long_emb, long_mask = t.text_cache["a b c d e"]
    # Trimmed to true length — NOT the sub-batch max of 5.
    assert short_emb.shape[0] == 2
    assert short_mask.shape[0] == 2
    assert long_emb.shape[0] == 5
    assert torch.all(short_emb == 2.0)

    # And the trimmed entries collate cleanly into a padded, masked batch.
    emb, mask = t.encode_text(["a b", "a b c d e"], torch.float32)
    assert emb.shape == (2, 5, _D)
    assert int(mask[0].sum()) == 2
    assert int(mask[1].sum()) == 5


# ── Documented fixed-length EXEMPTION: krea2 ───────────────────────────────


def test_krea2_is_fixed_length_and_exempt():
    """krea2 tokenizes with ``padding="max_length"`` (a fixed budget), so every
    per-caption entry has the SAME length and a plain ``torch.stack`` reassembly
    is safe. It is intentionally NOT on the ragged-collate pattern — this pins
    that the exemption premise (fixed length) still holds."""
    import inspect

    from app.engine.models.families.krea2.vendor import krea2_conditioning

    src = inspect.getsource(krea2_conditioning.get_text_hidden_states)
    assert 'padding="max_length"' in src
