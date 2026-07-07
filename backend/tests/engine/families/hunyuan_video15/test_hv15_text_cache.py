"""hv15 trainer text-cache tests — 4-tuple memory cache + te1/te2/te3 disk triple.

Pins:
- ``encode_text`` reassembles the batched ``(emb, mask, emb2, mask2)`` from
  the warm cache after the dual TE is offloaded; a miss then is a hard error.
- ``_pre_cache_text_embeddings`` persists the LTX-2-style triple with te1
  written LAST (commit marker): te1 = Qwen emb, te2 = ByT5 emb, te3 = the
  packed int64 mask pair.
- A warm second run loads the whole set from disk without any encoder.
- A PARTIAL triple on disk (te1 present, te2 missing) is treated as a miss.
- No-quote captions round-trip their ZERO te2 (+ zero mask tail) through disk.
- Sample prompts + the CFG negative prompt are warmed too.
"""

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import Hv15Driver
from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer

_L1, _D1 = 10, 16  # tiny Qwen-side dims
_L2, _D2 = 6, 8    # tiny ByT5-side dims


def _entry(fill: float = 1.0, zero_glyph: bool = True):
    return (
        torch.full((1, _L1, _D1), fill),
        torch.ones(1, _L1, dtype=torch.int64),
        torch.zeros(1, _L2, _D2) if zero_glyph else torch.full((1, _L2, _D2), 7.0),
        torch.zeros(1, _L2, dtype=torch.int64)
        if zero_glyph
        else torch.ones(1, _L2, dtype=torch.int64),
    )


class _FakeDriver:
    """Driver stand-in: encode_text returns deterministic tiny 4-tuples."""

    def __init__(self):
        self.text_encoder = object()  # "resident"
        self.te2_max_length = _L2
        self.encoded: list[str] = []

    def encode_text(self, captions, dtype):
        self.encoded.extend(captions)
        b = len(captions)
        zero = [extract_has_no_quotes(c) for c in captions]
        emb = torch.stack([torch.full((_L1, _D1), float(len(c))) for c in captions])
        mask = torch.ones(b, _L1, dtype=torch.int64)
        emb2 = torch.stack(
            [torch.zeros(_L2, _D2) if z else torch.full((_L2, _D2), 7.0) for c, z in zip(captions, zero)]
        )
        mask2 = torch.stack(
            [
                torch.zeros(_L2, dtype=torch.int64)
                if z
                else torch.ones(_L2, dtype=torch.int64)
                for z in zero
            ]
        )
        return emb.to(dtype), mask, emb2.to(dtype), mask2


def extract_has_no_quotes(caption: str) -> bool:
    from app.engine.models.families.hunyuan_video15.driver import extract_glyph_text

    return extract_glyph_text(caption) is None


def _trainer(
    tmp_path=None,
    captions: dict[str, str] | None = None,
    sample_prompts: list | None = None,
    driver=None,
) -> Hv15Trainer:
    t = object.__new__(Hv15Trainer)
    t.device = torch.device("cpu")
    t.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "mixed_precision": "bf16",
        "sample_prompts": sample_prompts or [],
        "sample_negative_prompt": "",
    }
    t.text_cache = {}
    t.driver = driver if driver is not None else _FakeDriver()
    t.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    t._log_writer = None
    t._resolve_te_cache_dirs = lambda: [str(tmp_path)] if tmp_path else []
    t._build_caption_hints = lambda: dict(captions or {})
    return t


# ── encode_text assembly from the warm cache ───────────────────────────────


def test_encode_text_assembles_batch_from_cache_after_offload():
    t = _trainer()
    t.driver.text_encoder = None  # offloaded
    t.text_cache = {"a": _entry(1.0), "b": _entry(2.0)}

    emb, mask, emb2, mask2 = t.encode_text(["a", "b"], torch.float32)
    assert emb.shape == (2, _L1, _D1)
    assert torch.all(emb[0] == 1.0) and torch.all(emb[1] == 2.0)
    assert mask.shape == (2, _L1)
    assert emb2.shape == (2, _L2, _D2)
    assert mask2.shape == (2, _L2)
    assert mask2.dtype == torch.int64


def test_encode_text_miss_after_offload_is_hard_error():
    t = _trainer()
    t.driver.text_encoder = None
    with pytest.raises(RuntimeError, match="not pre-cached"):
        t.encode_text(["never seen"], torch.float32)


def test_encode_text_miss_while_resident_encodes_and_caches():
    t = _trainer()
    out = t.encode_text(["fresh caption"], torch.float32)
    assert len(out) == 4
    assert "fresh caption" in t.text_cache


# ── Disk triple: save order (te1 LAST = commit marker) ─────────────────────


def test_precache_writes_triple_with_te1_last(tmp_path, monkeypatch):
    from app.engine.components import text_embeddings as te_mod

    order: list[str] = []
    real_save = te_mod.TextEmbeddingCache.save

    def spy_save(caption, tensor, cache_dir, source_hint=""):
        order.append(cache_dir.replace("\\", "/").rsplit("/", 1)[-1])
        return real_save(caption, tensor, cache_dir, source_hint)

    monkeypatch.setattr(te_mod.TextEmbeddingCache, "save", staticmethod(spy_save))

    t = _trainer(tmp_path, captions={"a caption": "hint"})
    t._pre_cache_text_embeddings()

    # One caption → te3, te2, te1 in exactly that order (te1 = commit marker).
    assert order == ["te3", "te2", "te1"]
    assert "a caption" in t.text_cache


def test_warm_run_loads_triple_from_disk_without_encoder(tmp_path):
    cold = _trainer(tmp_path, captions={"cap one": "", 'sign "HI"': ""})
    cold._pre_cache_text_embeddings()
    assert set(cold.text_cache) == {"cap one", 'sign "HI"'}

    warm_driver = _FakeDriver()
    warm = _trainer(tmp_path, captions={"cap one": "", 'sign "HI"': ""}, driver=warm_driver)
    warm._pre_cache_text_embeddings()
    assert set(warm.text_cache) == {"cap one", 'sign "HI"'}
    assert warm_driver.encoded == []  # nothing re-encoded

    # The 4-tuple survives the disk round trip.
    for cap in ("cap one", 'sign "HI"'):
        c_emb, c_mask, c_emb2, c_mask2 = cold.text_cache[cap]
        w_emb, w_mask, w_emb2, w_mask2 = warm.text_cache[cap]
        assert torch.equal(c_emb.float(), w_emb.float())
        assert torch.equal(c_mask, w_mask)
        assert torch.equal(c_emb2.float(), w_emb2.float())
        assert torch.equal(c_mask2, w_mask2)


def test_no_quote_caption_round_trips_zero_te2(tmp_path):
    cold = _trainer(tmp_path, captions={"no quotes here": ""})
    cold._pre_cache_text_embeddings()

    warm = _trainer(tmp_path, captions={"no quotes here": ""})
    warm._pre_cache_text_embeddings()
    _, _, emb2, mask2 = warm.text_cache["no quotes here"]
    assert torch.all(emb2 == 0)
    assert torch.all(mask2 == 0)
    assert mask2.dtype == torch.int64
    # And the batched assembly serves the zero glyph stream.
    warm.driver.text_encoder = None
    out = warm.encode_text(["no quotes here"], torch.float32)
    assert torch.all(out[2] == 0) and torch.all(out[3] == 0)


def test_partial_triple_is_treated_as_miss(tmp_path):
    import os

    cold = _trainer(tmp_path, captions={"partial cap": ""})
    cold._pre_cache_text_embeddings()

    # Simulate an interrupted pre-fix run: te2 file vanished, te1 present.
    te2_dir = os.path.join(str(tmp_path), "embeddings", "none", "te2")
    for f in os.listdir(te2_dir):
        os.remove(os.path.join(te2_dir, f))

    warm_driver = _FakeDriver()
    warm = _trainer(tmp_path, captions={"partial cap": ""}, driver=warm_driver)
    warm._pre_cache_text_embeddings()
    assert warm_driver.encoded == ["partial cap"]  # re-encoded, not poisoned
    assert "partial cap" in warm.text_cache


def test_sample_prompts_and_negative_are_warmed(tmp_path):
    t = _trainer(
        tmp_path,
        captions={"train cap": ""},
        sample_prompts=[{"prompt": "a preview prompt"}],
    )
    t.config["sample_negative_prompt"] = "blurry, low quality"
    t._pre_cache_text_embeddings()
    assert "a preview prompt" in t.text_cache
    assert "blurry, low quality" in t.text_cache


# ── Mask pack/unpack ───────────────────────────────────────────────────────


def test_mask_pair_pack_unpack_roundtrip():
    t = _trainer()
    mask = torch.randint(0, 2, (1, _L1), dtype=torch.int64)
    mask2 = torch.randint(0, 2, (1, _L2), dtype=torch.int64)
    packed = t._pack_masks(mask, mask2)
    assert packed.shape == (1, _L1 + _L2)
    m1, m2 = t._unpack_masks(packed)
    assert torch.equal(m1, mask)
    assert torch.equal(m2, mask2)


def test_caching_off_bypasses_cache():
    t = _trainer()
    t.config["cache_text_embeddings"] = False
    out = t.encode_text(["direct"], torch.float32)
    assert len(out) == 4
    assert t.text_cache == {}
