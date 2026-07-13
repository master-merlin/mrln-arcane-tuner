"""ACE-Step 1.5 composite (caption, lyrics) TE-cache unit tests.

Exercises ``AceStep15Trainer._compose_key`` and ``encode_text``'s reassembly
(padding-to-batch-max, "padding=longest" semantics) against a pre-seeded
``text_cache`` — no real driver/model, mirrors the style of
``tests/engine/families/test_trainer_seam_contract.py``.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace

from app.engine.models.families.ace_step15.trainer import AceStep15Trainer


def _make_trainer(**config_overrides) -> AceStep15Trainer:
    t = object.__new__(AceStep15Trainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True, "duration_s": 30.0, **config_overrides}
    t.text_cache = {}
    t.driver = SimpleNamespace(text_encoder=None)
    return t


# ── _compose_key ──────────────────────────────────────────────────────────


def test_compose_key_distinguishes_caption_and_lyrics():
    k1 = AceStep15Trainer._compose_key("a song", "verse one")
    k2 = AceStep15Trainer._compose_key("a song", "verse two")
    k3 = AceStep15Trainer._compose_key("another song", "verse one")
    assert len({k1, k2, k3}) == 3


def test_compose_key_empty_lyrics_default():
    assert AceStep15Trainer._compose_key("cap", "") == "cap␞"


# ── encode_text reassembly ────────────────────────────────────────────────


def test_encode_text_single_item_from_cache():
    t = _make_trainer()
    key = AceStep15Trainer._compose_key("a song", "la la")
    eh = torch.randn(1, 5, 8)
    em = torch.ones(1, 5, dtype=torch.bool)
    t.text_cache[key] = (eh, em)

    out_eh, out_em = t.encode_text(["a song"], torch.float32, batch={"lyrics": ["la la"]})
    assert torch.allclose(out_eh, eh)
    assert torch.equal(out_em, em)


def test_encode_text_pads_variable_length_batch():
    t = _make_trainer()
    k1 = AceStep15Trainer._compose_key("short", "")
    k2 = AceStep15Trainer._compose_key("long", "")
    t.text_cache[k1] = (torch.randn(1, 3, 4), torch.ones(1, 3, dtype=torch.bool))
    t.text_cache[k2] = (torch.randn(1, 7, 4), torch.ones(1, 7, dtype=torch.bool))

    eh, em = t.encode_text(["short", "long"], torch.float32, batch={"lyrics": ["", ""]})
    assert eh.shape == (2, 7, 4)  # padded to the batch max length
    assert em.shape == (2, 7)
    # Short item's padded tail is zero in both embedding and mask.
    assert torch.all(eh[0, 3:] == 0.0)
    assert torch.all(em[0, 3:] == False)  # noqa: E712
    assert torch.all(em[0, :3])


def test_encode_text_default_lyrics_when_batch_missing():
    """No ``batch=`` kwarg (or no 'lyrics' key) -> lyrics default to "" per
    item, matching the composite key an empty-lyrics pre-cache would use."""
    t = _make_trainer()
    key = AceStep15Trainer._compose_key("a song", "")
    eh = torch.randn(1, 4, 4)
    em = torch.ones(1, 4, dtype=torch.bool)
    t.text_cache[key] = (eh, em)

    out_eh, _ = t.encode_text(["a song"], torch.float32)
    assert torch.allclose(out_eh, eh)


def test_encode_text_cache_miss_without_driver_raises():
    t = _make_trainer()  # driver.text_encoder is None (offloaded) and cache empty
    import pytest

    with pytest.raises(RuntimeError, match="not pre-cached"):
        t.encode_text(["never cached"], torch.float32, batch={"lyrics": [""]})


def test_encode_text_live_when_caching_disabled():
    t = _make_trainer(cache_text_embeddings=False)
    called = {}

    def _fake_encode_condition(prompts, lyrics, dtype, audio_duration=None):
        called["prompts"] = prompts
        called["lyrics"] = lyrics
        return torch.randn(len(prompts), 3, 4), torch.ones(len(prompts), 3, dtype=torch.bool)

    t.driver = SimpleNamespace(encode_condition=_fake_encode_condition, text_encoder=object())
    eh, em = t.encode_text(["a"], torch.float32, batch={"lyrics": ["l"]})
    assert called["prompts"] == ["a"]
    assert called["lyrics"] == ["l"]
    assert eh.shape == (1, 3, 4)
