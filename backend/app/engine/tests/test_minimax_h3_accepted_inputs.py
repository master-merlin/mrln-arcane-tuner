"""What an H3 job ACCEPTS it must train correctly; the rest it refuses at setup.

LANE-92 VERIFY round 2. Every test here goes through ``h3_real_seams`` — real
files, the real ``prepare_data`` / pre-cache / batch / forward / loss path, a
tiny REAL visual VAE and transformer. Nothing is handed a fabricated latent.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.tests import h3_real_seams as seams


def _dataset(tmp_path, monkeypatch, *, clips=((24, True),), stills: int = 0) -> list[dict]:
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    pairs = [
        seams.write_clip(ds / f"clip{i}_{frames}f.mp4", frames, soundtrack=sound)
        for i, (frames, sound) in enumerate(clips)
    ]
    pairs += [seams.write_still(ds / f"still{i}.png") for i in range(stills)]
    seams.fake_api(monkeypatch, ds, pairs)
    return pairs


def _job(tmp_path, monkeypatch, build_tiny_transformer, *, clips=((24, True),), stills: int = 0, **config):
    _dataset(tmp_path, monkeypatch, clips=clips, stills=stills)
    t = seams.shell(seams.base_config(**config))
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)
    return t


# ── The positive control every refusal below is measured against ────────────


def test_the_default_configuration_trains_video_and_its_soundtrack(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer)
    (item,) = t.inventory
    loss, pred, _target, batch = seams.train_step(t, [item])
    assert batch["audio_mask"].tolist() == [1.0]
    assert batch["audio_clean"].any(), "the clip's soundtrack must reach the step"
    assert float(t.last_step_losses.loss_audio) > 0.0
    assert torch.isfinite(loss) and loss.requires_grad
    assert seams.events(t, "minimax_h3_data_summary")[-1]["clips_without_audio"] == 0


# ── 2.02: a still image in an H3 dataset ───────────────────────────────────
#
# Still-image training is NOT proven on the real H3 VAE in this lane: a still
# is skipped where short clips are skipped. With the skip removed this test
# fails inside the real `_pre_cache_latents` -> `LatentManager` ->
# `H3PixelAdaptedVAE` -> `AutoencoderKLMiniMaxH3._encode` with
# "Tensors must have same number of dimensions: got 4 and 5".


def test_a_still_is_skipped_loudly_and_never_reaches_the_encoder(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, stills=2)
    assert [i["is_video"] for i in t.inventory] == [True], "only the clip is trainable"
    skipped = seams.events(t, "still_image_skipped", level="warning")
    assert sorted(e["media"] for e in skipped) == ["still0.png", "still1.png"]
    for e in skipped:
        assert "17n+5" in e["message"] and "minimax" not in e["message"].lower()
    assert seams.events(t, "data_prepared")[-1]["skipped_stills"] == 2
    assert seams.events(t, "pre_caching_latents_done")[-1]["encoded"] == 1
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)


def test_a_dataset_of_only_stills_fails_loudly(tmp_path, monkeypatch, build_tiny_transformer):
    with pytest.raises(ValueError) as err:
        _job(tmp_path, monkeypatch, build_tiny_transformer, clips=(), stills=3)
    message = str(err.value)
    assert message.startswith("No training data found in datasets.")
    assert "3 still image(s)" in message and "17n+5" in message and "minimax" not in message.lower()


def test_another_video_family_still_takes_stills(tmp_path, monkeypatch):
    """Positive control for the shared ingestion code: a frame rule whose floor
    is one frame (`4n+1`) keeps its stills, and its summary line has no new key."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.engine.tests.test_minimax_h3_clock import WAN, _BarePipeline

    ds = tmp_path / "ds"
    ds.mkdir()
    seams.fake_api(monkeypatch, ds, [seams.write_still(ds / "still.png")])
    t = object.__new__(_BarePipeline)
    t.definition = seams.definition(WAN)
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True}
    t._assign_components()
    asyncio.run(t.prepare_data())
    assert [i["is_video"] for i in t.inventory] == [False]
    assert not seams.events(t, "still_image_skipped", level="warning")
    assert set(seams.events(t, "data_prepared")[-1]) == {"total_items", "skipped_short_clips", "snapped_clips"}


# ── 2.01: cache_latents=false ───────────────────────────────────────────────


def test_cache_latents_off_is_refused_at_setup_before_anything_loads():
    with pytest.raises(ValueError, match=r"^minimax_h3: cache_latents=False is not supported") as err:
        seams.shell(seams.base_config(cache_latents=False))
    assert "soundtrack" in str(err.value) and "Set cache_latents to true" in str(err.value)


def test_a_clip_with_a_soundtrack_is_supervised_or_the_run_is_refused(tmp_path, monkeypatch, build_tiny_transformer):
    """``cache_latents=false`` with ``train_audio=true``: either the step gets
    real audio supervision or setup refuses. Silently neither is the defect."""
    import app.engine.components.audio_io as audio_io

    calls: list[str] = []
    real = audio_io.load_audio_waveform
    monkeypatch.setattr(audio_io, "load_audio_waveform", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    try:
        t = _job(tmp_path, monkeypatch, build_tiny_transformer, cache_latents=False, train_audio=True)
    except ValueError as refusal:
        message = str(refusal)
        assert message.startswith("minimax_h3: cache_latents=False"), message
        assert "soundtrack" in message and "cache_latents" in message
        assert calls == []
        return
    extra = t.build_batch_extra(list(t.inventory))
    assert extra["audio_mask"].tolist() == [1.0], (
        f"train_audio=true, a clip WITH a soundtrack, {len(calls)} audio-loader calls, "
        f"audio_mask={extra['audio_mask'].tolist()}: the soundtrack is silently not trained"
    )
