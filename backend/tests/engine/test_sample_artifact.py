"""Tests for the image/video sample-artifact persistence path in the base
``GenericSamplingPipeline``.

Covers:
- A ``PIL.Image.Image`` → ``.png`` with the UNCHANGED filename pattern
  (``sample_{i:02d}_step{step:06d}.png`` / ``sample_{i:02d}_final.png``).
- A ``SampleArtifact`` → ``.mp4`` written via the real
  ``VideoFrameLoader.encode_video`` on tiny 4-frame tensors, then PyAV-probed
  back (frame count + fps), with audio muxed when provided.
- ``_broadcast_sample`` setting ``media_type`` to ``"image"`` vs ``"video"``.

A minimal concrete subclass of ``GenericSamplingPipeline`` is used so the base
persistence/broadcast logic runs on CPU without any model weights.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import av
import pytest
import torch
from PIL import Image

from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact


class _MinimalSampler(GenericSamplingPipeline):
    """Concrete subclass that stubs the abstract family hooks (unused here —
    we exercise ``_persist_artifact`` / ``_broadcast_sample`` directly)."""

    def encode_prompt(self, prompt):  # pragma: no cover - not exercised
        return None

    def denoise(self, noise, prompt_embedding, num_steps, guidance_scale, seed):
        return noise  # pragma: no cover - not exercised

    def decode_latents(self, latents):  # pragma: no cover - not exercised
        return Image.new("RGB", (8, 8))

    def _create_initial_noise(self, width, height, generator):
        return torch.zeros(1)  # pragma: no cover - not exercised


def _make_sampler():
    pipeline = SimpleNamespace(config={}, device=torch.device("cpu"))
    return _MinimalSampler(pipeline)


# ── Image → PNG (filename pattern preserved) ──────────────────────────────


def test_image_artifact_persists_as_png_step_pattern(tmp_path):
    s = _make_sampler()
    img = Image.new("RGB", (16, 16), color=(120, 30, 200))
    path = s._persist_artifact(img, tmp_path, index=3, displayed_step=150, final=False)
    assert path == tmp_path / "sample_03_step000150.png"
    assert path.exists()
    with Image.open(path) as reopened:
        assert reopened.size == (16, 16)


def test_image_artifact_persists_as_png_final_pattern(tmp_path):
    s = _make_sampler()
    img = Image.new("RGB", (8, 8))
    path = s._persist_artifact(img, tmp_path, index=0, displayed_step=999, final=True)
    assert path == tmp_path / "sample_00_final.png"
    assert path.exists()


# ── SampleArtifact → MP4 ──────────────────────────────────────────────────


def _probe_mp4(path):
    """Return (video_frame_count, fps, n_audio_streams)."""
    container = av.open(str(path))
    try:
        vstream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(vstream))
        fps = float(vstream.average_rate)
        n_audio = len(container.streams.audio)
    finally:
        container.close()
    return frames, fps, n_audio


def test_video_artifact_persists_as_mp4_step_pattern(tmp_path):
    s = _make_sampler()
    # Canonical [C, F, H, W] float in [-1, 1], 4 frames.
    frames = torch.rand(3, 4, 32, 32) * 2 - 1
    artifact = SampleArtifact(frames=frames, audio=None, fps=8.0)
    path = s._persist_artifact(
        artifact, tmp_path, index=2, displayed_step=200, final=False
    )
    assert path == tmp_path / "sample_02_step000200.mp4"
    assert path.exists()

    n_frames, fps, n_audio = _probe_mp4(path)
    assert n_frames == 4
    assert math.isclose(fps, 8.0, rel_tol=0.05)
    assert n_audio == 0


def test_video_artifact_final_pattern(tmp_path):
    s = _make_sampler()
    frames = torch.rand(3, 4, 32, 32) * 2 - 1
    artifact = SampleArtifact(frames=frames, fps=16.0)
    path = s._persist_artifact(
        artifact, tmp_path, index=1, displayed_step=0, final=True
    )
    assert path == tmp_path / "sample_01_final.mp4"
    assert path.exists()


def test_video_artifact_muxes_audio_when_present(tmp_path):
    s = _make_sampler()
    frames = torch.rand(3, 4, 32, 32) * 2 - 1
    # ~0.25s of mono noise at 44100 Hz.
    wav = torch.rand(11025) * 2 - 1
    artifact = SampleArtifact(frames=frames, audio=wav, fps=16.0)
    path = s._persist_artifact(
        artifact, tmp_path, index=0, displayed_step=50, final=False
    )
    n_frames, _, n_audio = _probe_mp4(path)
    assert n_frames == 4
    assert n_audio == 1


# ── media_type in the broadcast event ─────────────────────────────────────


class _CapturingSampler(_MinimalSampler):
    """Capture the structured ``sample_generated`` event kwargs."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.events: list[dict] = []

        class _Logger:
            def __init__(self, sink):
                self._sink = sink

            def info(self, _event, **kw):
                self._sink.append(kw)

            def warning(self, *a, **k):
                pass

            def debug(self, *a, **k):
                pass

        self.logger = _Logger(self.events)


def _make_capturing():
    pipeline = SimpleNamespace(config={}, device=torch.device("cpu"))
    return _CapturingSampler(pipeline)


def test_broadcast_defaults_media_type_image(tmp_path):
    s = _make_capturing()
    s._broadcast_sample(
        tmp_path / "sample_00_step000050.png", 50, 0, {"prompt": "p"}, "p"
    )
    assert s.events[-1]["media_type"] == "image"


def test_broadcast_marks_video_media_type(tmp_path):
    s = _make_capturing()
    s._broadcast_sample(
        tmp_path / "sample_00_step000050.mp4",
        50,
        0,
        {"prompt": "p"},
        "p",
        media_type="video",
    )
    assert s.events[-1]["media_type"] == "video"


def test_persist_then_broadcast_media_type_consistency(tmp_path):
    """Mirror generate_samples: image artifact → media_type image,
    SampleArtifact → media_type video."""
    s = _make_capturing()

    img = Image.new("RGB", (8, 8))
    img_path = s._persist_artifact(img, tmp_path, 0, 50, False)
    media = "video" if isinstance(img, SampleArtifact) else "image"
    s._broadcast_sample(img_path, 50, 0, {"prompt": "p"}, "p", media_type=media)
    assert img_path.suffix == ".png"
    assert s.events[-1]["media_type"] == "image"

    vid = SampleArtifact(frames=torch.rand(3, 4, 16, 16) * 2 - 1, fps=8.0)
    vid_path = s._persist_artifact(vid, tmp_path, 1, 50, False)
    media = "video" if isinstance(vid, SampleArtifact) else "image"
    s._broadcast_sample(vid_path, 50, 1, {"prompt": "p"}, "p", media_type=media)
    assert vid_path.suffix == ".mp4"
    assert s.events[-1]["media_type"] == "video"


# ── Sanity: SampleArtifact defaults ───────────────────────────────────────


def test_sample_artifact_defaults():
    art = SampleArtifact(frames=torch.zeros(3, 2, 4, 4))
    assert art.audio is None
    assert art.fps == 16.0


def test_persist_unknown_type_raises(tmp_path):
    """A non-image, non-artifact return is a programming error — .save() on a
    plain object raises AttributeError, surfacing the bug loudly."""
    s = _make_sampler()
    with pytest.raises(AttributeError):
        s._persist_artifact(object(), tmp_path, 0, 1, False)


# ── VRAM headroom guard (pre-sample skip to avoid WDDM shared-memory spill) ──


def _gpu(free_mb, total_mb=96000):
    return SimpleNamespace(vram_free_mb=free_mb, vram_total_mb=total_mb)


def _patch_snapshot(monkeypatch, gpus):
    monkeypatch.setattr(
        "app.core.system_monitor.system_monitor.snapshot",
        lambda: SimpleNamespace(gpus=gpus),
    )


def test_headroom_ok_when_ample_free(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_snapshot(monkeypatch, [_gpu(free_mb=80000)])
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.15
    assert s._vram_headroom_ok() is True


def test_headroom_not_ok_when_low_free(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_snapshot(monkeypatch, [_gpu(free_mb=5000)])  # ~5% of 96 GB
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.15
    assert s._vram_headroom_ok() is False


def test_headroom_disabled_with_zero_fraction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_snapshot(monkeypatch, [_gpu(free_mb=1)])  # almost nothing free
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.0
    assert s._vram_headroom_ok() is True  # 0 = never skip


def test_headroom_ok_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.5
    assert s._vram_headroom_ok() is True  # can't check → don't skip


def test_headroom_ok_when_no_gpu_in_snapshot(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_snapshot(monkeypatch, [])  # NVML returned no GPUs
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.5
    assert s._vram_headroom_ok() is True


def test_headroom_ok_when_snapshot_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom():
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(
        "app.core.system_monitor.system_monitor.snapshot", _boom
    )
    s = _make_sampler()
    s.config["sampling_min_free_vram_fraction"] = 0.5
    assert s._vram_headroom_ok() is True


# ── generate_samples: low VRAM skips the whole round, training continues ──────


class _RunSampler(_MinimalSampler):
    """Counts _sample_single calls and reclaim calls for generate_samples."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.singled = 0

    def _sample_single(self, prompt_cfg, step):
        self.singled += 1
        return Image.new("RGB", (8, 8))


def _make_run_sampler(tmp_path, prompts, fraction=0.0):
    model = torch.nn.Linear(2, 2)  # real nn.Module: .eval()/.train()/.training
    pipeline = SimpleNamespace(
        config={
            "sample_prompts": prompts,
            "output_dir": str(tmp_path),
            "lora_name": "t",
            "quantization": "none",
            "sampling_min_free_vram_fraction": fraction,
        },
        device=torch.device("cpu"),
        _get_primary_model=lambda: model,
    )
    return _RunSampler(pipeline)


def test_generate_samples_skips_when_low_vram(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # The pre-check reclaim runs before the headroom guard; stub the CUDA calls
    # (no real device on the test box) and leave free VRAM genuinely low so the
    # reclaim frees nothing and the round still skips.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _patch_snapshot(monkeypatch, [_gpu(free_mb=3000)])
    s = _make_run_sampler(tmp_path, [{"prompt": "a"}, {"prompt": "b"}], fraction=0.15)
    paths = s.generate_samples(step=100)
    assert paths == []
    assert s.singled == 0  # never entered the per-prompt loop


def test_generate_samples_reclaims_before_headroom_check(tmp_path, monkeypatch):
    """Step-0 baseline must not FALSE-skip on RECLAIMABLE reserved VRAM.

    ``generate_samples`` reclaims (``empty_cache``) BEFORE the headroom guard, so
    a large model whose transient load-peak reservation is still held at the
    step-0 baseline samples once that reservation is returned to the driver.
    Regression guard for "video models produce no step-0 sample": the guard reads
    device-wide NVML *free*, which counts the reserved-but-unused pool as used, so
    a big (video) model trips a false low-VRAM skip at step 0 while later periodic
    samples (allocator already cycled) pass.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    # Free VRAM starts below the 15% margin (would skip); the reclaim returns the
    # reserved pool and lifts it above the margin.
    state = {"free": 3000}
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: state.__setitem__("free", 80000)
    )
    monkeypatch.setattr(
        "app.core.system_monitor.system_monitor.snapshot",
        lambda: SimpleNamespace(gpus=[_gpu(free_mb=state["free"])]),
    )
    s = _make_run_sampler(tmp_path, [{"prompt": "a"}], fraction=0.15)
    paths = s.generate_samples(step=-1)  # step -1 → the step-0 baseline
    assert s.singled == 1  # sampled, NOT skipped
    assert len(paths) == 1


def test_generate_samples_reclaims_between_images(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    calls = {"empty": 0}
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.__setitem__("empty", calls["empty"] + 1)
    )
    # fraction 0 → headroom guard disabled so we isolate the reclaim behaviour
    s = _make_run_sampler(tmp_path, [{"prompt": "a"}, {"prompt": "b"}], fraction=0.0)
    paths = s.generate_samples(step=100)
    assert len(paths) == 2
    assert s.singled == 2
    # One reclaim per image (2) + the existing single reclaim in the finally (1)
    assert calls["empty"] >= 3
