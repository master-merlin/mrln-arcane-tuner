"""ONE timeline for a clip's two streams: the presentation timestamps.

LANE-92 VERIFY 5.01: the training-side audio decode concatenated the decoded
samples and sliced them as if the stream started at zero, while the video
loader selects frames by presentation time — so a file whose soundtrack starts
late trained every window against the wrong second of audio, silently. Every
fixture was zero-origin, which is why nothing saw it.

Each test builds a REAL file on disk (H.264 + AAC through PyAV, the muxer the
venv already has), decodes it through the REAL loaders and asserts WHICH
samples land in which window (tone vs silence), never only a shape. The five
cases are the rows of `.agent/output/h3-gates/timeline-sweep.md`.
"""

from __future__ import annotations

import pytest
import torch
from structlog.testing import capture_logs

from app.engine.components import audio_io
from app.engine.components.audio_io import load_audio_waveform
from app.engine.components.video import VideoFrameLoader
from app.engine.tests import h3_real_seams as seams

SR = seams.SR
WINDOW_S = 22 / 24  # one legal 22-frame window at 24 fps
SILENT, TONE = 1e-3, 0.1  # the tone's RMS is ~0.35


def _rms(wav: torch.Tensor, from_s: float, to_s: float) -> float:
    return float(wav[:, int(from_s * SR) : int(to_s * SR)].pow(2).mean().sqrt())


def _onset(wav: torch.Tensor) -> int:
    return int((wav[0].abs() > 0.1).nonzero()[0])


def _legacy_decode(path, *, trim_start_s, duration_s, target_sr):
    """The decode as it was before 5.01, kept here as the reference the
    zero-origin control is compared against: concatenate, slice from zero."""
    import av
    import numpy as np

    container = av.open(str(path))
    try:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=int(target_sr))
        parts = []
        for frame in container.decode(stream):
            parts += [r.to_ndarray() for r in resampler.resample(frame)]
        parts += [r.to_ndarray() for r in resampler.resample(None)]
        wav = np.concatenate([p for p in parts if p.size], axis=1).astype("float32")
    finally:
        container.close()
    n = int(round(duration_s * target_sr))
    start = int(round(trim_start_s * target_sr))
    out = torch.zeros(2, n)
    seg = wav[:, start : start + n]
    out[:, : seg.shape[1]] = torch.from_numpy(seg)
    return out.clamp_(-1.0, 1.0)


# ── positive control: a zero-origin file does not move by one sample ─────────


@pytest.mark.parametrize("target_sr", [SR, 16000])
@pytest.mark.parametrize("trim_start_s", [0.0, 0.5])
def test_a_zero_origin_file_decodes_to_the_same_samples_as_before(tmp_path, target_sr, trim_start_s):
    path = tmp_path / "zero.mp4"
    seams.write_clip(path, 48, soundtrack=True, sound_from_s=0.25)
    wav, sr = load_audio_waveform(str(path), trim_start_s=trim_start_s, duration_s=WINDOW_S, target_sr=target_sr)
    legacy = _legacy_decode(path, trim_start_s=trim_start_s, duration_s=WINDOW_S, target_sr=target_sr)
    assert sr == target_sr and wav.shape == legacy.shape
    assert float(wav.abs().max()) > 0.1, "the control must carry a signal to compare"
    assert torch.equal(wav, legacy)


# ── (i) the soundtrack starts late ───────────────────────────────────────────


@pytest.mark.parametrize("ext", ["mp4", "mkv"])
def test_i_a_late_soundtrack_is_silence_before_it_starts_and_tone_after(tmp_path, ext):
    """Video at 0, one second of tone presented from 1.0 s."""
    path = tmp_path / f"late.{ext}"
    seams.write_clip(path, 48, soundtrack=True, audio_offset_s=1.0, audio_seconds=1.0)
    first, _ = load_audio_waveform(str(path), trim_start_s=0.0, duration_s=WINDOW_S, target_sr=SR)
    second, _ = load_audio_waveform(str(path), trim_start_s=1.0, duration_s=WINDOW_S, target_sr=SR)
    got = {"window@0": _rms(first, 0.0, 0.9), "window@1": _rms(second, 0.1, 0.9)}
    assert got["window@0"] < SILENT < TONE < got["window@1"], f"audio RMS per window {got}: the soundtrack was slid to zero"


def test_i_the_other_family_loader_reads_the_same_timeline(tmp_path):
    """The joint audio+video family that shipped first carried its own copy of
    the zero-origin decode; its entry point now lands on the same clock."""
    from app.engine.models.families.ltx2.audio_io import load_audio_waveform as family_loader

    late, zero = tmp_path / "late.mp4", tmp_path / "zero.mp4"
    seams.write_clip(late, 48, soundtrack=True, audio_offset_s=1.0, audio_seconds=1.0)
    seams.write_clip(zero, 48, soundtrack=True, sound_from_s=0.25)
    first, _ = family_loader(str(late), trim_start_s=0.0, duration_s=WINDOW_S, target_sr=16000)
    second, _ = family_loader(str(late), trim_start_s=1.0, duration_s=WINDOW_S, target_sr=16000)
    rms = [float(w[:, 1600:14400].pow(2).mean().sqrt()) for w in (first, second)]
    assert rms[0] < SILENT < TONE < rms[1], rms
    control, _ = family_loader(str(zero), trim_start_s=0.5, duration_s=WINDOW_S, target_sr=16000)
    assert torch.equal(control, _legacy_decode(zero, trim_start_s=0.5, duration_s=WINDOW_S, target_sr=16000))


# ── (ii) the soundtrack starts early (encoder priming before t=0) ────────────


def test_ii_samples_presented_before_zero_are_dropped_not_slid_into_the_window(tmp_path):
    """Matroska keeps the AAC priming frame at a negative timestamp; the tone
    written at 0.5 s must be heard at 0.5 s, not one priming frame later."""
    path = tmp_path / "early.mkv"
    seams.write_clip(path, 48, soundtrack=True, sound_from_s=0.5)
    wav, _ = load_audio_waveform(str(path), trim_start_s=0.0, duration_s=2.0, target_sr=SR)
    onset = _onset(wav)
    assert abs(onset - SR // 2) <= 64, f"tone onset at sample {onset}, written at {SR // 2}"


# ── (iii) the VIDEO starts late, audio at 0 ──────────────────────────────────


def test_iii_a_window_before_the_video_starts_is_reported_by_the_real_loader(tmp_path):
    """There is no picture before the stream starts, so the loader holds its
    first frame — that must be said, naming the file and the offset. The window
    that starts with the stream gets distinct frames and says nothing."""
    path = tmp_path / "vlate.mp4"
    seams.write_clip(path, 48, soundtrack=True, video_offset_s=1.0)
    with capture_logs() as logs:
        held = VideoFrameLoader().load_clip(str(path), 22, 24.0, 0.0, None, 64, 64)
    distinct = len({float(held[:, i].sum()) for i in range(held.shape[1])})
    assert distinct == 1, "observed 2026-09-21: every stamp before the stream resolves to its first frame"
    events = [e for e in logs if e["event"] == "video_window_before_stream_start"]
    assert len(events) == 1 and events[0]["log_level"] == "warning", logs
    assert events[0]["path"] == str(path) and events[0]["first_frame_s"] == pytest.approx(1.0)
    assert events[0]["held_frames"] == 22

    with capture_logs() as logs:
        clip = VideoFrameLoader().load_clip(str(path), 22, 24.0, 1.0, None, 64, 64)
    assert len({float(clip[:, i].sum()) for i in range(clip.shape[1])}) == 22
    assert not [e for e in logs if e["event"] == "video_window_before_stream_start"]
    # The soundtrack of that file is at zero and stays there.
    wav, _ = load_audio_waveform(str(path), trim_start_s=1.0, duration_s=WINDOW_S, target_sr=SR)
    assert _rms(wav, 0.0, 0.9) > TONE


def test_iii_a_zero_origin_clip_never_warns(tmp_path):
    path = tmp_path / "zero.mp4"
    seams.write_clip(path, 48, soundtrack=True)
    with capture_logs() as logs:
        VideoFrameLoader().load_clip(str(path), 22, 24.0, 0.0, None, 64, 64)
        VideoFrameLoader().load_clip(str(path), 22, 24.0, 1.0, None, 64, 64)
        load_audio_waveform(str(path), trim_start_s=0.0, duration_s=WINDOW_S, target_sr=SR)
    assert not [e for e in logs if e["log_level"] == "warning"], logs


# ── (iv) the soundtrack is shorter than the window ───────────────────────────


def test_iv_a_short_soundtrack_is_tone_then_silence_to_the_exact_length(tmp_path):
    path = tmp_path / "short.mp4"
    seams.write_clip(path, 48, soundtrack=True, audio_seconds=0.5)
    wav, _ = load_audio_waveform(str(path), trim_start_s=0.0, duration_s=WINDOW_S, target_sr=SR)
    assert wav.shape == (2, int(round(WINDOW_S * SR)))
    assert _rms(wav, 0.0, 0.4) > TONE and _rms(wav, 0.6, 0.9) == 0.0


# ── (v) a hole in the soundtrack's timestamps ────────────────────────────────


def test_v_a_gap_inside_the_soundtrack_stays_a_gap_and_is_reported(tmp_path):
    """Tone throughout, but half a second of timestamps is missing at 0.5 s:
    that half second is silence on the timeline, and the tone resumes at 1.0 s."""
    path = tmp_path / "gap.mp4"
    seams.write_clip(path, 48, soundtrack=True, audio_gap=(0.5, 0.5))
    with capture_logs() as logs:
        wav, _ = load_audio_waveform(str(path), trim_start_s=0.0, duration_s=1.5, target_sr=SR)
    got = {"before": _rms(wav, 0.0, 0.4), "gap": _rms(wav, 0.6, 0.95), "after": _rms(wav, 1.1, 1.4)}
    assert got["gap"] < SILENT < TONE < min(got["before"], got["after"]), f"audio RMS {got}: the gap was closed up"
    events = [e for e in logs if e["event"] == "audio_timeline_discontinuity"]
    assert len(events) == 1 and events[0]["log_level"] == "warning", logs
    assert events[0]["path"] == str(path) and events[0]["count"] == 1
    assert events[0]["first_at_s"] == pytest.approx(0.5, abs=0.05)
    assert events[0]["first_jump_s"] == pytest.approx(0.5, abs=0.01)


# ── the caches of this decode are keyed on it ────────────────────────────────


def _h3_trainer():
    return seams.shell(seams.base_config())


def _ltx2_trainer():
    from types import SimpleNamespace

    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    t = object.__new__(Ltx2Trainer)
    t.driver = SimpleNamespace(audio_sampling_rate=16000, audio_vae=None)
    return t


# The directory names a cache written BEFORE 5.01 sits under (same stub inputs:
# no audio VAE loaded). A soundtrack aligned the old way may be in there, so
# the key must never resolve to them again.
_PRE_FIX_KEYS = {"h3": "vb3fa65a4a5", "ltx2": "v2e7defa3e3"}  # computed on 709ced93


@pytest.mark.parametrize("family, build", [("h3", _h3_trainer), ("ltx2", _ltx2_trainer)])
def test_every_cache_of_decoded_audio_is_keyed_on_the_decode_version(monkeypatch, family, build):
    t = build()
    now = t._audio_cache_version()
    assert audio_io.AUDIO_DECODE_VERSION >= 2
    assert now != _PRE_FIX_KEYS[family], "a cache written by the zero-origin decode would be served again"
    monkeypatch.setattr(audio_io, "AUDIO_DECODE_VERSION", audio_io.AUDIO_DECODE_VERSION + 1)
    assert t._audio_cache_version() != now, "the key does not move with the decode"


def test_the_gap_tolerance_is_below_one_video_frame():
    """A container that stores timestamps in milliseconds jitters by under a
    millisecond per frame; anything a viewer could see as a slip is a gap."""
    assert 0.001 < audio_io.TIMELINE_GAP_TOLERANCE_S < 1 / 60
