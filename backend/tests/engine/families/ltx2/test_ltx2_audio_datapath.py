"""LTX-2 audio data path: pre-cache + batch collation (no GPU, no weights).

These two hooks are what make ``batch["audio_clean"]`` exist — without them the
driver's joint forward always falls into the video-only branch and the audio
LoRA receives zero gradient (the "trains but no audio" bug). Tests use a fake
driver/latent-manager and monkeypatch the PyAV decode so they run on CPU with no
model and no media files.

Covered:
- ``_pre_cache_aux`` encodes + caches one audio latent per video clip WITH
  audio, skips stills and audio-less clips, is idempotent (re-run skips cached),
  and offloads the audio VAE afterwards.
- ``build_batch_extra`` stacks cached latents into ``audio_clean`` + a presence
  ``audio_mask``; absent items get a zero latent + mask 0; an all-absent batch
  (or audio-off run) yields ``{}`` so the forward stays video-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog
import torch
from safetensors.torch import load_file, save_file

import app.engine.models.families.ltx2.audio_io as audio_io
from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.trainer import Ltx2Trainer

L = 10  # audio latent length used throughout


class _FakeAudioVae:
    def __init__(self) -> None:
        self.offloaded_to = None

    def to(self, dev):
        self.offloaded_to = dev
        return self


class _FakeDriver:
    def __init__(self, *, train_audio=True, with_vae=True) -> None:
        self.train_audio = train_audio
        self.audio_vae = _FakeAudioVae() if with_vae else None
        self.audio_sampling_rate = 16000
        self.frame_rate = 24.0
        self.encode_calls = 0

    def encode_audio_clean(self, waveform, sample_rate):
        self.encode_calls += 1
        return torch.ones(1, L, 128)  # [B, L, 128]


class _FakeLM:
    """Deterministic, content-free cache filenames keyed on the item id."""

    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}.safetensors"


def _trainer(*, train_audio=True, with_vae=True, cache_latents=True):
    t = object.__new__(Ltx2Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_latents": cache_latents}
    t.driver = _FakeDriver(train_audio=train_audio, with_vae=with_vae)
    t.latent_manager = _FakeLM()
    t.inventory = []
    return t


def _video_item(tmp_path, ident, *, has_audio=True):
    return {
        "id": ident,
        "path": f"/clips/{ident}{'' if has_audio else '_noaudio'}.mkv",
        "cache_dir": str(tmp_path / ident),
        "is_video": True,
        "target_frames": 25,
        "target_fps": 24.0,
        "trim_start_s": 0.0,
        "trim_end_s": 1.0,
        "_has_audio": has_audio,
    }


def _patch_decode(monkeypatch):
    """Fake load_audio_waveform: a clip "has audio" unless its path says noaudio."""

    def fake(path, *, trim_start_s, duration_s, target_sr):
        if "noaudio" in path:
            return None
        return torch.zeros(1, int(duration_s * target_sr)), target_sr

    monkeypatch.setattr(audio_io, "load_audio_waveform", fake)


# ── _pre_cache_aux ────────────────────────────────────────────────────────


def test_precache_writes_one_latent_per_audio_clip(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [
        _video_item(tmp_path, "clipA"),
        _video_item(tmp_path, "clipB"),
        _video_item(tmp_path, "clipC", has_audio=False),  # no audio stream
        {"id": "img1", "path": "/img/a.png", "cache_dir": str(tmp_path / "img1"),
         "is_video": False},  # still → skipped
    ]

    t._pre_cache_aux()

    from pathlib import Path

    a = Path(t._audio_cache_dir(str(tmp_path / "clipA"))) / "clipA.safetensors"
    b = Path(t._audio_cache_dir(str(tmp_path / "clipB"))) / "clipB.safetensors"
    c = Path(t._audio_cache_dir(str(tmp_path / "clipC"))) / "clipC.safetensors"
    assert a.exists() and b.exists()
    assert not c.exists()  # audio-less clip not cached
    assert not (tmp_path / "img1" / "audio").exists()  # still skipped
    assert load_file(str(a))["audio_latents"].shape == (L, 128)
    assert t.driver.encode_calls == 2


def test_precache_offloads_audio_vae(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()

    assert t.driver.audio_vae.offloaded_to == "cpu"


def test_precache_is_idempotent(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()
    assert t.driver.encode_calls == 1
    t._pre_cache_aux()  # cache file now exists → skip
    assert t.driver.encode_calls == 1


def test_precache_noop_when_audio_off(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer(train_audio=False)
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()

    assert t.driver.encode_calls == 0
    assert not (tmp_path / "clipA" / "audio").exists()


# ── build_batch_extra ──────────────────────────────────────────────────────


def _seed_audio_cache(t, item, tensor):
    adir = t._audio_cache_dir(item["cache_dir"])
    import os

    os.makedirs(adir, exist_ok=True)
    fname = t.latent_manager.latent_filename(item["id"], item["path"])
    save_file({"audio_latents": tensor}, str(__import__("pathlib").Path(adir) / fname))


def test_build_batch_extra_stacks_and_masks(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    b = _video_item(tmp_path, "clipB")  # left uncached → absent
    c = {"id": "img1", "path": "/img/a.png", "cache_dir": str(tmp_path / "img1"),
         "is_video": False}
    _seed_audio_cache(t, a, torch.full((L, 128), 3.0))

    extra = t.build_batch_extra([a, b, c])

    assert set(extra) == {"audio_clean", "audio_mask"}
    assert extra["audio_clean"].shape == (3, L, 128)
    assert torch.equal(extra["audio_mask"], torch.tensor([1.0, 0.0, 0.0]))
    # Present item keeps its values; absent items are zeros.
    assert torch.equal(extra["audio_clean"][0], torch.full((L, 128), 3.0))
    assert torch.count_nonzero(extra["audio_clean"][1]) == 0
    assert torch.count_nonzero(extra["audio_clean"][2]) == 0


def test_build_batch_extra_empty_when_no_audio_in_batch(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")  # nothing cached
    assert t.build_batch_extra([a]) == {}


def test_build_batch_extra_empty_when_audio_off(tmp_path):
    t = _trainer(train_audio=False)
    a = _video_item(tmp_path, "clipA")
    _seed_audio_cache(t, a, torch.ones(L, 128))
    assert t.build_batch_extra([a]) == {}


def test_build_batch_extra_treats_shape_mismatch_as_absent(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    b = _video_item(tmp_path, "clipB")
    _seed_audio_cache(t, a, torch.ones(L, 128))
    _seed_audio_cache(t, b, torch.ones(L + 3, 128))  # wrong length → treated absent

    extra = t.build_batch_extra([a, b])

    assert extra["audio_clean"].shape == (2, L, 128)
    assert torch.equal(extra["audio_mask"], torch.tensor([1.0, 0.0]))


# ── encode_audio_clean device co-location ──────────────────────────────────


class _RecVae:
    """Audio VAE fake that records device moves and returns a fixed latent."""

    def __init__(self) -> None:
        self.moved_to: list = []
        self.dtype = torch.float32
        self.latents_mean = None
        self.latents_std = None

    def to(self, dev):
        self.moved_to.append(dev)
        return self

    def encode(self, mel):  # extract_audio_latents handles a raw-tensor return
        b = mel.shape[0]
        return torch.zeros(b, 8, L, 16)  # → pack_audio_latents → [B, L, 128]


class _RecLogger:
    def __init__(self):
        self.infos: list = []
        self.warnings: list = []

    def info(self, event, **kw):
        self.infos.append((event, kw))

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def debug(self, *a, **k):
        pass


def test_precache_partial_audio_failure_warns_and_continues(tmp_path, monkeypatch):
    """A per-clip audio encode failure must not kill the run when OTHER clips
    encode — counted, left uncached (audio_mask=0 downstream), and visible."""
    _patch_decode(monkeypatch)
    t = _trainer()
    rec = _RecLogger()
    t.logger = rec

    calls = {"n": 0}

    def _sometimes_boom(waveform, sample_rate):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first clip audio encode blew up")
        return torch.ones(1, L, 128)

    t.driver.encode_audio_clean = _sometimes_boom
    # clipBad encoded first (fails), clipGood second (succeeds) → partial failure.
    t.inventory = [
        _video_item(tmp_path, "clipBad"),
        _video_item(tmp_path, "clipGood"),
    ]

    t._pre_cache_aux()  # must not raise — partial failure degrades gracefully

    good = Path(t._audio_cache_dir(str(tmp_path / "clipGood"))) / "clipGood.safetensors"
    bad = Path(t._audio_cache_dir(str(tmp_path / "clipBad"))) / "clipBad.safetensors"
    assert good.exists() and not bad.exists()
    done = [kw for ev, kw in rec.infos if ev == "ltx2_audio_precache_done"]
    assert done and done[0]["failed"] == 1 and done[0]["encoded"] == 1
    warn_events = [ev for ev, _ in rec.warnings]
    assert "ltx2_audio_precache_incomplete" in warn_events


def test_precache_raises_when_all_audio_clips_fail(tmp_path, monkeypatch):
    """TOTAL audio encode failure must ESCALATE — audio-on training with zero
    audio latents is a misconfigured run, not a silent degrade."""
    _patch_decode(monkeypatch)
    t = _trainer()
    rec = _RecLogger()
    t.logger = rec

    def _boom(waveform, sample_rate):
        raise RuntimeError("audio vae encode blew up")

    t.driver.encode_audio_clean = _boom
    t.inventory = [_video_item(tmp_path, "clipA")]

    with pytest.raises(RuntimeError, match="ltx2_audio_precache_incomplete"):
        t._pre_cache_aux()

    adir = t._audio_cache_dir(str(tmp_path / "clipA"))
    assert not (Path(adir) / "clipA.safetensors").exists()
    warn_events = [ev for ev, _ in rec.warnings]
    assert "ltx2_audio_precache_incomplete" in warn_events


def test_precache_resume_with_cached_clips_does_not_escalate(tmp_path, monkeypatch):
    """Resume nuance: when prior clips are already cached (skipped>0) and only
    the NEW clip(s) fail, the run still carries real audio latents for the
    cached majority — partial-degrade (warn + audio_mask=0), NOT total failure.
    Escalating here would hard-abort a healthy resume."""
    _patch_decode(monkeypatch)
    t1 = _trainer()
    t1.driver.encode_audio_clean = lambda waveform, sample_rate: torch.ones(1, L, 128)
    t1.inventory = [_video_item(tmp_path, "clipCached")]
    t1._pre_cache_aux()  # clipCached lands on disk

    t2 = _trainer()
    rec = _RecLogger()
    t2.logger = rec

    def _boom(waveform, sample_rate):
        raise RuntimeError("audio vae encode blew up")

    t2.driver.encode_audio_clean = _boom
    t2.inventory = [
        _video_item(tmp_path, "clipCached"),
        _video_item(tmp_path, "clipNewBad"),
    ]

    t2._pre_cache_aux()  # must NOT raise: skipped=1, failed=1, encoded=0

    done = [kw for ev, kw in rec.infos if ev == "ltx2_audio_precache_done"]
    assert done and done[0]["skipped"] == 1 and done[0]["failed"] == 1
    warn_events = [ev for ev, _ in rec.warnings]
    assert "ltx2_audio_precache_incomplete" in warn_events


# ── audio-latent cache versioning (stale-cache guard) ──────────────────────


def test_audio_cache_dir_carries_a_version_segment(tmp_path):
    import os as _os

    t = _trainer()
    adir = t._audio_cache_dir(str(tmp_path / "cache"))
    parts = _os.path.normpath(adir).split(_os.sep)
    assert "audio" in parts
    i = parts.index("audio")
    assert i + 1 < len(parts) and parts[i + 1].startswith("v")


def test_audio_cache_version_stable_for_same_params(tmp_path):
    t = _trainer()
    assert t._audio_cache_dir(str(tmp_path / "cache")) == t._audio_cache_dir(
        str(tmp_path / "cache")
    )


def test_audio_cache_version_changes_with_sampling_rate(tmp_path):
    t = _trainer()
    v1 = t._audio_cache_dir(str(tmp_path / "cache"))
    t.driver.audio_sampling_rate = 22050
    v2 = t._audio_cache_dir(str(tmp_path / "cache"))
    assert v1 != v2


def test_audio_cache_version_changes_with_vae_stats(tmp_path):
    t = _trainer()
    t.driver.audio_vae.latents_mean = torch.zeros(128)
    t.driver.audio_vae.latents_std = torch.ones(128)
    v1 = t._audio_cache_dir(str(tmp_path / "cache"))
    t.driver.audio_vae.latents_mean = torch.ones(128)  # different VAE identity
    v2 = t._audio_cache_dir(str(tmp_path / "cache"))
    assert v1 != v2


def test_encode_audio_clean_colocates_vae_with_input():
    """Regression: the audio VAE must be moved to the mel's device before encode.

    The generic orchestration only relocates the VIDEO VAE, so without an
    explicit move the CPU-resident audio VAE meets a CUDA mel → "Input type
    (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be the same".
    Asserted device-agnostically (CPU in CI) by checking the VAE was moved to
    ``driver.device``.
    """
    drv = object.__new__(Ltx2Driver)
    drv.device = torch.device("cpu")
    drv.audio_sampling_rate = 16000
    drv._audio_mel = None
    drv.audio_vae = _RecVae()

    out = drv.encode_audio_clean(torch.zeros(1, 2, 1600), 16000)  # [B, C=2, N]

    assert drv.audio_vae.moved_to, "audio VAE was never co-located with the input"
    assert torch.device(drv.audio_vae.moved_to[-1]) == drv.device
    assert out.shape[-1] == 128  # packed audio latent feature dim


# ── corrupt cache file (poison-pill) handling ──────────────────────────────
#
# Task W2.T2 widened build_batch_extra's load-site catch from (OSError,
# KeyError) to Exception, because safetensors 0.8.0's SafetensorError
# subclasses Exception DIRECTLY (not OSError) — so a truncated/corrupt cache
# file used to crash every subsequent run at the same step (os.path.exists
# still counts the truncated file as "cached"). These tests write a REAL
# corrupt .safetensors file (not a mock) at the exact lookup path and drive
# the real load path.


def _write_corrupt_audio_cache_file(t, item) -> str:
    """A genuinely corrupt (truncated) .safetensors file at the exact path
    ``build_batch_extra`` looks up — matches the real crash mode: a bad
    header that raises ``safetensors.SafetensorError`` (NOT OSError/KeyError)."""
    import os

    adir = t._audio_cache_dir(item["cache_dir"])
    os.makedirs(adir, exist_ok=True)
    fname = t.latent_manager.latent_filename(item["id"], item["path"])
    path = str(Path(adir) / fname)
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)
    return path


def test_build_batch_extra_degrades_to_miss_on_corrupt_cache_file(tmp_path):
    """A pre-existing corrupt audio-latent cache file must degrade to a MISS,
    never raise — the exact poison-pill regression this task guards against.
    The corrupt file must also be DISCARDED (best-effort unlink) so the
    poison pill does not persist across every future run — see the self-heal
    test below."""
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    path = _write_corrupt_audio_cache_file(t, a)

    extra = t.build_batch_extra([a])  # must not raise

    assert extra == {}
    assert not Path(path).exists()  # corrupt file discarded, not left behind


def test_build_batch_extra_zero_fills_item_with_corrupt_cache_alongside_good(tmp_path):
    """Mixed batch: clipGood has a genuinely valid cached latent, clipBad's
    cache file is present-but-corrupt. clipBad must degrade exactly like
    "absent" (zero latent, mask 0), clipGood must be unaffected, and a
    visible warning must name the corrupt path — never an unhandled raise."""
    t = _trainer()
    good = _video_item(tmp_path, "clipGood")
    bad = _video_item(tmp_path, "clipBad")
    _seed_audio_cache(t, good, torch.full((L, 128), 3.0))
    _write_corrupt_audio_cache_file(t, bad)

    rec = _RecLogger()
    t.logger = rec
    extra = t.build_batch_extra([good, bad])  # must not raise

    assert extra["audio_clean"].shape == (2, L, 128)
    assert torch.equal(extra["audio_mask"], torch.tensor([1.0, 0.0]))
    assert torch.equal(extra["audio_clean"][0], torch.full((L, 128), 3.0))
    assert torch.count_nonzero(extra["audio_clean"][1]) == 0
    warn = [kw for ev, kw in rec.warnings if ev == "ltx2_audio_cache_load_failed"]
    assert warn and "clipBad" in warn[0]["path"]
    bad_path = Path(t._audio_cache_dir(bad["cache_dir"])) / "clipBad.safetensors"
    assert not bad_path.exists()  # discarded, not left as a poison pill


def test_precache_aux_regenerates_after_build_batch_extra_discards_corrupt_file(
    tmp_path, monkeypatch
):
    """CORRECTED contract (supersedes this test's original 6c87dea2 version,
    which asserted the file was left "permanently" corrupt and the clip
    "permanently degrades to audio_mask=0"). ``_pre_cache_aux``'s skip check
    is STILL a plain content-blind ``os.path.exists`` — it does not itself
    repair a corrupt file sitting untouched on disk (that deeper
    content-aware-skip fix is out of scope here, same as ``LatentManager``'s
    pattern). But that is no longer the whole story: ``build_batch_extra``'s
    load-time catch now discards the poison pill (best-effort unlink) the
    moment it is encountered, so by the time the NEXT precache pass runs
    (e.g. the following training run, or a later resume), the file is gone
    and the content-blind exists-check correctly sees it absent and
    regenerates it. The damage is bounded to the run that hit the corrupt
    file — not permanent."""
    _patch_decode(monkeypatch)
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    path = _write_corrupt_audio_cache_file(t, a)
    rec = _RecLogger()
    t.logger = rec

    # This run's load attempt hits the poison pill mid-training.
    extra = t.build_batch_extra([a])  # must not raise
    assert extra == {}  # this run's batch(es) still degrade for this clip
    assert not Path(path).exists()  # corrupt file discarded, not left behind
    warn = [kw for ev, kw in rec.warnings if ev == "ltx2_audio_cache_load_failed"]
    assert warn  # degradation for this run was logged visibly

    # Next precache pass (simulating the following run/resume) sees the file
    # absent and regenerates it with a real latent — poison pill cleared.
    t.inventory = [a]
    t._pre_cache_aux()

    assert Path(path).exists()
    assert load_file(path)["audio_latents"].shape == (L, 128)
    assert t.driver.encode_calls == 1  # regenerated exactly once


# ── corrupt cache → user-visible warning (W2 pre-merge finding) ────────────
#
# self.logger.warning goes to the structured backend logger and never reaches
# job_log.jsonl / the Jobs screen (see pipeline_train.py's nan_window_skipped
# handling). Without ALSO emitting through self._emit_warning (the JobLogWriter
# IPC seam sampling failures already use), an LTX-2 run whose audio stream
# silently drops to mask=0 completes looking perfectly healthy. These tests
# assert against a fake ``_log_writer`` — the real seam ``_emit_warning``
# writes through — not against the structured logger mock used above.


class _WarningRecorder:
    """JobLogWriter stand-in recording every warning(...) call."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def status(self, label: str) -> None:  # pragma: no cover - unused here
        pass


def test_build_batch_extra_corrupt_cache_emits_user_visible_warning(tmp_path):
    """The corrupt-cache degrade must ALSO surface via _emit_warning (→
    job_log.jsonl / Jobs screen), naming the clip and stating plainly that it
    trained with no audio and that the cache will regenerate."""
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    _write_corrupt_audio_cache_file(t, a)
    lw = _WarningRecorder()
    t._log_writer = lw

    extra = t.build_batch_extra([a])  # must not raise

    assert extra == {}
    assert len(lw.warnings) == 1
    msg = lw.warnings[0]
    assert "clipA" in msg
    assert "audio" in msg.lower()
    assert "regenerate" in msg.lower() or "next pre-cache" in msg.lower()


def test_build_batch_extra_corrupt_cache_warns_user_once_per_item(tmp_path):
    """build_batch_extra runs once per training step — repeated calls for the
    SAME corrupt clip must not flood job_log.jsonl with a duplicate warning
    per step."""
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    _write_corrupt_audio_cache_file(t, a)
    lw = _WarningRecorder()
    t._log_writer = lw

    t.build_batch_extra([a])
    # The corrupt file was discarded after the first call — recreate it so
    # the SECOND call hits the load-failure branch again (simulating the
    # cache still being absent/corrupt for this same run/step).
    _write_corrupt_audio_cache_file(t, a)
    t.build_batch_extra([a])
    t.build_batch_extra([a])

    assert len(lw.warnings) == 1


def test_build_batch_extra_without_log_writer_is_safe(tmp_path):
    """No _log_writer attached (e.g. other unit tests) → no crash."""
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    _write_corrupt_audio_cache_file(t, a)

    extra = t.build_batch_extra([a])  # must not raise

    assert extra == {}
