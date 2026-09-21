"""H3 has ONE clock and the definition states it (LANE-92 VERIFY round 1,
finding 1.02; controller ruling).

The defect: a 30-fps source with ``target_fps`` unset was ingested at 30 fps
(``pipeline_data._resolve_clip_base_fps`` prefers the clip's own rate), its
soundtrack was cut to ``frames / 30`` seconds, and the driver then fitted the
audio rows — and ``packing.py`` placed the video frames — on the definition's
24-fps clock: 90 frames -> 3.0 s of audio = 120 latents, zero-padded to the
150 the 24-fps clock expects, the valid mask kept.

Seams under test are REAL: ``prepare_data`` (the dataset API faked at the HTTP
client, as the short-clip suite does), the trainer's ``_pre_cache_aux`` on a
real WAV through ``load_audio_waveform``, the audio cache on disk and
``build_batch_extra``'s fitting. Only the audio VAE is a stub — a time-local
one (one latent per 800-sample hop), so a click keeps its position.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL, StubVisualVAE
from app.engine.tests.test_minimax_h3_trainer import _StubAudioVAE

H3 = "minimax-h3-t2va"
WAN = "wan2.1-t2v-1.3b"  # keeps the source fps today: the byte-identical control
SOURCE_FPS = 30.0
SOURCE_SECONDS = 3.0
EVENT_S = 2.0  # a click in the soundtrack AND the frame shown at that instant
SR = 32000


class _PaddingAudioVAE(_StubAudioVAE):
    """The real audio VAE pads a clip UP to whole 800-sample latents (5 frames
    -> 6667 samples -> 9 latents, the driver suite's GATE-0 note); the shared
    stub only takes whole hops. Same time-local projection, padded first."""

    def encode(self, sample: torch.Tensor):
        short = (-sample.shape[-1]) % self.hop
        return super().encode(torch.nn.functional.pad(sample, (0, short)))


class _BarePipeline(GenericTrainingPipeline):
    def _setup_family(self) -> None:  # pragma: no cover - never called by the shell
        pass


def _definition(def_id: str):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def _write_click_wav(path) -> None:
    """3 s of stereo silence with a 10 ms burst at ``EVENT_S``."""
    n = int(SOURCE_SECONDS * SR)
    start = int(EVENT_S * SR)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for i in range(n):
            v = int(30000 * math.sin(i * 0.3)) if start <= i < start + SR // 100 else 0
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))


def _fake_api(monkeypatch, tmp_path, fps: float = SOURCE_FPS) -> None:
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    _write_click_wav(ds / "clip.wav")
    pairs = [
        {
            "media_file": "clip.wav",
            "caption_content": "a click",
            "metadata": {
                "width": 64, "height": 64, "is_video": True, "enabled": True,
                "fps": fps, "duration_s": SOURCE_SECONDS,
            },
        }
    ]
    info = {"path": str(ds), "version": "1.0.0", "kind": "standard"}

    async def _get(_self, url, *args, **kwargs):
        body = pairs if url.endswith("/pairs") else info
        return SimpleNamespace(status_code=200, json=lambda: json.loads(json.dumps(body)))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


def _h3(tmp_path, monkeypatch, **config) -> MiniMaxH3Trainer:
    t = object.__new__(MiniMaxH3Trainer)
    t.device = torch.device("cpu")
    t.definition = _definition(H3)
    t.logger = MagicMock()
    t._log_writer = None
    t.text_cache = {}
    t.config = {
        "resolutions": [64],
        "datasets": [{"dataset_name": "ds"}],
        "cache_latents": True,
        "train_audio": True,
        "num_frames": 107,
        **config,
    }
    t._setup_family()
    t.components = {
        "vae": H3PixelAdaptedVAE(StubVisualVAE()),
        "audio_vae": _PaddingAudioVAE(),
        "tokenizer": StubProcessor(),
        "text_encoder": StubQwen3VL(hidden_size=8),
    }
    t._assign_components()
    _fake_api(monkeypatch, tmp_path)
    return t


def _clock(t) -> tuple[float, float]:
    arch = t.definition.architecture_params
    return float(arch["video.frame_rate"]), float(arch["audio.latent_rate"])


def test_a_30_fps_source_with_target_fps_unset_trains_on_one_clock(tmp_path, monkeypatch):
    t = _h3(tmp_path, monkeypatch, target_fps=0)
    clock_fps, latent_rate = _clock(t)
    asyncio.run(t.prepare_data())
    (item,) = t.inventory
    t._pre_cache_aux()
    cached = int(t._load_cached_audio(item).shape[-1])
    extra = t.build_batch_extra([item])
    fitted = extra["audio_clean"][0]
    assert extra["audio_mask"].tolist() == [1.0]
    # NO zero padding beyond what the duration explains: the fit may crop the
    # VAE's round-up, never invent rows the soundtrack does not have.
    assert int(fitted.shape[-1]) <= cached, (
        f"{item['target_frames']} frames ingested at {item['target_fps']:g} fps: {cached} audio "
        f"latents padded with zeros to {int(fitted.shape[-1])} on the {clock_fps:g}-fps clock"
    )
    # The event coincides: the loader samples frame k at source time
    # k / target_fps (`components/video.py`), packing places frame k at
    # k / clock_fps; an audio latent j sits at j / latent_rate.
    frame_at_event = round(EVENT_S * float(item["target_fps"]))
    video_time = frame_at_event / clock_fps
    audio_time = int(fitted[0, -1].abs().argmax()) / latent_rate
    assert abs(video_time - audio_time) <= 1.0 / clock_fps, (
        f"the event at {EVENT_S} s lands at {video_time:.3f} s on the video clock "
        f"and {audio_time:.3f} s on the audio clock"
    )
    assert float(item["target_fps"]) == clock_fps


def test_the_job_says_which_clips_it_resampled(tmp_path, monkeypatch):
    t = _h3(tmp_path, monkeypatch)  # target_fps absent altogether
    clock_fps, _ = _clock(t)
    asyncio.run(t.prepare_data())
    per_clip = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "clip_fps_resampled"]
    assert len(per_clip) == 1, "changing the user's frames silently is the defect row 2.7 exists to prevent"
    kw = per_clip[0].kwargs
    assert (kw["media"], kw["source_fps"], kw["used_fps"]) == ("clip.wav", SOURCE_FPS, clock_fps)
    summary = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert summary[-1].kwargs["resampled_clips"] == 1


def test_a_source_already_on_the_clock_is_not_reported(tmp_path, monkeypatch):
    t = _h3(tmp_path, monkeypatch)
    clock_fps, _ = _clock(t)
    _fake_api(monkeypatch, tmp_path, fps=clock_fps)
    asyncio.run(t.prepare_data())
    assert not [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "clip_fps_resampled"]
    summary = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert summary[-1].kwargs["resampled_clips"] == 0


@pytest.mark.parametrize(
    ("config", "numbers"),
    [
        ({"target_fps": 30}, ("target_fps=30", "24")),
        ({"target_fps": "23.976"}, ("target_fps=23.976", "24")),
        ({"frame_stride": 2}, ("frame_stride=2", "12", "24")),
    ],
)
def test_a_setting_that_contradicts_the_clock_is_refused_at_setup(tmp_path, monkeypatch, config, numbers):
    with pytest.raises(ValueError) as err:
        _h3(tmp_path, monkeypatch, **config)
    message = str(err.value)
    assert message.startswith("minimax_h3:")
    for number in numbers:
        assert number in message, f"{number!r} missing from: {message}"


@pytest.mark.parametrize("config", [{}, {"target_fps": 0}, {"target_fps": "0"}, {"target_fps": 24}, {"frame_stride": 1}])
def test_settings_on_the_clock_are_accepted(tmp_path, monkeypatch, config):
    assert _h3(tmp_path, monkeypatch, **config).settings is not None


def test_another_family_still_keeps_its_source_fps(tmp_path, monkeypatch):
    """Positive control for the shared ingestion code: nothing changes for a
    family that did not ask for a fixed clock — value, per-clip lines and the
    summary line's keys."""
    t = object.__new__(_BarePipeline)
    t.definition = _definition(WAN)
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True}
    t._assign_components()
    _fake_api(monkeypatch, tmp_path)
    asyncio.run(t.prepare_data())
    (item,) = t.inventory
    assert float(item["target_fps"]) == SOURCE_FPS
    assert not [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "clip_fps_resampled"]
    summary = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert set(summary[-1].kwargs) == {"total_items", "skipped_short_clips", "snapped_clips"}


# ── VERIFY 3.01: the clock is a statement of the DEFINITION ────────────────
#
# The clock used to be a trainer class flag, so no API consumer could know it
# and the SPA approved clips on the source fps that ingestion then skipped or
# shortened. One statement (`video.ingest_at_native_fps`), one resolver
# (`resolve_video_profile(...).ingest_fps`), read by the shared ingestion AND
# served by the selector route.


def _bare(def_or_id, tmp_path, monkeypatch) -> _BarePipeline:
    """The SHARED ingestion with no family trainer around it."""
    t = object.__new__(_BarePipeline)
    t.definition = _definition(def_or_id) if isinstance(def_or_id, str) else def_or_id
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True}
    t._assign_components()
    _fake_api(monkeypatch, tmp_path)
    return t


def _served_ingest_fps() -> dict:
    from app.api.caption_context_routes import list_definitions

    return {d.id: d.model_dump()["ingest_fps"] for d in asyncio.run(list_definitions())}


def test_the_shared_ingestion_reads_the_clock_from_the_definition(tmp_path, monkeypatch):
    """No trainer class is involved: the definition alone decides."""
    t = _bare(H3, tmp_path, monkeypatch)
    clock_fps, _ = _clock(t)
    asyncio.run(t.prepare_data())
    (item,) = t.inventory
    assert float(item["target_fps"]) == clock_fps != SOURCE_FPS
    summary = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert summary[-1].kwargs["resampled_clips"] == 1


def test_no_class_flag_states_the_clock():
    from app.engine.core.pipeline.pipeline_data import PipelineDataMixin

    for cls in (MiniMaxH3Trainer, PipelineDataMixin, GenericTrainingPipeline):
        assert not hasattr(cls, "_ingest_video_at_native_fps"), f"{cls.__name__} states the clock a second time"


def test_the_api_serves_the_clock_the_trainer_ingests_on(tmp_path, monkeypatch):
    t = _h3(tmp_path, monkeypatch)
    asyncio.run(t.prepare_data())
    (item,) = t.inventory
    served = _served_ingest_fps()
    assert served[H3] == 24.0 == float(item["target_fps"])
    assert served[WAN] is None, "a family whose clips keep their own fps states no ingestion clock"
    # The verdict case the SPA must reproduce from that number: 90 source
    # frames at 30 fps are 72 on the clock, and `17n+5` trains 56 of them.
    assert int(SOURCE_SECONDS * SOURCE_FPS) == 90
    assert int(SOURCE_SECONDS * served[H3]) == 72
    assert int(item["target_frames"]) == 56


def test_without_the_statement_the_clip_keeps_its_fps_and_the_family_refuses(tmp_path, monkeypatch):
    """The negative: take the statement away from the SAME definition."""
    stated = _definition(H3)
    arch = {k: v for k, v in stated.architecture_params.items() if k != "video.ingest_at_native_fps"}
    assert len(arch) == len(stated.architecture_params) - 1, "the definition does not state its ingestion clock"
    unstated = stated.model_copy(update={"architecture_params": arch})

    t = _bare(unstated, tmp_path, monkeypatch)
    asyncio.run(t.prepare_data())
    (item,) = t.inventory
    assert float(item["target_fps"]) == SOURCE_FPS

    h3 = object.__new__(MiniMaxH3Trainer)
    h3.device = torch.device("cpu")
    h3.definition = unstated
    h3.logger = MagicMock()
    h3._log_writer = None
    h3.text_cache = {}
    h3.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True, "num_frames": 107}
    with pytest.raises(ValueError, match="video.ingest_at_native_fps"):
        h3._setup_family()


def test_a_statement_without_an_fps_is_an_invalid_definition():
    from app.engine.core.video_contract import resolve_video_profile, validate_video_config

    stated = _definition(H3)
    arch = {k: v for k, v in stated.architecture_params.items() if k not in ("video.frame_rate", "video.native_fps")}
    clockless = stated.model_copy(update={"architecture_params": arch})
    assert resolve_video_profile(clockless).ingest_fps is None
    report = validate_video_config(clockless, {})
    assert any("video.ingest_at_native_fps" in e for e in report.errors), report.errors
    assert validate_video_config(stated, {}).ok


def test_only_the_definitions_that_state_a_clock_have_one():
    from app.engine.core.video_contract import resolve_video_profile

    _definition(H3)
    with_clock = {
        def_id: resolve_video_profile(d).ingest_fps
        for def_id, d in ModelRegistry._definitions.items()
        if resolve_video_profile(d).ingest_fps is not None
    }
    assert with_clock == {"minimax-h3-t2va": 24.0, "minimax-h3-fl2va": 24.0, "minimax-h3-ref2va": 24.0}
