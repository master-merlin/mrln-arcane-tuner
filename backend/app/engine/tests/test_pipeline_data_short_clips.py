"""Short clips are refused at INGESTION (plan row 2.7; LANE-92 SPEC round 10,
MAJOR 10.01).

Production caller: ``pipeline_data.py:prepare_data`` itself. A driver-side
refusal would run AFTER ingestion has hidden the frame count: ``bucketing.py``
``frame_bucket_for`` gives a clip shorter than every bucket the SMALLEST
bucket, ``video.py:load_clip`` then pads an untrimmed clip by repeating frames
(3 available -> 5 requested -> source indices ``[0, 1, 2, 2, 2]``) and raises
``VideoClipTooShort`` for a trimmed one, which ``pipeline_caching.py`` turns
into a run-aborting ``RuntimeError``. So ``prepare_data`` owns ONE additive
check before ``get_bucket_for_video``: a clip with fewer frames than the
definition's frame-rule floor (5 for ``17n+5``; 1 for ``4n+1`` / ``8n+1``, so
no shipped family changes behaviour) is SKIPPED, logged per item, and counted
as ``skipped_short_clips`` on the ``data_prepared`` line (UAT item 6).

These four tests drive the REAL ``prepare_data`` (the dataset API faked at the
HTTP client, the seam the cache-integration suite already uses).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubVisualVAE

FPS = 24  # 3 / 24 = 0.125 s is exact in binary: int(0.125 * 24) == 3, no rounding luck
H3 = "minimax-h3-t2va"
WAN = "wan2.1-t2v-1.3b"  # a `4n+1` family: floor 1, the byte-identical control


class _BarePipeline(GenericTrainingPipeline):
    """The base pipeline made concrete: `prepare_data` is the base's, untouched."""

    def _setup_family(self) -> None:  # pragma: no cover - never called by the shell
        pass


def _definition(def_id: str):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def _pair(name: str, frames: int, *, trim_frames: int | None = None) -> dict:
    """A /pairs row for a clip of ``frames`` frames at ``FPS``; with
    ``trim_frames`` the clip is longer but its usable window is that short."""
    duration = frames / FPS if trim_frames is None else 2.0
    meta = {
        "width": 64,
        "height": 64,
        "is_video": True,
        "enabled": True,
        "fps": FPS,
        "duration_s": duration,
    }
    if trim_frames is not None:
        meta["trim_start_s"] = 0.0
        meta["trim_end_s"] = trim_frames / FPS
    return {"media_file": name, "caption_content": f"caption {name}", "metadata": meta}


def _fake_api(monkeypatch, tmp_path, pairs: list[dict]) -> None:
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    info = {"path": str(ds), "version": "1.0.0", "kind": "standard"}

    async def _get(_self, url, *args, **kwargs):
        body = pairs if url.endswith("/pairs") else info
        return SimpleNamespace(status_code=200, json=lambda: json.loads(json.dumps(body)))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


def _pipeline(def_id: str, monkeypatch, tmp_path, pairs: list[dict]):
    if def_id == H3:
        t = object.__new__(MiniMaxH3Trainer)
        t.definition = _definition(def_id)
        t.driver = MiniMaxH3Driver(t.definition, torch.device("cpu"))
        t.components = {"vae": H3PixelAdaptedVAE(StubVisualVAE())}
    else:
        t = object.__new__(_BarePipeline)
        t.definition = _definition(def_id)
        # `prepare_data` never consults the driver; only the shell's
        # `_assign_components` hand-off needs a receiver.
        t.driver = SimpleNamespace(assign_components=lambda components: None)
        t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {
        "resolutions": [64],
        "datasets": [{"dataset_name": "ds"}],
        "cache_latents": True,
    }
    t._assign_components()
    _fake_api(monkeypatch, tmp_path, pairs)
    return t


def _prepare(t) -> dict[str, dict]:
    asyncio.run(t.prepare_data())
    return {item["id"]: item for item in t.inventory}


def _skip_lines(t) -> list:
    return [c for c in t.logger.warning.call_args_list if c.args and c.args[0] == "short_clip_skipped"]


def _skipped_count(t) -> int | None:
    lines = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert lines, "no data_prepared summary line"
    return lines[-1].kwargs.get("skipped_short_clips")


def test_1_untrimmed_three_frame_clip_is_skipped_under_h3(monkeypatch, tmp_path):
    """3 frames < the `17n+5` floor of 5: the clip never becomes an inventory
    item, so the loader can never pad it to `[0, 1, 2, 2, 2]`."""
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("short.mp4", 3), _pair("ok.mp4", 5)])
    items = _prepare(t)
    assert "short.mp4" not in items, "the 3-frame clip was ingested (it would be padded to 5)"
    assert set(items) == {"ok.mp4"}
    lines = _skip_lines(t)
    assert len(lines) == 1, f"expected one short_clip_skipped line, got {len(lines)}"
    msg = lines[0].kwargs.get("message", "")
    assert msg.startswith("clip has 3 frames; the smallest legal"), msg
    assert msg.endswith("5 (17n+5)"), msg
    assert _skipped_count(t) == 1


def test_2_three_frame_trim_window_is_skipped_not_aborted(monkeypatch, tmp_path):
    """A long clip whose trim window holds 3 frames: skipped at ingestion, so
    the trimmed-clip `VideoClipTooShort` -> `RuntimeError` abort never runs."""
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("trimmed.mp4", 48, trim_frames=3), _pair("ok.mp4", 5)])
    items = _prepare(t)  # completes: no RuntimeError
    assert "trimmed.mp4" not in items, "the 3-frame trim window was ingested"
    assert _skipped_count(t) == 1


def test_3_five_frame_clip_is_ingested_under_h3(monkeypatch, tmp_path):
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("ok.mp4", 5)])
    items = _prepare(t)
    assert items["ok.mp4"]["target_frames"] == 5
    assert _skip_lines(t) == []
    assert _skipped_count(t) == 0


def test_4_three_frame_clip_under_a_4n_plus_1_family_is_ingested_as_today(monkeypatch, tmp_path):
    """The floor of `4n+1` is 1: the other families' ingestion is untouched —
    the 3-frame clip lands in the smallest bucket exactly as before this row."""
    t = _pipeline(WAN, monkeypatch, tmp_path, [_pair("short.mp4", 3)])
    items = _prepare(t)
    assert set(items) == {"short.mp4"}
    assert items["short.mp4"]["target_frames"] == 1
    assert _skip_lines(t) == []
    assert _skipped_count(t) == 0
