"""How many whole frames a trim window holds: one count, every consumer.

LANE-92 VERIFY 4.01. ``int(duration * fps)`` truncated a product that is an
integer in everything but floating point: at 24 fps the window [24/24, 29/24]
multiplies out to 4.999999999999998, so a legal 5-frame clip was skipped under
a ``17n+5`` rule, and [48/24, 70/24] (21.999999999999996) was cut from 22
frames to 5. The UI mirrored the same expression and approved neither.

The expected counts are DATA (``frame-count.golden.json``, hand-written, read
by this module and by the frontend spec) — nothing here recomputes them with
the formula under test. The claims about ingestion and the loader go through
the real ``prepare_data`` and the real ``VideoFrameLoader`` on real mp4 files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from app.engine.components.video import VideoFrameLoader
from app.engine.core.video_contract import whole_frames
from app.engine.tests import h3_real_seams as seams
from app.engine.tests.test_minimax_h3_clock import WAN, _BarePipeline, _definition

# Anchored on this file, never the CWD: <repo>/frontend/src/testing/golden/.
GOLDEN = Path(__file__).resolve().parents[4] / "frontend" / "src" / "testing" / "golden" / "frame-count.golden.json"
CASES = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]


def test_the_golden_holds_the_cases_the_review_named():
    ids = {c["id"] for c in CASES}
    assert {"five_frames_from_1s_at_24", "22_frames_from_2s_at_24", "fraction_floors_0.98s_at_24"} <= ids
    assert len(ids) == len(CASES) >= 10


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_frame_count_golden_backend(case):
    clock = case["ingest_fps"] or case["source_fps"]
    assert whole_frames(case["end_s"] - case["start_s"], clock) == case["frames"], case["note"]


@pytest.mark.parametrize(("duration", "fps"), [(0.0, 24.0), (-1.0, 24.0), (1.0, 0.0), (1.0, -5.0), (float("nan"), 24.0)])
def test_nothing_to_count_is_zero_frames(duration, fps):
    assert whole_frames(duration, fps) == 0


# ── Through the real seams ─────────────────────────────────────────────────


def _trimmed(pair: dict, start_s: float, end_s: float) -> dict:
    pair["metadata"].update(trim_start_s=start_s, trim_end_s=end_s)
    return pair


def _load(item: dict) -> torch.Tensor:
    """The real loader on exactly what ingestion wrote into the inventory."""
    return VideoFrameLoader().load_clip(
        item["path"], int(item["target_frames"]), float(item["target_fps"]),
        float(item["trim_start_s"]), item["trim_end_s"], item["target_w"], item["target_h"],
    )


def test_prepare_data_and_the_loader_agree_on_a_frame_aligned_trim(tmp_path, monkeypatch, build_tiny_transformer):
    """The two verdict windows, a rule whose floor is above a still. Before the
    fix: ``five.mp4`` is skipped (``short_clip_skipped``: "clip has 4 frames")
    and ``twentytwo.mp4`` trains 5 frames. The loader must then SUPPLY the count
    ingestion promised, from the same window of the same file — an ingestion
    that says 22 and a loader that says "too short" aborts the run."""
    ds = tmp_path / "ds"
    ds.mkdir()
    pairs = [
        _trimmed(seams.write_clip(ds / "five.mp4", 48, soundtrack=True), 24 / 24, 29 / 24),
        _trimmed(seams.write_clip(ds / "twentytwo.mp4", 96, soundtrack=True), 48 / 24, 70 / 24),
        _trimmed(seams.write_clip(ds / "fraction.mp4", 30, soundtrack=True), 0.0, 0.98),
    ]
    seams.fake_api(monkeypatch, ds, pairs)
    # A cap above every clip here: the counts below are the clips' own, and the
    # snap line (reported only for a cut the RULE made) names the raw 23.
    t = seams.shell(seams.base_config(num_frames=39))
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)

    assert not seams.events(t, "short_clip_skipped", level="warning")
    frames = {i["id"]: int(i["target_frames"]) for i in t.inventory}
    assert frames == {"five.mp4": 5, "twentytwo.mp4": 22, "fraction.mp4": 22}
    # 0.98 s is 23.52 frames: the fraction still floors (23), the rule snaps it.
    (snap,) = seams.events(t, "clip_frames_snapped")
    assert (snap["media"], snap["source_frames"], snap["used_frames"]) == ("fraction.mp4", 23, 22)

    assert seams.events(t, "pre_caching_latents_done")[-1]["encoded"] == 3
    for item in t.inventory:
        assert _load(item).shape[1] == frames[item["id"]], item["id"]
        loss, *_ = seams.train_step(t, [item])
        assert torch.isfinite(loss)

    # D10, derived data keyed on every input: the frame count is part of the
    # latent cache's directory, so a clip whose count moved cannot read the
    # latent cached under its old count.
    dirs = {i["id"]: Path(i["cache_dir"]).name for i in t.inventory}
    assert "x5f" in dirs["five.mp4"] and "x22f" in dirs["twentytwo.mp4"], dirs
    assert dirs["five.mp4"] != dirs["twentytwo.mp4"]


def _shared_ingestion(tmp_path, monkeypatch, pairs: list[dict], **config) -> _BarePipeline:
    """The SHARED ingestion for a family with no fixed clock and a rule whose
    floor is a still."""
    t = object.__new__(_BarePipeline)
    t.definition = _definition(WAN)
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True, **config}
    t._assign_components()
    seams.fake_api(monkeypatch, tmp_path, pairs)
    asyncio.run(t.prepare_data())
    return t


def test_another_rule_gains_the_frame_truncation_cost_it_and_nothing_else_moves(tmp_path, monkeypatch):
    """Positive controls on the shared code under ``4n+1``: a window whose exact
    product is 81 (the double says 80.99999999999999) used to train 77 frames
    and now trains 81 — UP, and only there. A window with a real fraction
    (81.6 frames) and one that is an exact double (81.0) train what they did."""
    pairs = [
        _trimmed(seams.write_clip(tmp_path / "gained.mp4", 136, soundtrack=False, fps=16.0), 2.95, 8.0125),
        _trimmed(seams.write_clip(tmp_path / "fraction.mp4", 136, soundtrack=False, fps=16.0), 1.0, 6.1),
        _trimmed(seams.write_clip(tmp_path / "exact.mp4", 136, soundtrack=False, fps=16.0), 0.0625, 5.125),
    ]
    t = _shared_ingestion(tmp_path, monkeypatch, pairs, num_frames=81)
    frames = {i["id"]: int(i["target_frames"]) for i in t.inventory}
    assert frames == {"gained.mp4": 81, "fraction.mp4": 81, "exact.mp4": 81}
    for item in t.inventory:
        assert _load(item).shape[1] == 81, item["id"]


def test_a_window_one_frame_short_of_a_rule_length_still_snaps_down(tmp_path, monkeypatch):
    """The negative for the tolerance: 80 frames is 80 frames (-> 77)."""
    pairs = [_trimmed(seams.write_clip(tmp_path / "eighty.mp4", 136, soundtrack=False, fps=16.0), 0.0, 5.0)]
    t = _shared_ingestion(tmp_path, monkeypatch, pairs, num_frames=81)
    (item,) = t.inventory
    assert int(item["target_frames"]) == 77
