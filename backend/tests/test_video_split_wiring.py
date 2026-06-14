"""Light wiring tests for video split + routes.

Synthesizes a tiny real mp4 with PyAV, then:
  * exercises the cutlist/parse endpoint (multipart upload),
  * runs the split worker over 2 short segments and asserts 2 output mp4s
    exist + the source was archived,
  * checks the split/scene-detect routes enqueue on the cpu lane.

The split worker's rescan seam is stubbed so the test never touches the real
DatasetManager / DB.
"""

import json

import pytest

from app.core.tasks.task_manager import task_manager
from app.core.video import split_batch


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_loop():
    """Detach the task broadcast loop (mirrors test_crop_batch)."""
    task_manager.set_loop(None)


def _make_mp4(path, *, seconds=2.0, fps=10, w=64, h=64):
    """Encode a tiny solid-color mp4 (no external ffmpeg needed — PyAV bundles it)."""
    import av
    import numpy as np

    n_frames = int(seconds * fps)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        arr = np.full((h, w, 3), (i * 8) % 256, dtype="uint8")
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def tiny_dataset(tmp_path):
    """A dataset dir containing one tiny source mp4. Returns (dir, source_name)."""
    src_name = "long.mp4"
    _make_mp4(tmp_path / src_name)
    return tmp_path, src_name


# ── Split worker ─────────────────────────────────────────────────────────────


def test_split_produces_two_clips_and_archives(monkeypatch, tiny_dataset):
    dataset_dir, src_name = tiny_dataset

    # Point the worker's source-dir seam at our tmp dataset.
    monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda name: dataset_dir)
    # Stub the rescan so we don't hit DatasetManager / DB.
    scanned = []
    monkeypatch.setattr(split_batch, "_scan", lambda name: scanned.append(name))

    segments = [
        {"start_s": 0.0, "end_s": 0.5},
        {"start_s": 0.5, "end_s": 1.0},
    ]
    t = task_manager.create(type="video_split", title="x", total=2, dataset_name="ds")
    split_batch.run_video_split_batch(
        t.id,
        dataset_name="ds",
        source_rel_path=src_name,
        segments=segments,
        mode="reencode",  # force re-encode → no keyframe dependence, deterministic
        output_prefix=None,
        archive_source=True,
    )

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 2
    assert task.failed == 0

    # Two output clips landed in the dataset dir.
    assert (dataset_dir / "long_000.mp4").exists()
    assert (dataset_dir / "long_001.mp4").exists()

    # Source archived into the dot-dir; gone from the dataset root.
    assert (dataset_dir / ".video_sources" / "long.mp4").exists()
    assert not (dataset_dir / "long.mp4").exists()

    # Rescan was triggered once.
    assert scanned == ["ds"]


def test_split_missing_source_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda name: tmp_path)
    monkeypatch.setattr(split_batch, "_scan", lambda name: None)

    t = task_manager.create(type="video_split", title="x", total=1, dataset_name="ds")
    split_batch.run_video_split_batch(
        t.id,
        dataset_name="ds",
        source_rel_path="nope.mp4",
        segments=[{"start_s": 0.0, "end_s": 1.0}],
        mode="reencode",
    )
    assert task_manager.get(t.id).status.value == "failed"


def test_split_segment_failure_isolated(monkeypatch, tiny_dataset):
    dataset_dir, src_name = tiny_dataset
    monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda name: dataset_dir)
    monkeypatch.setattr(split_batch, "_scan", lambda name: None)

    calls = {"n": 0}
    real_cut = split_batch._cut_segment

    def flaky_cut(src, out, start, end, mode):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_cut(src, out, start, end, mode)

    monkeypatch.setattr(split_batch, "_cut_segment", flaky_cut)

    t = task_manager.create(type="video_split", title="x", total=2, dataset_name="ds")
    split_batch.run_video_split_batch(
        t.id,
        dataset_name="ds",
        source_rel_path=src_name,
        segments=[{"start_s": 0.0, "end_s": 0.5}, {"start_s": 0.5, "end_s": 1.0}],
        mode="reencode",
        archive_source=False,
    )
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 1
    assert task.failed == 1


# ── Cutlist parse endpoint ───────────────────────────────────────────────────


def test_cutlist_parse_endpoint(monkeypatch, client):
    from app.api.dataset import video_routes

    # Stub dataset resolution so we don't need a real registered dataset.
    class _DS:
        path = "."

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())

    body = json.dumps(
        {
            "cutSegments": [
                {"start": 0.0, "end": 1.0, "name": "a"},
                {"start": 1.0, "end": 2.0},
            ]
        }
    ).encode()

    resp = client.post(
        "/api/datasets/ds/video/cutlist/parse",
        files={"file": ("project.llc", body, "application/json")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["format"] == "llc"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["label"] == "a"
    assert data["segments"][0]["start_s"] == 0.0


# ── Route enqueue (cpu lane) ─────────────────────────────────────────────────


def test_split_route_enqueues_cpu_lane(monkeypatch, client):
    from app.api.dataset import video_routes

    class _DS:
        path = "."

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())
    monkeypatch.setattr(video_routes, "_guard_source", lambda ds, p: p)

    captured = {}

    def fake_enqueue(task_id, worker_fn, *, lane="gpu"):
        captured["task_id"] = task_id
        captured["lane"] = lane

    monkeypatch.setattr(video_routes.task_manager, "enqueue", fake_enqueue)

    resp = client.post(
        "/api/datasets/ds/video/split",
        json={
            "source_rel_path": "long.mp4",
            "segments": [{"start_s": 0.0, "end_s": 1.0, "label": None}],
            "mode": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["task_id"] == captured["task_id"]
    assert captured["lane"] == "cpu"


# ── Scene-detect worker ──────────────────────────────────────────────────────


def _make_mp4_with_cut(path, *, fps=10, w=64, h=64):
    """Encode a clip with a hard scene change at the midpoint (for ContentDetector)."""
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(30):
        v = 0 if i < 15 else 240  # abrupt black → white cut at frame 15
        arr = np.full((h, w, 3), v, dtype="uint8")
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_scene_detect_worker_writes_proposals(monkeypatch, tmp_path):
    """The worker detects scenes on a real clip and writes the proposals JSON."""
    from app.core.video import scene_detect_batch

    _make_mp4_with_cut(tmp_path / "long.mp4")
    monkeypatch.setattr(
        scene_detect_batch, "_resolve_dataset_path", lambda name: tmp_path
    )

    t = task_manager.create(type="scene_detect", title="x", total=30, dataset_name="ds")
    scene_detect_batch.run_scene_detect_batch(
        t.id,
        dataset_name="ds",
        source_rel_path="long.mp4",
        threshold=27.0,
        min_scene_len_s=0.2,
    )
    assert task_manager.get(t.id).status.value == "completed"

    proposals = tmp_path / ".video" / "proposals" / "long.json"
    assert proposals.exists()
    data = json.loads(proposals.read_text(encoding="utf-8"))
    assert data["source_rel_path"] == "long.mp4"
    assert isinstance(data["segments"], list)
    # Hard black→white cut → at least 2 scene proposals.
    assert len(data["segments"]) >= 2
    for seg in data["segments"]:
        assert "start_s" in seg and "end_s" in seg


def test_auto_mode_decides_copy_on_keyframe(monkeypatch, tiny_dataset):
    """auto mode stream-copies when a keyframe sits near the segment start."""
    dataset_dir, src_name = tiny_dataset
    monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda name: dataset_dir)
    monkeypatch.setattr(split_batch, "_scan", lambda name: None)

    # Force the keyframe seam to report a keyframe exactly at the start.
    monkeypatch.setattr(split_batch, "_nearest_keyframe", lambda path, t: 0.0)

    used_modes = []
    real_run = split_batch._run_ffmpeg

    def spy_args(args, progress_cb=None):
        # "-c copy" present → stream-copy chosen.
        used_modes.append("copy" if "copy" in args else "reencode")
        return real_run(args, progress_cb)

    monkeypatch.setattr(split_batch, "_run_ffmpeg", spy_args)

    t = task_manager.create(type="video_split", title="x", total=1, dataset_name="ds")
    split_batch.run_video_split_batch(
        t.id,
        dataset_name="ds",
        source_rel_path=src_name,
        segments=[{"start_s": 0.0, "end_s": 1.0}],
        mode="auto",
        archive_source=False,
    )
    assert used_modes == ["copy"]
    assert task_manager.get(t.id).ok == 1


def test_scene_detect_route_enqueues_cpu_lane(monkeypatch, client, tiny_dataset):
    from pathlib import Path

    from app.api.dataset import video_routes

    dataset_dir, src_name = tiny_dataset

    class _DS:
        path = str(dataset_dir)

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())
    monkeypatch.setattr(
        video_routes, "_guard_source", lambda ds, p: Path(dataset_dir) / p
    )

    captured = {}

    def fake_enqueue(task_id, worker_fn, *, lane="gpu"):
        captured["lane"] = lane

    monkeypatch.setattr(video_routes.task_manager, "enqueue", fake_enqueue)

    resp = client.post(
        "/api/datasets/ds/video/scene-detect",
        json={"source_rel_path": src_name, "threshold": 27.0, "min_scene_len_s": 1.0},
    )
    assert resp.status_code == 200, resp.text
    assert captured["lane"] == "cpu"


# ── Trim + health routes ─────────────────────────────────────────────────────


def test_trim_route_updates_and_returns_warnings(monkeypatch, client):
    from app.api.dataset import video_routes
    from app.core.dataset_manager import dataset_manager

    # fps 16, full clip 66 frames → 66%4==2 warns; trim to 65 frames → healthy.
    meta = {
        "fps": 16.0,
        "duration_s": 66 / 16.0,
        "width": 512,
        "height": 512,
        "has_audio": False,
        "is_video": True,
    }

    class _DS:
        name = "ds"
        path = "."
        media_metadata = {"clip.mp4": meta}

    ds = _DS()
    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: ds)

    persisted = []

    async def fake_persist(dataset, rel):
        persisted.append(rel)

    monkeypatch.setattr(dataset_manager, "_persist_media_item_async", fake_persist)

    events = []

    async def fake_broadcast(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(video_routes.event_manager, "broadcast", fake_broadcast)

    resp = client.patch(
        "/api/datasets/ds/video/trim",
        json={"media_file": "clip.mp4", "trim_start_s": 0.0, "trim_end_s": 65 / 16.0},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "trimmed"
    # wan frame-rule satisfied after the trim → no frame-count warning.
    assert all("frame count" not in w for w in data["clip_warnings"]["wan"])
    # The in-memory metadata was updated + persisted + invalidation broadcast.
    assert meta["trim_end_s"] == 65 / 16.0
    assert persisted == ["clip.mp4"]
    assert ("dataset.invalidated", {"name": "ds"}) in events


def test_trim_route_404_for_unknown_media(monkeypatch, client):
    from app.api.dataset import video_routes

    class _DS:
        name = "ds"
        path = "."
        media_metadata = {}

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())
    resp = client.patch(
        "/api/datasets/ds/video/trim",
        json={"media_file": "nope.mp4", "trim_start_s": 0.0, "trim_end_s": 1.0},
    )
    assert resp.status_code == 404


def test_health_route_summary(monkeypatch, client):
    from app.api.dataset import video_routes

    good = {
        "fps": 16.0,
        "duration_s": 65 / 16.0,
        "width": 512,
        "height": 512,
        "has_audio": True,
        "is_video": True,
    }
    bad = {
        "fps": 16.0,
        "duration_s": 64 / 16.0,
        "width": 500,
        "height": 512,
        "has_audio": False,
        "is_video": True,
    }
    not_video = {"is_video": False}

    class _DS:
        name = "ds"
        path = "."
        media_metadata = {"good.mp4": good, "bad.mp4": bad, "img.png": not_video}

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())

    resp = client.get("/api/datasets/ds/video/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 2  # the non-video item is excluded
    wan = data["families"]["wan"]
    assert wan["healthy"] == 1
    assert wan["warning"] == 1
    assert wan["offenders"][0]["media_file"] == "bad.mp4"


def test_scene_proposals_not_ready(monkeypatch, client, tmp_path):
    from app.api.dataset import video_routes

    class _DS:
        path = str(tmp_path)

    monkeypatch.setattr(video_routes, "_resolve_dataset", lambda name: _DS())
    resp = client.get(
        "/api/datasets/ds/video/scene-proposals",
        params={"source_rel_path": "long.mp4"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"segments": [], "ready": False}
