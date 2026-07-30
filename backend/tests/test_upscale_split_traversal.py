"""Traversal-guard regression tests for the two sites the W1 sweep missed.

Both accept a client-supplied string that is joined onto a dataset path and
then WRITTEN to, with no containment check pre-fix:

  * ``upscale_routes.upscale_media`` — ``dataset_root / request.image_path``
    is opened by PIL and then ``result_img.save(str(img_path))`` overwrites
    it. The ``exists()`` precondition makes it overwrite-only, so it can
    corrupt any existing file the backend can write (the SQLite DB, a
    definition yaml, a model ``.safetensors``, the served frontend bundle).
  * ``video_routes.video_split`` → ``split_batch`` — ``output_prefix`` reaches
    ``source_dir / f"{prefix}_{i:03d}.mp4"`` and ffmpeg runs with ``-y``, so
    an escaping prefix overwrites outside the dataset.

``upscale_routes.list_upscale_models`` additionally took ``request.folder``
verbatim, making it an arbitrary directory lister (names, full paths, sizes).

Mirrors the local-router TestClient pattern of ``test_media_route_traversal``:
mount just the module's router and monkeypatch the module's
``dataset_manager`` singleton.
"""

from __future__ import annotations

import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.tasks.task_manager import task_manager
from app.core.video import split_batch


def _client(module):
    app = FastAPI()
    app.include_router(module.router, prefix="/api")
    return TestClient(app)


def _fake_dataset(tmp_path):
    return types.SimpleNamespace(path=str(tmp_path), media_metadata={})


# ── upscale_routes ────────────────────────────────────────────────────────


class TestUpscaleTraversal:
    def test_upscale_rejects_traversal_and_leaves_outside_file_untouched(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image

        from app.api.dataset import upscale_routes

        dataset_dir = tmp_path / "ds"
        dataset_dir.mkdir()
        outside = tmp_path / "secret.png"
        Image.new("RGB", (8, 8), "red").save(outside)
        original_bytes = outside.read_bytes()

        client = _client(upscale_routes)
        monkeypatch.setattr(
            upscale_routes.dataset_manager,
            "get_dataset",
            lambda name: _fake_dataset(dataset_dir),
        )

        res = client.post(
            "/api/datasets/ds/upscale",
            json={"model_path": str(tmp_path / "m.pth"), "image_path": "../secret.png"},
        )
        assert res.status_code == 403
        # Pixels were never re-encoded over the out-of-dataset file.
        assert outside.read_bytes() == original_bytes

    def test_list_models_rejects_folder_outside_allowed_roots(
        self, tmp_path, monkeypatch
    ):
        from app.api.dataset import upscale_routes

        client = _client(upscale_routes)
        res = client.post(
            "/api/upscale/list-models", json={"folder": str(tmp_path)}
        )
        assert res.status_code == 403


# ── video split: output_prefix ────────────────────────────────────────────


class TestSplitOutputPrefix:
    def test_route_sanitizes_output_prefix_before_enqueue(self, tmp_path, monkeypatch):
        """The route must hand the worker a bare filename component."""
        from app.api.dataset import video_routes

        (tmp_path / "long.mp4").write_bytes(b"\x00")

        # Stub the worker BEFORE the request: the route does a function-local
        # ``from ... import run_video_split_batch``, so the enqueued closure
        # captures whatever the module attribute is at route-execution time.
        seen: dict = {}
        monkeypatch.setattr(
            split_batch, "run_video_split_batch", lambda tid, **kw: seen.update(kw)
        )

        client = _client(video_routes)
        monkeypatch.setattr(
            video_routes.dataset_manager,
            "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )
        captured: dict = {}
        monkeypatch.setattr(
            video_routes.task_manager,
            "enqueue",
            lambda tid, fn, **kw: captured.setdefault("fn", fn),
        )

        res = client.post(
            "/api/datasets/ds/video/split",
            json={
                "source_rel_path": "long.mp4",
                "segments": [{"start_s": 0.0, "end_s": 0.5}],
                "output_prefix": "../../pwned",
            },
        )
        assert res.status_code == 200

        captured["fn"]("task-1")
        assert seen["output_prefix"] == "pwned"

    @pytest.mark.parametrize("prefix", ["../../pwned", "..", "C:/abs/pwned", "a/b/c"])
    def test_worker_never_writes_outside_dataset_dir(
        self, tmp_path, monkeypatch, prefix
    ):
        """Defence in depth: the worker contains its OWN output path.

        Mirrors the ``_render_one`` precedent — both current callers guard, but
        a future caller (resume, retry, replay from a persisted task record)
        must not be able to reopen the write primitive.
        """
        task_manager.set_loop(None)
        dataset_dir = tmp_path / "ds"
        dataset_dir.mkdir()
        (dataset_dir / "long.mp4").write_bytes(b"\x00")

        monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda name: dataset_dir)
        monkeypatch.setattr(split_batch, "_scan", lambda name: None)
        monkeypatch.setattr(split_batch, "_nearest_keyframe", lambda p, t: 0.0)

        written: list[str] = []

        def _fake_ffmpeg(args, progress_cb=None, *, should_abort=None, timeout=None):
            written.append(args[-1])  # out_path is the last argv entry
            open(args[-1], "wb").close()
            return 0

        monkeypatch.setattr(split_batch, "_run_ffmpeg", _fake_ffmpeg)

        t = task_manager.create(
            type="video_split", title="x", total=1, dataset_name="ds"
        )
        split_batch.run_video_split_batch(
            t.id,
            dataset_name="ds",
            source_rel_path="long.mp4",
            segments=[{"start_s": 0.0, "end_s": 0.5}],
            mode="copy",
            output_prefix=prefix,
            archive_source=False,
        )

        assert written, "no segment was cut"
        for out in written:
            from pathlib import Path

            assert Path(out).resolve().is_relative_to(dataset_dir.resolve()), out
