"""
Image-only-op route guards for audio files (C0).

Manager-level guards for crop/apply_adjustments/harmonize are unit-tested
directly in ``test_dataset_manager_audio.py`` (they raise ``ValueError``,
which the crop/adjust routes already map to HTTP 400 — pre-existing,
unchanged behavior). This file covers the routes that had NO guard at all
before this change and would otherwise hit ``PIL.Image.open`` directly on
an audio file: upscale, color-match preview, histogram, mask generate/apply.
All use the shared ``app.api._path_guard.reject_audio_op`` helper, which
raises ``HTTPException(400)`` before any decode is attempted.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest


def _fake_dataset(tmp_path):
    return types.SimpleNamespace(path=str(tmp_path))


# ── Upscale ──────────────────────────────────────────────────────────────


@patch("app.api.dataset.upscale_routes.dataset_manager")
def test_upscale_rejects_audio(mock_dm, client, tmp_path):
    (tmp_path / "song.wav").write_bytes(b"not a real wav but existence is enough")
    (tmp_path / "model.pth").write_bytes(b"fake model")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.post("/api/datasets/myds/upscale", json={
        "image_path": "song.wav",
        "model_path": str(tmp_path / "model.pth"),
        "tile_size": 512,
        "tile_pad": 16,
        "target_scale": 2.0,
        "resize_method": "lanczos",
    })

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()


# ── Color Match Preview ─────────────────────────────────────────────────


@patch("app.api.dataset.adjustment_routes.dataset_manager")
def test_color_match_rejects_audio_source(mock_dm, client, tmp_path):
    (tmp_path / "song.wav").write_bytes(b"x")
    (tmp_path / "ref.png").write_bytes(b"x")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.post("/api/datasets/myds/color-match", json={
        "source_path": "song.wav",
        "reference_path": "ref.png",
        "method": "mean_std",
        "strength": 1.0,
    })

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()


# ── Histogram ────────────────────────────────────────────────────────────


@patch("app.api.dataset.adjustment_routes.dataset_manager")
def test_histogram_rejects_audio(mock_dm, client, tmp_path):
    (tmp_path / "song.wav").write_bytes(b"x")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.get("/api/datasets/myds/histogram?image_path=song.wav")

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()


# ── Mask Generate / Apply ────────────────────────────────────────────────


@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_mask_generate_rejects_audio(mock_to_thread, mock_dm, client, tmp_path):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    (tmp_path / "song.wav").write_bytes(b"x")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.post("/api/datasets/myds/masking/generate", json={
        "dataset_name": "myds",
        "image_rel_path": "song.wav",
        "model_id": "rembg",
        "params": {},
    })

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()


@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_mask_apply_rejects_audio(mock_to_thread, mock_dm, client, tmp_path):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    (tmp_path / "song.wav").write_bytes(b"x")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.post("/api/datasets/myds/masking/apply", json={
        "image_rel_path": "song.wav",
        "opacity": 0.5,
    })

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()


# ── Render Pipeline (single) ─────────────────────────────────────────────


@patch("app.api.dataset.overlay_routes.dataset_manager")
def test_render_pipeline_rejects_audio(mock_dm, client, tmp_path):
    (tmp_path / "song.wav").write_bytes(b"x")
    mock_dm.get_dataset.return_value = _fake_dataset(tmp_path)

    response = client.post("/api/datasets/myds/render-pipeline", json={
        "image_path": "song.wav",
        "blocks": [],
        "tile_size": 512,
        "tile_pad": 16,
        "replace_recipe": True,
    })

    assert response.status_code == 400
    assert "audio" in response.json()["detail"].lower()
