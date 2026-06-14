"""Video caption dispatch tests.

Covers the routing decision in ``CaptionService.generate_caption`` (video →
``generate_video`` with N frames; image → ``generate``) and the batch worker's
masked-skip-for-video behaviour. PyAV synthesises a tiny clip so the suite needs
no fixture media.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import av
from PIL import Image

from app.core.captioning.caption_service import CaptionService
from app.core.captioning.models.base import CaptionModel


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_clip(path: Path, *, n_frames: int = 12, fps: int = 8) -> Path:
    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream("libx264", rate=fps)
        vstream.width = 48
        vstream.height = 48
        vstream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((48, 48, 3), (i * 18) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
    return path


class RecordingModel(CaptionModel):
    """Fake caption model recording which generate path was taken."""

    supports_multi_image = True

    def __init__(self):
        self.image_calls = 0
        self.video_calls = 0
        self.video_frame_counts: list[int] = []

    def load(self, variant: str = None):
        return None, None

    def unload(self):
        pass

    @property
    def model_id(self) -> str:
        return "recording"

    def generate(self, image: Image.Image, params: dict) -> str:
        self.image_calls += 1
        return "image-caption"

    def generate_video(self, frames: list, params: dict) -> str:
        self.video_calls += 1
        self.video_frame_counts.append(len(frames))
        return "video-caption"


@pytest.fixture
def svc():
    CaptionService._instance = None
    CaptionService._active_model_key = None
    service = CaptionService.get_instance()
    yield service
    CaptionService._instance = None
    CaptionService._active_model_key = None


# ── Service dispatch ─────────────────────────────────────────────────────


def test_video_item_routes_to_generate_video_with_n_frames(svc, tmp_path):
    clip = _write_clip(tmp_path / "clip.mp4", n_frames=12)
    rec = RecordingModel()
    svc.plugins["recording"] = rec

    caption = svc.generate_caption(str(clip), "recording", {"video_frames": 4})

    assert caption == "video-caption"
    assert rec.video_calls == 1
    assert rec.image_calls == 0
    assert rec.video_frame_counts == [4]


def test_image_item_routes_to_generate_caption(svc, tmp_path):
    img_path = tmp_path / "still.png"
    Image.new("RGB", (32, 32)).save(img_path)
    rec = RecordingModel()
    svc.plugins["recording"] = rec

    caption = svc.generate_caption(str(img_path), "recording", {})

    assert caption == "image-caption"
    assert rec.image_calls == 1
    assert rec.video_calls == 0


def test_explicit_is_video_flag_forces_video_path(svc, tmp_path):
    """An item whose extension is ambiguous still routes to video when the
    media metadata flag is supplied via params['is_video']."""
    clip = _write_clip(tmp_path / "noext_clip.mp4", n_frames=10)
    rec = RecordingModel()
    svc.plugins["recording"] = rec

    svc.generate_caption(str(clip), "recording", {"is_video": True, "video_frames": 3})

    assert rec.video_calls == 1
    assert rec.video_frame_counts == [3]


def test_default_frame_count_used_when_unspecified(svc, tmp_path):
    from app.core.captioning.caption_service import DEFAULT_VIDEO_FRAMES

    clip = _write_clip(tmp_path / "default_n.mp4", n_frames=24, fps=12)
    rec = RecordingModel()
    svc.plugins["recording"] = rec

    svc.generate_caption(str(clip), "recording", {})

    assert rec.video_calls == 1
    # Clip has 24 frames so the default 8 should all be sampled.
    assert rec.video_frame_counts[0] == DEFAULT_VIDEO_FRAMES


def test_base_generate_video_falls_back_to_middle_frame():
    """A model without a video override captions the middle frame via generate."""

    class SingleFrameModel(CaptionModel):
        def __init__(self):
            self.seen = None

        def load(self, variant: str = None):
            return None, None

        def unload(self):
            pass

        @property
        def model_id(self) -> str:
            return "single"

        def generate(self, image, params):
            self.seen = image
            return "single-caption"

    model = SingleFrameModel()
    frames = [Image.new("RGB", (8, 8), (i, i, i)) for i in (10, 20, 30)]
    out = model.generate_video(frames, {})

    assert out == "single-caption"
    # Middle of 3 frames is index 1 → grey value 20.
    assert model.seen.getpixel((0, 0)) == (20, 20, 20)


# ── Batch: masked skip for video ─────────────────────────────────────────


def test_batch_skips_masked_target_for_video(monkeypatch, tmp_path):
    """A masked-caption batch must SKIP video items (masks are out of scope for
    video) with a warning, not fail them — and never call the masked source
    resolver for the video item."""
    from app.core.captioning import caption_batch
    from app.core.tasks.task_manager import task_manager

    task_manager.set_loop(None)

    clip = _write_clip(tmp_path / "vid.mp4", n_frames=8)

    class Svc:
        def generate_caption(
            self, image_path, model_id, params, extra_image_paths=None
        ):
            return "should-not-be-called-for-skipped-video"

    masked_calls = []
    writes = []

    monkeypatch.setattr(caption_batch, "_get_service", lambda: Svc())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", lambda: None)
    monkeypatch.setattr(
        caption_batch, "_video_meta", lambda ds, rel: {"is_video": True}
    )
    monkeypatch.setattr(
        caption_batch,
        "_masked_path",
        lambda ds, rel: masked_calls.append(rel) or str(clip),
    )
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: str(clip))
    monkeypatch.setattr(
        caption_batch,
        "_write_caption",
        lambda ds, rel, text, target: writes.append(rel),
    )
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    t = task_manager.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id,
        dataset_name="ds",
        image_rel_paths=["vid.mp4"],
        model_id="recording",
        params={},
        system_prompt=None,
        target="masked",
    )

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert masked_calls == []  # masked source never resolved for the video
    assert writes == []  # nothing written for the skipped video
    assert task.ok == 0
    assert task.failed == 0


def test_batch_routes_video_original_through_video_path(monkeypatch, tmp_path):
    """A non-masked video batch flows the video overlay (is_video + a default
    motion prompt) into the caption call and writes a caption."""
    from app.core.captioning import caption_batch
    from app.core.tasks.task_manager import task_manager

    task_manager.set_loop(None)
    clip = _write_clip(tmp_path / "vid2.mp4", n_frames=8)
    seen_params = []
    writes = []

    class Svc:
        def generate_caption(
            self, image_path, model_id, params, extra_image_paths=None
        ):
            seen_params.append(params)
            return "vid-cap"

    monkeypatch.setattr(caption_batch, "_get_service", lambda: Svc())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", lambda: None)
    monkeypatch.setattr(
        caption_batch,
        "_video_meta",
        lambda ds, rel: {"is_video": True, "trim_start_s": 0.1},
    )
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: str(clip))
    monkeypatch.setattr(
        caption_batch,
        "_write_caption",
        lambda ds, rel, text, target: writes.append((rel, text)),
    )
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    t = task_manager.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id,
        dataset_name="ds",
        image_rel_paths=["vid2.mp4"],
        model_id="recording",
        params={},
        system_prompt=None,
        target="original",
    )

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert writes == [("vid2.mp4", "vid-cap")]
    assert seen_params[0]["is_video"] is True
    assert seen_params[0]["trim_start_s"] == 0.1
    # Motion-aware default prompt injected when no system prompt supplied.
    assert seen_params[0]["system_prompt"] == caption_batch.VIDEO_CAPTION_PROMPT
