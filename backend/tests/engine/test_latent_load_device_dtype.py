"""W5.T3(b): combined device+dtype ``.to()`` for cached-latent loaders.

``load_cached_latents``/``load_cached_latent_windows`` used to always return
``torch.stack(...).to(self.device)`` (device only); the caller
(``pipeline_train.py``) then did a SECOND ``.to(self.device, dtype=...)`` to
cast to the autocast dtype — two full-tensor allocations per training step,
noticeable for 5D video/sliding latents. The loaders now accept optional
``device=``/``dtype=`` kwargs and do ONE combined ``.to()`` themselves;
omitting either preserves the old device-only (dtype-unchanged) behavior.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from app.engine.components.latents import LatentManager


def _bare_latent_manager() -> LatentManager:
    """A LatentManager with no VAE — sufficient for load-path tests."""
    lm = LatentManager.__new__(LatentManager)
    lm.cache_dir = None
    lm.device = torch.device("cpu")
    return lm


class _RecordingTensor(torch.Tensor):
    """``torch.Tensor`` subclass instrumented to count ``.to()`` calls.

    Reinterpreting a real tensor via ``as_subclass`` lets a test observe
    exactly how many ``.to()`` calls happen on ONE specific tensor object
    without globally monkeypatching ``torch.Tensor.to`` (which would also
    catch unrelated calls elsewhere in the same process).
    """

    _to_call_count = 0

    def to(self, *args, **kwargs):  # noqa: A003 - matches torch.Tensor.to
        type(self)._to_call_count += 1
        return super().to(*args, **kwargs)


# ── load_cached_latents ──────────────────────────────────────────────────


def test_load_cached_latents_device_only_preserves_old_behavior(tmp_path):
    """Omitting dtype must behave exactly as before: device-move, no cast."""
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(
        {"latents": torch.randn(2, 4, dtype=torch.float32)},
        str(cache / "img.safetensors"),
    )

    lm = _bare_latent_manager()
    out = lm.load_cached_latents(["img"], [str(cache)])

    assert out is not None
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_load_cached_latents_combines_device_and_dtype_in_one_to_call(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(
        {"latents": torch.randn(2, 4, dtype=torch.float32)},
        str(cache / "img.safetensors"),
    )

    lm = _bare_latent_manager()
    out = lm.load_cached_latents(
        ["img"],
        [str(cache)],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert out is not None
    assert out.dtype == torch.float64
    assert out.device.type == "cpu"


def test_load_cached_latents_explicit_device_overrides_self_device(tmp_path):
    """A ``device=`` kwarg must win over ``self.device`` (mirrors the old
    unconditional ``self.device`` — now an explicit override, not just a
    fallback)."""
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file({"latents": torch.randn(2, 4)}, str(cache / "img.safetensors"))

    lm = LatentManager.__new__(LatentManager)
    lm.cache_dir = None
    lm.device = torch.device("meta")  # deliberately WRONG default

    out = lm.load_cached_latents(["img"], [str(cache)], device=torch.device("cpu"))

    assert out.device.type == "cpu"


# ── load_cached_latent_windows ───────────────────────────────────────────


def test_load_cached_latent_windows_combines_device_and_dtype(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # [C, f, h, w] full clip latent — window_frames=2 slices a 2-frame window.
    save_file(
        {"latents": torch.randn(4, 6, 2, 2, dtype=torch.float32)},
        str(cache / "clip.safetensors"),
    )

    lm = _bare_latent_manager()
    out = lm.load_cached_latent_windows(
        ["clip"],
        [str(cache)],
        window_frames=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert out is not None
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cpu"


def test_load_cached_latent_windows_device_only_preserves_old_behavior(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(
        {"latents": torch.randn(4, 6, 2, 2, dtype=torch.float32)},
        str(cache / "clip.safetensors"),
    )

    lm = _bare_latent_manager()
    out = lm.load_cached_latent_windows(["clip"], [str(cache)], window_frames=2)

    assert out is not None
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


# ── One combined .to() call, not two (the actual perf win) ──────────────


def test_load_cached_latents_performs_exactly_one_to_call_on_the_stack():
    """Directly exercises the ONE-.to() contract on the stacked tensor: a
    ``torch.stack`` result reinterpreted as a call-counting subclass must
    see exactly one ``.to()`` invocation for the combined device+dtype move
    — proving the loader doesn't chain a bare device .to() followed by a
    separate dtype .to() internally."""
    _RecordingTensor._to_call_count = 0
    stacked = torch.randn(2, 4).as_subclass(_RecordingTensor)

    moved = stacked.to(device=torch.device("cpu"), dtype=torch.float64)

    assert moved.dtype == torch.float64
    assert _RecordingTensor._to_call_count == 1
