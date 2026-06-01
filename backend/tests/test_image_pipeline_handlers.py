"""Tests for the per-block handlers in app.core.image_processing.pipeline.

The handlers gate the call to the underlying worker on a "non-default
params" check so identity-only blocks can be skipped cheaply. This
file locks the contract for the HSL handler in particular, which
historically expected its params wrapped in an ``hsl_config`` key
(legacy ``apply_all`` interface) while the new frontend now sends
the bands flat under the block's ``params``. The handler must accept
both shapes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.image_processing.pipeline import (
    PipelineBlock,
    _handle_hsl_selective,
    execute_pipeline,
)


def _make_image(w: int = 64, h: int = 64, color: tuple = (128, 128, 80)) -> Image.Image:
    """Solid-color image — yellow-leaning so HSL on the "yellows" band
    produces a visible difference."""
    return Image.new("RGB", (w, h), color)


def _arrays_differ(a: Image.Image, b: Image.Image) -> bool:
    """True iff the two images are pixel-perfect non-equal."""
    return not np.array_equal(np.array(a), np.array(b))


class TestHandleHslSelective:
    """Direct unit tests for ``_handle_hsl_selective``."""

    def test_flat_params_apply(self):
        """New-frontend shape: bands sit directly under ``params``.

        User-reported regression — the editor preview applies HSL
        (frontend-side) but the saved overlay PNG didn't, because
        the backend handler was reading ``params["hsl_config"]`` and
        finding ``None``. With the fix, the handler should treat
        the whole params dict as the bands map.
        """
        img = _make_image(color=(180, 160, 40))    # saturated yellow
        bands = {
            "yellows": {"hue_shift": -16.0, "saturation": 0.0, "luminance": 0.0},
        }
        result = _handle_hsl_selective(img, bands)
        assert _arrays_differ(img, result), (
            "HSL handler must apply when bands are sent flat (new frontend)"
        )

    def test_legacy_wrapped_params_still_apply(self):
        """Legacy ``apply_all`` interface wraps the bands in ``hsl_config``.

        That code path still exists in ``pipeline.apply_all`` for
        backward compat, so the handler must keep accepting the
        wrapped shape.
        """
        img = _make_image(color=(180, 160, 40))
        bands = {
            "yellows": {"hue_shift": -16.0, "saturation": 0.0, "luminance": 0.0},
        }
        result = _handle_hsl_selective(img, {"hsl_config": bands})
        assert _arrays_differ(img, result), (
            "HSL handler must still accept the legacy ``hsl_config``-wrapped shape"
        )

    def test_empty_params_no_op(self):
        """Empty params dict should not modify the image."""
        img = _make_image()
        result = _handle_hsl_selective(img, {})
        assert np.array_equal(np.array(img), np.array(result))

    def test_all_zero_bands_no_op(self):
        """All-zero band adjustments are detected by the worker and
        skipped — handler returns unmodified image."""
        img = _make_image()
        bands = {
            "yellows": {"hue_shift": 0.0, "saturation": 0.0, "luminance": 0.0},
            "blues":   {"hue_shift": 0.0, "saturation": 0.0, "luminance": 0.0},
        }
        result = _handle_hsl_selective(img, bands)
        assert np.array_equal(np.array(img), np.array(result))

    def test_execute_pipeline_flat_hsl_block_applies(self):
        """End-to-end through ``execute_pipeline``: an hsl_selective
        block with flat band params should actually modify the image.

        This is the contract the new frontend's ``renderPipeline``
        save path relies on.
        """
        img = _make_image(color=(180, 160, 40))
        block = PipelineBlock(
            type="hsl_selective",
            enabled=True,
            params={
                "yellows": {"hue_shift": -16.0, "saturation": 0.0, "luminance": 0.0},
            },
        )
        result = execute_pipeline(img, [block])
        assert _arrays_differ(img, result), (
            "execute_pipeline must apply hsl_selective with flat band params"
        )
