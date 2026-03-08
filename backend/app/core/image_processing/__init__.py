"""Image processing pipeline — re-exports all public functions.

This package replaces the monolithic ``image_adjustments.py`` module.
All public symbols are re-exported here for backward compatibility.
"""

from app.core.image_processing.curves import (
    CurvePoint,
    CubeLUTData,
    apply_curves,
    parse_cube_file,
    apply_lut_cube,
    export_curves_as_cube,
)
from app.core.image_processing.color import (
    apply_hue_saturation,
    apply_contrast,
    apply_white_balance,
)
from app.core.image_processing.spatial import (
    apply_sharpening,
    apply_vignette,
    apply_lens_correction,
)
from app.core.image_processing.hsl import (
    HSL_RANGES,
    apply_hsl_selective,
)
from app.core.image_processing.color_match import (
    apply_color_match,
    compute_histogram,
)
from app.core.image_processing.pipeline import apply_all

__all__ = [
    # Data structures
    "CurvePoint",
    "CubeLUTData",
    # Curves & LUT
    "apply_curves",
    "parse_cube_file",
    "apply_lut_cube",
    "export_curves_as_cube",
    # Color
    "apply_hue_saturation",
    "apply_contrast",
    "apply_white_balance",
    # Spatial
    "apply_sharpening",
    "apply_vignette",
    "apply_lens_correction",
    # HSL
    "HSL_RANGES",
    "apply_hsl_selective",
    # Color Match
    "apply_color_match",
    "compute_histogram",
    # Pipeline
    "apply_all",
]
