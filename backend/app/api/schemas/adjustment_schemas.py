"""Image adjustment schemas — curves, LUT, color, sharpening, etc."""

from __future__ import annotations

from pydantic import BaseModel


class CurvePointModel(BaseModel):
    """A single control point on a curves graph."""
    x: int
    y: int


class CurvesConfig(BaseModel):
    """Per-channel curves configuration."""
    master: list[CurvePointModel] = []
    r: list[CurvePointModel] = []
    g: list[CurvePointModel] = []
    b: list[CurvePointModel] = []


class SharpeningConfig(BaseModel):
    """Sharpening method and parameters."""
    method: str = "unsharp_mask"
    params: dict[str, float] = {}


class WhiteBalanceConfig(BaseModel):
    """White balance configuration."""
    temperature: float = 6500.0
    tint: float = 0.0


class VignetteConfig(BaseModel):
    """Vignette configuration."""
    amount: float = 0.0
    midpoint: float = 0.5
    feather: float = 0.5


class LensCorrectionConfig(BaseModel):
    """Lens correction configuration."""
    barrel: float = 0.0
    vertical_keystone: float = 0.0
    horizontal_keystone: float = 0.0


class HSLRangeConfig(BaseModel):
    """Per-hue-range HSL adjustment."""
    hue_shift: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0


class ColorMatchConfig(BaseModel):
    """Color match configuration (embedded in adjustment requests)."""
    reference_path: str
    method: str = "cdf"
    strength: float = 1.0


class AdjustmentRequest(BaseModel):
    """Request body for single-image adjustments."""
    path: str
    color_match: ColorMatchConfig | None = None
    curves: CurvesConfig | None = None
    cube_lut: str | None = None
    cube_lut_strength: float = 1.0
    hue_shift: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0
    sharpening: SharpeningConfig | None = None
    white_balance: WhiteBalanceConfig | None = None
    vignette: VignetteConfig | None = None
    lens_correction: LensCorrectionConfig | None = None
    hsl_selective: dict[str, HSLRangeConfig] | None = None


class BatchAdjustmentRequest(BaseModel):
    """Request body for batch image adjustments."""
    paths: list[str]
    color_match: ColorMatchConfig | None = None
    curves: CurvesConfig | None = None
    cube_lut: str | None = None
    cube_lut_strength: float = 1.0
    hue_shift: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0
    sharpening: SharpeningConfig | None = None
    white_balance: WhiteBalanceConfig | None = None
    vignette: VignetteConfig | None = None
    lens_correction: LensCorrectionConfig | None = None
    hsl_selective: dict[str, HSLRangeConfig] | None = None


class ColorMatchRequest(BaseModel):
    """Request body for standalone color match preview."""
    source_path: str
    reference_path: str
    method: str = "cdf"
    strength: float = 1.0


class ExportCubeRequest(BaseModel):
    """Request body to export curves as a .cube LUT file."""
    curves: CurvesConfig
    size: int = 33
