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


# ── Response models ──────────────────────────────────────────────────────


class HistogramResponse(BaseModel):
    """Per-channel histograms (R, G, B, luminance), each 256 bins."""
    r: list[int]
    g: list[int]
    b: list[int]
    luminance: list[int]
