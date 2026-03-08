"""Curves and CUBE LUT operations — PCHIP interpolation, 3D LUT, cube export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.interpolate import PchipInterpolator


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CurvePoint:
    """Single control point on a curves graph (0-255 range)."""

    x: int
    y: int


@dataclass(frozen=True)
class CubeLUTData:
    """Parsed contents of a .cube 3D-LUT file."""

    title: str
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    table: NDArray[np.float32]  # shape (size, size, size, 3)


# ── Internal Helpers ─────────────────────────────────────────────────────


def _build_lut_from_points(points: list[CurvePoint]) -> NDArray[np.uint8]:
    """Interpolate control points into a 256-entry LUT using PCHIP."""
    if len(points) < 2:
        return np.arange(256, dtype=np.uint8)

    sorted_pts = sorted(points, key=lambda p: p.x)
    xs = np.array([p.x for p in sorted_pts], dtype=np.float64)
    ys = np.array([p.y for p in sorted_pts], dtype=np.float64)

    # Ensure endpoints cover full range if not already present
    if xs[0] != 0:
        xs = np.concatenate([[0], xs])
        ys = np.concatenate([[0], ys])
    if xs[-1] != 255:
        xs = np.concatenate([xs, [255]])
        ys = np.concatenate([ys, [255]])

    interp = PchipInterpolator(xs, ys)
    lut = interp(np.arange(256))
    return np.clip(lut, 0, 255).astype(np.uint8)


# ── Public API ───────────────────────────────────────────────────────────


def apply_curves(
    img: Image.Image,
    master: list[CurvePoint] | None = None,
    r: list[CurvePoint] | None = None,
    g: list[CurvePoint] | None = None,
    b: list[CurvePoint] | None = None,
) -> Image.Image:
    """Apply per-channel curves via LUT.  Master curve is applied first."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)

    # Master (applied to all channels)
    if master and len(master) > 1:
        m_lut = _build_lut_from_points(master)
        arr = m_lut[arr]

    # Individual channels
    for ch_idx, pts in enumerate([r, g, b]):
        if pts and len(pts) > 1:
            ch_lut = _build_lut_from_points(pts)
            arr[:, :, ch_idx] = ch_lut[arr[:, :, ch_idx]]

    return Image.fromarray(arr, "RGB")


def parse_cube_file(content: str) -> CubeLUTData:
    """Parse a .cube file string into a CubeLUTData object."""
    title = ""
    size = 0
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    data_lines: list[list[float]] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        upper = line.upper()
        if upper.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUT files are not supported — only 3D LUT (.cube) is accepted")
        elif upper.startswith("TITLE"):
            title = line.split('"')[1] if '"' in line else line.split(None, 1)[1]
        elif upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
        elif upper.startswith("DOMAIN_MIN"):
            parts = line.split()
            domain_min = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif upper.startswith("DOMAIN_MAX"):
            parts = line.split()
            domain_max = (float(parts[1]), float(parts[2]), float(parts[3]))
        else:
            parts = line.split()
            if len(parts) == 3:
                try:
                    data_lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue

    if size == 0:
        # Auto-detect size from data count
        count = len(data_lines)
        if count == 0:
            raise ValueError("LUT_3D_SIZE not found and no data entries present")
        size = round(count ** (1 / 3))
        if size ** 3 != count:
            raise ValueError(f"LUT_3D_SIZE not found; cannot auto-detect from {count} entries")

    expected = size ** 3
    if len(data_lines) < expected:
        raise ValueError(f"Expected {expected} entries for LUT_3D_SIZE {size}, got {len(data_lines)}")

    table = np.array(data_lines[:expected], dtype=np.float32).reshape(size, size, size, 3)
    return CubeLUTData(title=title, size=size, domain_min=domain_min, domain_max=domain_max, table=table)


def apply_lut_cube(img: Image.Image, lut: CubeLUTData, strength: float = 1.0) -> Image.Image:
    """Apply a 3D CUBE LUT via trilinear interpolation with optional strength blending."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    # Normalize to LUT domain
    dmin = np.array(lut.domain_min, dtype=np.float32)
    dmax = np.array(lut.domain_max, dtype=np.float32)
    drange = dmax - dmin
    drange[drange == 0] = 1.0

    normalized = (arr - dmin) / drange
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to LUT indices
    s = lut.size - 1
    coords = normalized * s

    # Floor/ceil indices for trilinear interpolation
    c0 = np.floor(coords).astype(np.int32)
    c1 = np.minimum(c0 + 1, s)
    frac = coords - c0.astype(np.float32)

    # Flatten for indexing
    # .cube format: R varies fastest (inner), G mid, B slowest (outer)
    # After reshape(size, size, size, 3): axis0=B, axis1=G, axis2=R
    r0, g0, b0 = c0[..., 0], c0[..., 1], c0[..., 2]
    r1, g1, b1 = c1[..., 0], c1[..., 1], c1[..., 2]
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

    # Trilinear interpolation (8 corners) — index as [b, g, r]
    def _lookup(ri: np.ndarray, gi: np.ndarray, bi: np.ndarray) -> np.ndarray:
        return lut.table[bi, gi, ri]

    c000 = _lookup(r0, g0, b0)
    c001 = _lookup(r0, g0, b1)
    c010 = _lookup(r0, g1, b0)
    c011 = _lookup(r0, g1, b1)
    c100 = _lookup(r1, g0, b0)
    c101 = _lookup(r1, g0, b1)
    c110 = _lookup(r1, g1, b0)
    c111 = _lookup(r1, g1, b1)

    fr3 = fr[..., np.newaxis]
    fg3 = fg[..., np.newaxis]
    fb3 = fb[..., np.newaxis]

    c00 = c000 * (1 - fr3) + c100 * fr3
    c01 = c001 * (1 - fr3) + c101 * fr3
    c10 = c010 * (1 - fr3) + c110 * fr3
    c11 = c011 * (1 - fr3) + c111 * fr3

    c0_interp = c00 * (1 - fg3) + c10 * fg3
    c1_interp = c01 * (1 - fg3) + c11 * fg3

    result = c0_interp * (1 - fb3) + c1_interp * fb3

    # Strength blending: (1-s)*original + s*LUT
    if strength < 1.0:
        result = arr * (1.0 - strength) + result * strength

    return Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), "RGB")


def export_curves_as_cube(
    master: list[CurvePoint] | None = None,
    r: list[CurvePoint] | None = None,
    g: list[CurvePoint] | None = None,
    b: list[CurvePoint] | None = None,
    size: int = 33,
) -> str:
    """Bake per-channel curves into a .cube 3D LUT string."""
    m_lut = _build_lut_from_points(master) if master else np.arange(256, dtype=np.uint8)
    r_lut = _build_lut_from_points(r) if r else np.arange(256, dtype=np.uint8)
    g_lut = _build_lut_from_points(g) if g else np.arange(256, dtype=np.uint8)
    b_lut = _build_lut_from_points(b) if b else np.arange(256, dtype=np.uint8)

    lines = [
        'TITLE "Exported Curves LUT"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]

    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                rv = int(round(ri / (size - 1) * 255))
                gv = int(round(gi / (size - 1) * 255))
                bv = int(round(bi / (size - 1) * 255))

                # Apply master then channel LUTs
                ro = r_lut[m_lut[rv]] / 255.0
                go = g_lut[m_lut[gv]] / 255.0
                bo = b_lut[m_lut[bv]] / 255.0
                lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")

    return "\n".join(lines)
