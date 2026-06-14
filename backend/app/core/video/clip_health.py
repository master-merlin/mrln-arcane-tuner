"""Per-clip health validation for video-LoRA training.

Pure, I/O-free functions so they're trivially testable and reusable by a job
pre-flight later. The single source of truth is :data:`FAMILY_RULES`, a registry
of per-architecture constraints:

  * ``frame_rule``   — predicate ``(n: int) -> bool`` the (post-trim) frame count
                       must satisfy. WAN wants ``4k+1`` frames; LTX wants ``8k+1``.
  * ``native_fps``   — the fps the model was trained at (warn on mismatch).
                       ``None`` = no fps constraint.
  * ``dim_multiple`` — width and height must each be a multiple of this.
  * ``audio``        — whether the family consumes/needs an audio track. When
                       ``True`` and the clip has none, a warning is raised.

``compute_clip_warnings(meta)`` evaluates a single clip's metadata against every
family and returns ``{family: [warning, ...]}`` (empty list = healthy for that
family). The effective frame count is computed AFTER trim:
``round((eff_end - eff_start) * fps)`` where the effective window respects
``trim_start_s`` / ``trim_end_s`` when present (else the whole clip).

``summarize(items)`` rolls a list of clip metas into per-family counts +
the offending media_files, for the dataset health endpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FamilyRule:
    """Immutable per-family clip constraints."""

    __slots__ = ("frame_rule", "frame_rule_desc", "native_fps", "dim_multiple", "audio")

    def __init__(
        self,
        *,
        frame_rule: Callable[[int], bool],
        frame_rule_desc: str,
        native_fps: float | None,
        dim_multiple: int,
        audio: bool,
    ) -> None:
        self.frame_rule = frame_rule
        self.frame_rule_desc = frame_rule_desc
        self.native_fps = native_fps
        self.dim_multiple = dim_multiple
        self.audio = audio


FAMILY_RULES: dict[str, FamilyRule] = {
    "wan": FamilyRule(
        frame_rule=lambda n: n % 4 == 1,
        frame_rule_desc="4k+1",
        native_fps=16.0,
        dim_multiple=16,
        audio=False,
    ),
    "ltx": FamilyRule(
        frame_rule=lambda n: n % 8 == 1,
        frame_rule_desc="8k+1",
        native_fps=None,
        dim_multiple=32,
        audio=True,
    ),
}

# fps mismatch tolerance (frames-per-second) before warning.
_FPS_TOL = 0.5


def effective_window(meta: dict[str, Any]) -> tuple[float, float]:
    """Return the (start, end) seconds the clip trains on, honoring trim.

    ``trim_start_s`` defaults to 0; ``trim_end_s`` defaults to ``duration_s``.
    The window is clamped to ``[0, duration_s]`` and start<=end.
    """
    duration = float(meta.get("duration_s") or 0.0)
    start = meta.get("trim_start_s")
    end = meta.get("trim_end_s")
    start = float(start) if start is not None else 0.0
    end = float(end) if end is not None else duration
    start = max(0.0, start)
    if duration > 0:
        end = min(end, duration) if end > 0 else duration
        start = min(start, duration)
    if end < start:
        end = start
    return start, end


def effective_frame_count(meta: dict[str, Any]) -> int:
    """Post-trim frame count: ``round((eff_end - eff_start) * fps)``.

    Returns 0 when fps is unknown/zero (no frame-rule warning is then emitted,
    since we can't evaluate it).
    """
    fps = float(meta.get("fps") or 0.0)
    if fps <= 0:
        return 0
    start, end = effective_window(meta)
    span = max(0.0, end - start)
    return int(round(span * fps))


def compute_clip_warnings(meta: dict[str, Any]) -> dict[str, list[str]]:
    """Evaluate one clip against every family. Returns ``{family: [warnings]}``.

    An empty list for a family means the clip is healthy for that architecture.
    All families are always present in the result so callers can index directly.

    ``meta`` keys consulted: ``fps``, ``duration_s``, ``width``, ``height``,
    ``has_audio``, ``trim_start_s``, ``trim_end_s``. Missing keys are treated
    leniently (the corresponding check is skipped rather than failing).
    """
    fps = float(meta.get("fps") or 0.0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    has_audio = bool(meta.get("has_audio"))
    n = effective_frame_count(meta)

    result: dict[str, list[str]] = {}
    for family, rule in FAMILY_RULES.items():
        warnings: list[str] = []

        # Frame-count rule (only when we have a usable frame count).
        if n > 0 and not rule.frame_rule(n):
            warnings.append(
                f"frame count {n} violates {rule.frame_rule_desc} "
                f"(nearest valid: {_nearest_valid(n, rule)})"
            )

        # fps mismatch vs native.
        if rule.native_fps is not None and fps > 0:
            if abs(fps - rule.native_fps) > _FPS_TOL:
                warnings.append(f"fps {fps:g} differs from native {rule.native_fps:g}")

        # Dimension multiples.
        if width > 0 and width % rule.dim_multiple != 0:
            warnings.append(f"width {width} is not a multiple of {rule.dim_multiple}")
        if height > 0 and height % rule.dim_multiple != 0:
            warnings.append(f"height {height} is not a multiple of {rule.dim_multiple}")

        # Audio requirement (families that train with audio).
        if rule.audio and not has_audio:
            warnings.append("audio-training family but clip has no audio track")

        result[family] = warnings

    return result


def _nearest_valid(n: int, rule: FamilyRule) -> int:
    """Nearest frame count >= 1 satisfying the family's frame rule."""
    # frame rules here are k*m+1; find the multiple-base by probing.
    for delta in range(0, 16):
        for cand in (n - delta, n + delta):
            if cand >= 1 and rule.frame_rule(cand):
                return cand
    return n


def is_healthy(meta: dict[str, Any], family: str) -> bool:
    """Convenience: True if the clip has no warnings for *family*."""
    return not compute_clip_warnings(meta).get(family, [])


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up a list of clip metas into per-family health counts.

    Each item should carry its ``media_file`` (or ``rel_path``) plus the metadata
    keys :func:`compute_clip_warnings` consults. Returns::

        {
          "total": <int>,
          "families": {
            "wan": {"healthy": int, "warning": int,
                     "offenders": [{"media_file": str, "warnings": [...]}, ...]},
            "ltx": {...},
          },
        }
    """
    families: dict[str, dict[str, Any]] = {
        fam: {"healthy": 0, "warning": 0, "offenders": []} for fam in FAMILY_RULES
    }

    for item in items:
        media_file = (
            item.get("media_file") or item.get("rel_path") or item.get("path") or ""
        )
        per_family = compute_clip_warnings(item)
        for fam, warns in per_family.items():
            bucket = families[fam]
            if warns:
                bucket["warning"] += 1
                bucket["offenders"].append(
                    {"media_file": media_file, "warnings": warns}
                )
            else:
                bucket["healthy"] += 1

    return {"total": len(items), "families": families}
