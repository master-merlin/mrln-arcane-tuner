"""Video-training contract — one model-derived source of truth + run-config validation.

A video family declares its facts in two places that already exist:
  * capability flags (``is_video``/``has_audio``/``dual_expert``/``has_image_encoder``)
    via the family's ``capability_overrides`` (see :mod:`app.engine.core.archetypes`), and
  * numeric/behavioral facts under ``architecture_params`` (``video.frame_rule``,
    ``video.native_fps`` / ``video.frame_rate``, ``video.vae_spatial``,
    ``video.vae_temporal``, ``video.divisibility``, and ``mode``).

:func:`resolve_video_profile` projects both into a single :class:`VideoProfile`
— the authority every consumer reads.  :func:`validate_video_config` enforces the
governing principle: **derive everything we can from the model so an invalid
config cannot be expressed; hard-reject any residual invalid setting** (no silent
coercion that would mask a user mistake).

Pure logic (mirrors :mod:`app.engine.core.edit_validation`): a ``Report`` dataclass
+ a pure validate function, used both at config-assembly time (``job_manager``)
and defensively at trainer init (``pipeline_data``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.engine.components.bucketing import BucketManager

# fps mismatch tolerance (frames/sec) before a target_fps is rejected.
_FPS_TOL = 0.5


def frame_predicate(rule: str | None) -> Callable[[int], bool]:
    """Return a predicate ``(n: int) -> bool`` for an ``Nn+1`` frame rule.

    ``"4n+1"`` → ``n%4==1``, ``"8n+1"`` → ``n%8==1``, etc.  ``None`` / an
    unrecognized rule → always ``True`` (no constraint).  A single still
    (``n==1``) satisfies every ``Nn+1`` rule.  The ``Nn+1`` parsing lives once in
    :meth:`BucketManager._parse_frame_step` so bucketing and validation agree.
    """
    step = BucketManager._parse_frame_step(rule)
    if not step:
        return lambda n: True
    return lambda n: int(n) >= 1 and (int(n) - 1) % step == 0


@dataclass(frozen=True)
class VideoProfile:
    """Model-derived video facts — the single source of truth for a family."""

    is_video: bool
    mode: str | None  # "t2v" | "i2v" | "both" | None
    frame_rule: str | None  # "4n+1" | "8n+1" | None
    native_fps: float | None
    vae_spatial: int | None
    vae_temporal: int | None
    divisibility: int
    has_audio: bool
    has_image_encoder: bool
    dual_expert: bool

    def supports_i2v(self) -> bool:
        return self.mode in ("i2v", "both")

    def frame_ok(self, num_frames: int) -> bool:
        return frame_predicate(self.frame_rule)(num_frames)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve_video_profile(definition) -> VideoProfile:
    """Build the :class:`VideoProfile` from capability flags + architecture_params."""
    from app.engine.core.archetypes import resolve_capabilities

    caps = resolve_capabilities(definition)["capabilities"]
    arch = getattr(definition, "architecture_params", {}) or {}

    # fps key differs across families: LTX-2 → video.frame_rate, WAN → video.native_fps.
    native_fps_raw = arch.get("video.native_fps", arch.get("video.frame_rate"))
    native_fps = _to_float(native_fps_raw) if native_fps_raw is not None else None

    return VideoProfile(
        is_video=bool(caps.get("is_video", False)),
        mode=arch.get("mode"),
        frame_rule=arch.get("video.frame_rule") or None,
        native_fps=native_fps,
        vae_spatial=_int_or_none(arch.get("video.vae_spatial")),
        vae_temporal=_int_or_none(arch.get("video.vae_temporal")),
        divisibility=int(arch.get("video.divisibility", 32) or 32),
        has_audio=bool(caps.get("has_audio", False)),
        has_image_encoder=bool(caps.get("has_image_encoder", False)),
        dual_expert=bool(caps.get("dual_expert", False)),
    )


@dataclass
class VideoConfigReport:
    """Outcome of a video-config validation pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Model-owned settings to fold into the effective config (e.g. frame_rule).
    derived: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing blocks the run (warnings don't block)."""
        return not self.errors


def _truthy(value: Any) -> bool:
    return bool(value) and str(value).lower() not in ("false", "0", "")


def validate_video_config(definition, config: dict[str, Any]) -> VideoConfigReport:
    """Validate a training config against the model's video profile.

    Returns hard ``errors`` (any → block the run) and ``derived`` model-owned
    settings the caller should fold into the effective config (notably
    ``frame_rule`` — the gap that left temporal bucketing disengaged).
    """
    report = VideoConfigReport()
    profile = resolve_video_profile(definition)

    # Audio is only valid on audio-capable models (applies to image families too).
    if not profile.has_audio and _truthy(config.get("train_audio")):
        report.errors.append(
            "This model has no audio modality — turn off 'train_audio'."
        )

    if not profile.is_video:
        # Image model: video knobs are inert (the data path keeps stills at F=1);
        # nothing to derive or further validate.
        return report

    # ── Video model: fold the model-owned facts into the effective config ──
    if profile.frame_rule:
        report.derived["frame_rule"] = profile.frame_rule
    if profile.native_fps is not None:
        report.derived["video_native_fps"] = profile.native_fps
    report.derived["video_divisibility"] = profile.divisibility

    # num_frames must satisfy the family's Nn+1 rule (the UI offers only valid
    # values; a residual bad value — e.g. via direct API — is a hard error).
    num_frames = _int_or_none(config.get("num_frames")) or 0
    if num_frames and not profile.frame_ok(num_frames):
        report.errors.append(
            f"num_frames={num_frames} violates this model's frame rule "
            f"'{profile.frame_rule}'. Use a value of that form (1, "
            f"{_first_ladder_values(profile.frame_rule)} …)."
        )

    # target_fps: 0 means "use native"; a set value far from native is rejected.
    fps = _to_float(config.get("target_fps"))
    if fps and profile.native_fps and abs(fps - profile.native_fps) > _FPS_TOL:
        report.errors.append(
            f"target_fps={fps} does not match this model's native fps "
            f"{profile.native_fps}. Use 0 (native) or {profile.native_fps}."
        )

    # image-to-video only when the model supports it.
    if str(config.get("video_mode", "t2v")) == "i2v" and not profile.supports_i2v():
        report.errors.append(
            f"This model is text-to-video only (mode='{profile.mode or 't2v'}') "
            "— image-to-video is not supported."
        )

    return report


def _first_ladder_values(rule: str | None, n: int = 3) -> str:
    """A short human hint of the first few valid frame counts (e.g. '5, 9, 13')."""
    ladder = BucketManager.frame_ladder(BucketManager._default_max_frames(rule), rule)
    return ", ".join(str(f) for f in ladder[1 : n + 1]) or "1"
