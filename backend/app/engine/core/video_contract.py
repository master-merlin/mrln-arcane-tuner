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


def snap_frames(num_frames: int, rule: str | None) -> int:
    """Snap an arbitrary frame count DOWN to the nearest valid ``Nn+1`` value.

    A per-prompt preview frame count (``SamplePromptConfig.num_frames``) is NOT
    run through the video-contract validator, so it may violate the family's
    frame rule; snapping keeps the sampled latent grid legal. ``F=1`` satisfies
    every rule (still image). ``"8n+1"``: 30 → 25; ``"4n+1"``: 30 → 29. A
    ``None``/unrecognized rule imposes no constraint (returned unchanged, min 1).
    """
    n = max(int(num_frames), 1)
    step = BucketManager._parse_frame_step(rule)
    if not step:
        return n
    return ((n - 1) // step) * step + 1


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

    # Single-expert (high/low-noise-only) training needs a dual-expert MoE model.
    # ``both`` is the universal default; high/low on a single-transformer model
    # is rejected hard (rather than silently ignored) so the mistake is visible.
    expert_mode = str(config.get("expert_mode", "both") or "both").lower()
    if expert_mode not in ("both", "high", "low"):
        report.errors.append(
            f"expert_mode={expert_mode!r} is invalid — use 'both', 'high', or 'low'."
        )
    elif expert_mode != "both" and not profile.dual_expert:
        report.errors.append(
            "This model has a single transformer — 'expert_mode' must be 'both'. "
            "High/low-noise-only training requires a dual-expert (MoE) model."
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

    # model_shift timestep params — match the model's inference scheduler so
    # training-time timestep sampling reproduces the inference noise schedule.
    arch = getattr(definition, "architecture_params", {}) or {}
    use_dyn = arch.get("scheduler.use_dynamic_shifting")
    base_shift = arch.get("scheduler.base_shift")
    max_shift = arch.get("scheduler.max_shift")
    flow_shift = arch.get("scheduler.flow_shift")
    if use_dyn and base_shift is not None and max_shift is not None:
        report.derived["model_shift_base_shift"] = float(base_shift)
        report.derived["model_shift_max_shift"] = float(max_shift)
        report.derived["model_shift_base_seq"] = float(
            arch.get("scheduler.base_image_seq_len", 1024)
        )
        report.derived["model_shift_max_seq"] = float(
            arch.get("scheduler.max_image_seq_len", 4096)
        )
    elif flow_shift is not None:
        report.derived["model_shift_fixed"] = float(flow_shift)

    # num_frames must satisfy the family's Nn+1 rule (the UI offers only valid
    # values; a residual bad value — e.g. via direct API — is a hard error).
    num_frames = _int_or_none(config.get("num_frames")) or 0
    if num_frames and not profile.frame_ok(num_frames):
        report.errors.append(
            f"num_frames={num_frames} violates this model's frame rule "
            f"'{profile.frame_rule}'. Use a value of that form (1, "
            f"{_first_ladder_values(profile.frame_rule)} …)."
        )

    # Per-dataset frame overrides (DatasetItem.num_frames) must also satisfy the
    # Nn+1 rule. 0 = inherit the run-level num_frames (exempt); >0 is validated.
    for ds in config.get("datasets", []) or []:
        if not isinstance(ds, dict):
            continue
        ds_nf = _int_or_none(ds.get("num_frames")) or 0
        if ds_nf and not profile.frame_ok(ds_nf):
            report.errors.append(
                f"Dataset '{ds.get('dataset_name', '?')}' num_frames={ds_nf} "
                f"violates this model's frame rule '{profile.frame_rule}'. "
                "Use 0 (inherit) or a valid value."
            )

    # still_resolutions (Phase 3): F=1 stills in a video job bucket at their own
    # resolutions; empty/unset inherits `resolutions`. Only entry positivity is
    # validated here — BucketManager snaps to divisibility itself, and Task 1's
    # data-path consumer (resolve_still_resolutions) is the field's real reader.
    # _int_or_none (file convention) so malformed payloads ("abc") produce a
    # clean validation error instead of an uncaught ValueError.
    bad_still = [
        r
        for r in (config.get("still_resolutions") or [])
        if (_int_or_none(r) or 0) <= 0
    ]
    if bad_still:
        report.errors.append(
            f"still_resolutions entries must be positive ints, got {bad_still}."
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

    # ── Phase 1 temporal-sampling knobs ──
    coverage = str(config.get("temporal_coverage", "first") or "first")
    if coverage not in ("first", "tiled", "sliding"):
        report.errors.append(
            f"temporal_coverage={coverage!r} is invalid — use 'first', "
            "'tiled', or 'sliding'."
        )

    if coverage == "tiled":
        overlap = _to_float(config.get("window_overlap"))
        if not (0.0 <= overlap < 1.0):
            report.errors.append(
                f"window_overlap={overlap} must be in [0.0, 1.0)."
            )
        # Default to the schema default (10) when omitted so a config relying
        # on defaults validates; only an explicit bad value is rejected.
        max_w = _int_or_none(config.get("max_windows", 10))
        if max_w is None or max_w < 1:
            report.errors.append(
                f"max_windows={config.get('max_windows')} must be >= 1."
            )

    # Default to the schema default (1) when omitted (e.g. an older/partial
    # config the defensive validator must accept); reject an explicit < 1.
    stride = _int_or_none(config.get("frame_stride", 1))
    if stride is None or stride < 1:
        report.errors.append(
            f"frame_stride={config.get('frame_stride')} must be >= 1."
        )
    elif stride > 1 and _to_float(config.get("target_fps")) > 0.0:
        # Stride is the only fps lever — it divides the NATIVE rate. A manually
        # set target_fps combined with stride would compound (or contradict),
        # so reject it. Leave target_fps at 0/native when striding.
        report.errors.append(
            "frame_stride > 1 cannot be combined with a manually set "
            "target_fps — leave target_fps at 0 (native); stride sets the "
            "effective fps."
        )

    # Sliding-mode clip-length guard (0 = disabled).
    smcs = _to_float(config.get("sliding_max_clip_seconds"))
    if smcs < 0.0:
        report.errors.append(
            f"sliding_max_clip_seconds={config.get('sliding_max_clip_seconds')} "
            "must be >= 0."
        )

    return report


def _first_ladder_values(rule: str | None, n: int = 3) -> str:
    """A short human hint of the first few valid frame counts (e.g. '5, 9, 13')."""
    ladder = BucketManager.frame_ladder(BucketManager._default_max_frames(rule), rule)
    return ", ".join(str(f) for f in ladder[1 : n + 1]) or "1"
