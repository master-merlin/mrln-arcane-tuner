"""
Pipeline Data Mixin — dataset loading, bucketing, batch construction.
"""

import os
import random
from typing import Any

import httpx
import structlog
import torch
from PIL import Image
from torchvision import transforms

from app.engine.components.bucketing import BucketManager
from app.engine.components.latents import LatentManager

logger = structlog.get_logger(__name__)

# Constant "spatial" dims stamped on every audio inventory item — audio has no
# width/height, but `_iter_training_batches`' bucket key is
# (target_w, target_h, target_frames); a fixed dummy pair keeps that machinery
# working unmodified (bucket "volume" then varies with target_frames alone —
# the audio duration-window length in latent frames — so the VRAM-safe warmup
# still reserves the longest window first, exactly the video precedent).
AUDIO_DUMMY_DIM = 8


def _internal_api_headers() -> dict[str, str]:
    """Auth header for the trainer's own loopback API calls.

    The trainer runs as a separate subprocess with no browser cookie, so when
    a shared access token is configured (``MRLN_AUTH_TOKEN``) its calls to
    ``http://localhost/api/...`` would be rejected by ``TokenAuthMiddleware``
    (every ``/api`` path is gated). The subprocess inherits the token via the
    environment, so forward it as the ``X-Auth-Token`` header the middleware
    accepts. No token configured → no header, leaving local dev unchanged.
    """
    from app.core.container_config import auth_token

    token = auth_token()
    return {"X-Auth-Token": token} if token else {}


def video_trim_extra_key(item: dict) -> str:
    """Cache-filename discriminator for a clip's trim window ("" for images).

    Folded into the latent-cache hash so two trims of the same source clip
    (same spatial/temporal bucket, same cache dir) stay distinct. This MUST be
    identical between batch building (:meth:`PipelineDataMixin._build_batch`)
    and pre-caching (``PipelineCachingMixin._pre_cache_latents``) — otherwise a
    pre-cached video latent is written under one name and looked up under
    another, so it's silently re-encoded on-the-fly at train time (defeating
    the pre-cache, and risking the VAE-fallback OOM the pre-cache exists to
    avoid). Images return "" → byte-identical to the legacy image cache path.
    """
    if not item.get("is_video"):
        return ""
    ts = item.get("trim_start_s") or 0.0
    te = item.get("trim_end_s")
    key = f"t{ts}-{te}"
    if item.get("temporal_mode") == "sliding":
        # The sliding latent holds the FULL clip (cache_frames); a first-mode
        # latent of the same untrimmed clip holds only target_frames. Both hash
        # to "t0.0-None" without this discriminator → collision + wrong frame
        # count loaded. Fold the full-clip length in so they stay distinct.
        key += f"-slideF{int(item.get('cache_frames') or 0)}"
    return key


def resolve_still_resolutions(config: dict, is_video_family: bool) -> list[int]:
    """Effective spatial resolutions for F=1 stills.

    Phase-3 contract: empty/unset inherits `resolutions`. Image families
    ALWAYS inherit — the field is is_video-gated and the capability
    allowlist strips it at job create/update, but old DB rows may still
    carry it, and it must never change image-job bucketing.
    """
    base = list(config.get("resolutions") or [1024])
    if not is_video_family:
        return base
    still = [int(r) for r in (config.get("still_resolutions") or []) if int(r) > 0]
    return still or base


def _coerce_fps(value: Any) -> float:
    """Parse a config/metadata fps value to a non-negative float.

    Form-layer values can arrive as strings (e.g. the literal ``"0"``); a bare
    ``value or default`` then misfires because the non-empty string ``"0"`` is
    truthy, so it wins the ``or`` and ``float("0")`` collapses the result to
    0.0. Coerce explicitly: junk/``None`` → 0.0, and ``"0"`` means 0.0 (= unset
    → use native), never a truthy override that zeroes out the fps.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f > 0.0 else 0.0


def _resolve_clip_base_fps(
    config_target_fps: Any, clip_fps: Any, model_native_fps: Any
) -> float:
    """Base fps for a clip BEFORE Axis-B frame-stride division.

    Precedence: an explicit positive ``target_fps`` wins; else the clip's own
    probed fps; else the model's native fps — so a clip whose metadata is
    missing fps (e.g. a freshly split segment not yet re-probed) still gets a
    usable rate instead of 0, which crashes the clip loader with
    ``target_fps must be > 0``. All inputs are coerced so a stringified ``"0"``
    can't masquerade as a real override.
    """
    target = _coerce_fps(config_target_fps)
    if target > 0.0:
        return target
    clip = _coerce_fps(clip_fps)
    if clip > 0.0:
        return clip
    return _coerce_fps(model_native_fps)


class PipelineDataMixin:
    """Dataset preparation, inventory building, and batch construction."""

    # ── Prepare Data (shared) ────────────────────────────────────────────

    def _video_bucket_manager_for(self, max_frames: int):
        """Return a frame-ladder ``BucketManager`` capped at ``max_frames``.

        ``None`` when the model has no frame rule (no temporal bucketing). The
        cap of ``0`` falls back to the family default max. Results are cached
        per cap so a per-dataset override builds its ladder only once.
        """
        if not self._video_frame_rule:
            return None
        from app.engine.components.bucketing import BucketManager as _BM

        cap = int(max_frames) or _BM._default_max_frames(self._video_frame_rule)
        cache = self._video_bm_cache
        if cap not in cache:
            ladder = _BM.frame_ladder(cap, self._video_frame_rule)
            cache[cap] = BucketManager(
                base_resolutions=self._video_resolutions,
                frame_buckets=ladder,
            )
        return cache[cap]

    def _compute_tiled_windows(
        self,
        *,
        trim_start_s: float,
        end_s: float,
        window_span_s: float,
        overlap: float,
        max_windows: int,
    ) -> list[tuple[float, float]]:
        """Partition ``[trim_start_s, end_s]`` into up to ``max_windows`` windows.

        Each window covers ``window_span_s`` seconds — the time a clip needs to
        supply ``target_frames`` at the effective fps — and is stepped by
        ``(1 - overlap) * window_span_s``. Windows never extend past ``end_s``;
        a trailing partial shorter than ``window_span_s`` is dropped (it would
        not supply ``target_frames`` and would crash ``load_clip``). If the
        usable duration is shorter than a single window, returns ``[]`` and the
        caller falls back to the original (validated) single window. Trim bounds
        are rounded to 3 decimals so two runs hash to the same cache filename.

        Defensive bounds (the contract validator already rejects these upstream,
        but the helper stays safe if called directly): ``overlap >= 1`` is
        clamped so the step is at least 1 ms (never zero → no infinite loop),
        and ``max_windows <= 0`` is floored to 1.
        """
        usable = max(end_s - trim_start_s, 0.0)
        if window_span_s <= 0.0 or usable < window_span_s:
            # Cannot supply a full window → let the caller use the legacy window.
            return []

        step = max((1.0 - overlap) * window_span_s, 1e-3)
        windows: list[tuple[float, float]] = []
        start = trim_start_s
        while len(windows) < max(max_windows, 1):
            win_end = start + window_span_s
            if win_end > end_s + 1e-6:
                break
            windows.append((round(start, 3), round(win_end, 3)))
            start += step
        return windows

    def _effective_fps(self, native_or_target_fps: float) -> float:
        """Apply Axis-B frame stride: effective fps = fps / frame_stride.

        Divides the NATIVE (or native-equal target) fps — never a user-lowered
        target_fps (the contract rejects that combination). Stride is the only
        fps lever in Phase 1.
        """
        stride = int(self.config.get("frame_stride", 1) or 1)
        if stride <= 1 or native_or_target_fps <= 0.0:
            return float(native_or_target_fps)
        return float(native_or_target_fps) / float(stride)

    def _emit_temporal_items(
        self,
        *,
        base_item: dict,
        trim_start_s: float,
        end_s: float,
        window_span_s: float,
        repeats: int,
        full_clip_frames: int = 0,
    ) -> list[dict]:
        """Expand one clip×bucket into per-window inventory items × repeats.

        ``first`` (default) → one item per repeat with the original trim window.
        ``tiled`` → K full windows (``_compute_tiled_windows``), each repeated
        ``repeats`` times (K×repeats, so each window keeps the clip's repeat
        weighting). A clip too short for one window falls back to the single
        original window.
        ``sliding`` → ONE full-clip item per repeat, flagged
        ``temporal_mode="sliding"`` and carrying ``cache_frames`` (the full-clip
        ladder count); the train loop slices a random per-step window from the
        cached full-clip latent. A clip with no slide room
        (``full_clip_frames <= target_frames``) falls back to ``first``; a clip
        longer than ``sliding_max_clip_seconds`` falls back to ``tiled`` (logged)
        — one full-clip latent would be large and the causal-slice risk grows.

        The first/tiled fallback clones ``base_item`` VERBATIM — it does NOT
        overwrite ``trim_start_s``/``trim_end_s`` with the computed ``end_s``.
        This keeps the trim-cache key byte-identical to the pre-tiling path: an
        untrimmed clip carries ``trim_end_s = None`` and must keep it
        (``video_trim_extra_key`` formats ``t{start}-{end}``, so writing a
        concrete end where ``None`` was would silently re-key the latent cache).
        Only genuine tiled sub-windows overwrite the trim bounds.
        """
        coverage = getattr(self, "_temporal_coverage", "first")
        is_video = bool(base_item.get("is_video"))

        # ── sliding: one FULL-CLIP item per repeat (sliced per-step at train) ──
        if is_video and coverage == "sliding":
            usable = max(end_s - trim_start_s, 0.0)
            max_secs = float(getattr(self, "_sliding_max_clip_seconds", 0.0) or 0.0)
            has_room = int(full_clip_frames) > int(base_item.get("target_frames", 1))
            if max_secs > 0.0 and usable > max_secs:
                # Over-long clip: a single full-clip latent is large and the
                # causal-slice risk grows — tile the clip instead (full
                # coverage, bounded per-window). Never silently cap.
                self.logger.info(
                    "sliding_clip_too_long_tiling",
                    id=base_item.get("id"),
                    duration_s=round(usable, 1),
                    max_clip_seconds=max_secs,
                )
                coverage = "tiled"
            elif not has_room:
                # Clip barely longer than one window → no slide room → first.
                coverage = "first"
            else:
                return [
                    {
                        **base_item,
                        "temporal_mode": "sliding",
                        "cache_frames": int(full_clip_frames),
                    }
                    for _ in range(repeats)
                ]

        windows: list[tuple[float, float]] = []
        if is_video and coverage == "tiled":
            windows = self._compute_tiled_windows(
                trim_start_s=trim_start_s,
                end_s=end_s,
                window_span_s=window_span_s,
                overlap=getattr(self, "_window_overlap", 0.0),
                max_windows=getattr(self, "_max_windows", 10),
            )

        out: list[dict] = []
        if not windows:
            # first mode, non-video, or clip shorter than one window: preserve
            # the base item's ORIGINAL trim window verbatim (including a None
            # end) so the cache key matches the pre-tiling path exactly.
            for _ in range(repeats):
                out.append(dict(base_item))
            return out

        for w_start, w_end in windows:
            for _ in range(repeats):
                clone = dict(base_item)
                clone["trim_start_s"] = w_start
                clone["trim_end_s"] = w_end
                out.append(clone)
        return out

    def _append_audio_item(
        self,
        inventory: list[dict[str, Any]],
        *,
        img_path: str,
        img_rel: str,
        caption: str,
        meta: dict,
        lyrics_content: str,
        ds_path: str,
        model_name: str,
        ds_version: str,
        prefix: str,
        repeats: int,
        ds_config: dict,
        ds_use_captions: bool,
        ds_use_model_aware: bool,
    ) -> None:
        """Build + append ``repeats`` audio inventory items for one pair.

        Cache-keys on a duration-window res_str ``"{sample_rate}Hz-{window_s}s"``
        (the audio analogue of video's ``WxHxNfFPS``) — rounded to a coarse
        bucket (:func:`round_duration_bucket`) so same-length clips share a
        cache dir / batch bucket. ``target_frames`` carries the window's LATENT
        frame count (not a pixel frame count) purely so the existing
        ``(target_w, target_h, target_frames)`` bucket-key machinery groups
        same-duration items together for ``train_batch_size > 1``.
        """
        from app.engine.components.audio import round_duration_bucket
        from app.engine.components.latents import LatentManager

        duration_s = float(meta.get("duration_s") or 0.0)
        sample_rate = (
            int(meta.get("sample_rate") or 0) or self._audio_target_sample_rate
        )
        channels = int(meta.get("channels") or 0) or self._audio_target_channels

        window_s = round_duration_bucket(duration_s, self._audio_duration_cap)
        target_frames = max(round(window_s * self._audio_latent_hz), 1)
        res_str = f"{self._audio_target_sample_rate}Hz-{window_s:g}s"
        cache_dir = LatentManager.resolve_cache_dir(
            ds_path, model_name, ds_version, res_str, "original"
        )

        item: dict[str, Any] = {
            "path": img_path,
            "id": img_rel,
            "caption": caption,
            "dataset_path": ds_path,
            "prefix": prefix,
            "dropout_rate": float(ds_config.get("caption_dropout_rate", 0.0)),
            "use_captions": ds_use_captions,
            "use_model_aware_captions": ds_use_model_aware,
            "orig_w": 0,
            "orig_h": 0,
            "target_w": AUDIO_DUMMY_DIM,
            "target_h": AUDIO_DUMMY_DIM,
            "target_frames": target_frames,
            "cache_dir": cache_dir,
            "variant": "original",
            "is_video": False,
            "is_audio": True,
            "has_masked": False,
            "duration_s": duration_s,
            "window_s": window_s,
            "source_sample_rate": sample_rate,
            "source_channels": channels,
            "lyrics_content": lyrics_content,
        }
        for _ in range(repeats):
            inventory.append(item)

    async def prepare_data(self):
        """Fetch datasets via API, build inventory with aspect-ratio bucketing."""
        self.logger.info("preparing_data")

        resolutions = self.config.get("resolutions", [1024])
        self.bucket_manager = BucketManager(base_resolutions=resolutions)

        # Stills mixed into a video job bucket at their OWN resolutions
        # (Phase 3). Empty/unset still_resolutions inherits `resolutions`, in
        # which case we reuse the same manager object → byte-identical to the
        # pre-Phase-3 single-manager behavior. Image families always inherit.
        still_res = resolve_still_resolutions(self.config, self.is_video_family)
        self.still_bucket_manager = (
            self.bucket_manager
            if still_res == resolutions
            else BucketManager(base_resolutions=still_res)
        )

        # ── Video contract (defensive) ──
        # Config assembly (job_manager) already derives frame_rule + validates,
        # but a job created before the contract existed — or via a direct API
        # call — may bypass it. Re-derive the model-owned settings and re-reject
        # illegal combos here so the trainer is self-defending.
        from app.engine.core.video_contract import validate_video_config

        _vc = validate_video_config(self.definition, self.config)
        for _k, _v in _vc.derived.items():
            self.config.setdefault(_k, _v)
        for _w in _vc.warnings:
            self.logger.warning("video_config_warning", message=_w)
        if not _vc.ok:
            raise ValueError("Video configuration invalid: " + "; ".join(_vc.errors))

        # ── Video bucketing config (fields land fully in phase B6) ──
        # Read defensively: config may not carry video knobs yet. When a frame
        # rule is present we build a temporal-aware BucketManager; otherwise
        # video items fall back to a single requested frame count.
        self._video_frame_rule = self.config.get("frame_rule") or None
        # Default target fps: native (0/unset → use clip's own fps later).
        # Coerced because the form layer can submit this as the string "0",
        # which is truthy and would otherwise win the `or` and zero the fps.
        self._video_target_fps = _coerce_fps(self.config.get("target_fps")) or None
        # Model's native fps (contract-derived) — fallback when a clip's own
        # probed fps is missing (e.g. a freshly split segment not yet re-probed).
        self._model_native_fps = _coerce_fps(self.config.get("video_native_fps"))
        # Max frames a run is willing to train on (caps the frame ladder).
        # This is the GENERAL setting; a dataset may override it (see below).
        self._video_num_frames = int(self.config.get("num_frames", 0) or 0)
        self._video_resolutions = resolutions
        # ── Phase 1 temporal-sampling knobs (read defensively) ──
        self._temporal_coverage = str(
            self.config.get("temporal_coverage", "first") or "first"
        )
        self._window_overlap = float(self.config.get("window_overlap", 0.0) or 0.0)
        self._max_windows = int(self.config.get("max_windows", 10) or 10)
        self._frame_stride = int(self.config.get("frame_stride", 1) or 1)
        self._sliding_max_clip_seconds = float(
            self.config.get("sliding_max_clip_seconds", 0.0) or 0.0
        )
        # Cache of frame-ladder BucketManagers keyed by max-frames cap, so a
        # per-dataset override builds its ladder once and is reused.
        self._video_bm_cache: dict[int, Any] = {}
        self._video_bucket_manager = self._video_bucket_manager_for(
            self._video_num_frames
        )
        # Full-clip ladder snap for sliding: capped at the FAMILY ceiling
        # (81/121), NOT the run's num_frames — otherwise cache_frames collapses
        # to the per-step window and sliding gains nothing. cap=0 → family max.
        self._sliding_full_bm = (
            self._video_bucket_manager_for(0)
            if self._temporal_coverage == "sliding"
            else None
        )

        # ── Audio bucketing config (ace_step15 and future audio families) ──
        # Read defensively — a non-audio family never sets these architecture
        # params, so the defaults are inert (no audio items ever reach
        # ``_append_audio_item`` on an image/video run).
        arch_params = getattr(self.definition, "architecture_params", {}) or {}
        self._audio_target_sample_rate = int(
            arch_params.get("audio.sample_rate", 48000)
        )
        self._audio_target_channels = int(arch_params.get("audio.channels", 2))
        self._audio_latent_hz = float(arch_params.get("audio.latent_hz", 25.0))
        self._audio_duration_cap = float(self.config.get("duration_s", 30.0) or 30.0)

        datasets_config = self.config.get("datasets", [])
        inventory: list[dict[str, Any]] = []

        # ── Paired edit-model config ──
        # control_inputs > 0 → train on the pair's effective TARGET while
        # conditioning on its effective CONTROLS (resolved roles from /pairs).
        control_inputs = int(getattr(self.definition, "control_inputs", 0) or 0)
        is_edit_run = control_inputs > 0
        control_resolution = int(self.config.get("control_resolution", 0) or 0)
        dataset_kinds: dict[str, str] = {}
        edit_candidates = 0
        edit_skipped = 0

        # ── Global augmentation config ──
        self._aug_h_flip = bool(self.config.get("h_flip", False))
        self._aug_v_flip = bool(self.config.get("v_flip", False))

        self.logger.info(
            "augmentation_config",
            h_flip=self._aug_h_flip,
            v_flip=self._aug_v_flip,
        )

        # Resolve backend port
        port = 8000
        try:
            from app.core.settings_manager import get_settings_manager

            sm = get_settings_manager()
            app_settings = sm.get_module_settings("application")
            if app_settings:
                port = app_settings.get("backend_port", 8000)
        except (ImportError, AttributeError, KeyError):
            pass

        api_url = f"http://localhost:{port}/api"
        model_name = self.definition.id.split("/")[-1]

        async with httpx.AsyncClient(
            timeout=60.0,
            headers=_internal_api_headers(),
        ) as client:
            for ds_config in datasets_config:
                if isinstance(ds_config, str):
                    ds_config = {"dataset_name": ds_config}

                name = ds_config.get("dataset_name")
                repeats = int(ds_config.get("num_repeats", 1))
                prefix = ds_config.get("caption_prefix", "")
                ignore_filter = ds_config.get("ignore_filter", False)

                # Per-dataset frame override: 0 = inherit the run's general
                # num_frames; >0 caps THIS dataset's clips at its own ladder.
                # (Images in any dataset stay at F=1 regardless.)
                ds_num_frames = int(ds_config.get("num_frames", 0) or 0)
                ds_effective_frames = (
                    ds_num_frames if ds_num_frames > 0 else self._video_num_frames
                )
                ds_bucket_manager = (
                    self._video_bucket_manager_for(ds_effective_frames)
                    if ds_num_frames > 0
                    else self._video_bucket_manager
                )

                # Per-dataset masking config
                ds_masking_enabled = bool(ds_config.get("masking_enabled", False))
                ds_original_weight = max(
                    float(ds_config.get("original_weight", 0.70)), 0.50
                )

                # Per-dataset caption flags (defaults preserve old-config behavior)
                ds_use_captions = bool(ds_config.get("use_captions", True))
                ds_use_model_aware = bool(
                    ds_config.get("use_model_aware_captions", True)
                )

                try:
                    resp = await client.get(f"{api_url}/datasets/{name}/pairs")
                    if resp.status_code != 200:
                        self.logger.warning(
                            "dataset_fetch_failed", name=name, status=resp.status_code
                        )
                        continue

                    pairs = resp.json()
                    ds_resp = await client.get(f"{api_url}/datasets/{name}")
                    ds_data = ds_resp.json()
                    ds_path = ds_data.get("path")
                    ds_version = ds_data.get("version", "1.0.0")
                    dataset_kinds[name] = ds_data.get("kind", "standard")

                    # Recreate masked images if requested
                    if ds_masking_enabled and bool(
                        ds_config.get("recreate_masks", False)
                    ):
                        mask_opacity = float(ds_config.get("mask_opacity", 0.0))
                        self.logger.info(
                            "recreating_masks",
                            dataset=name,
                            opacity=mask_opacity,
                        )
                        from app.core.masking.masking_service import MaskingService

                        masking_svc = MaskingService()
                        result = masking_svc.mass_apply(
                            ds_path,
                            mask_opacity,
                            overwrite=True,
                        )
                        self.logger.info(
                            "masks_recreated",
                            dataset=name,
                            applied=result["applied"],
                            skipped=result["skipped"],
                        )

                    for pair in pairs:
                        media_rel = pair.get("media_file")
                        if not media_rel:
                            continue
                        if not ignore_filter:
                            meta = pair.get("metadata") or {}
                            if meta.get("enabled") is False:
                                continue

                        # Edit runs train the pair's effective TARGET (role
                        # ordering may point a control slot at it). Standard
                        # runs (and standard models on edit datasets) train the
                        # root media file — controls are ignored.
                        if is_edit_run:
                            edit_candidates += 1
                            img_rel = pair.get("effective_target") or media_rel
                            # Partial pair (fewer controls than the model needs):
                            # skip + count. Threshold-checked after the loop.
                            if (
                                len(pair.get("effective_controls") or [])
                                < control_inputs
                            ):
                                edit_skipped += 1
                                self.logger.warning(
                                    "edit_pair_incomplete",
                                    dataset=name,
                                    media=media_rel,
                                    have=len(pair.get("effective_controls") or []),
                                    need=control_inputs,
                                )
                                continue
                        else:
                            img_rel = media_rel

                        img_path = os.path.join(ds_path, img_rel)
                        caption = pair.get("caption_content", "")

                        meta = pair.get("metadata", {})
                        # meta dims describe the ROOT media; when a role flip
                        # points the target at a control slot, read the actual
                        # target file's dims instead.
                        if is_edit_run and img_rel != media_rel:
                            w = h = None
                        else:
                            w, h = meta.get("width"), meta.get("height")
                        # ── Video item detection + temporal bucketing ──
                        # ``meta`` (from /pairs) carries phase-A1 video fields.
                        is_video = bool(meta.get("is_video"))
                        vid_fps = vid_trim_start = vid_trim_end = None
                        vid_target_frames = 1
                        vid_target_fps = None
                        if is_video:
                            duration_s = float(meta.get("duration_s") or 0.0)
                            vid_trim_start = float(meta.get("trim_start_s") or 0.0)
                            vid_trim_end = meta.get("trim_end_s")
                            vid_trim_end = (
                                float(vid_trim_end)
                                if vid_trim_end is not None
                                else None
                            )
                            # Effective usable (trimmed) duration in seconds.
                            end_s = (
                                vid_trim_end if vid_trim_end is not None else duration_s
                            )
                            eff_dur = max(end_s - vid_trim_start, 0.0)
                            # Target fps: config override → native clip fps,
                            # then divided by Axis-B frame stride so the model
                            # is told the EFFECTIVE (sampled) rate. available_
                            # frames (bucket selection) and res_str use the same
                            # effective rate, so a strided run buckets and caches
                            # by what it actually samples.
                            # Positive target_fps → clip's own fps → model
                            # native. Coerced so a stringified "0" can't pose as
                            # a real override and zero out the effective rate.
                            _base_fps = _resolve_clip_base_fps(
                                self.config.get("target_fps"),
                                meta.get("fps"),
                                self._model_native_fps,
                            )
                            vid_target_fps = self._effective_fps(_base_fps)
                            available_frames = (
                                int(eff_dur * vid_target_fps)
                                if vid_target_fps > 0
                                else 0
                            )
                            vid_fps = vid_target_fps

                        # ── Audio item detection + duration-window bucketing ──
                        # Audio has no spatial dims — a dummy constant width/
                        # height keeps it flowing through the SAME bucket-key
                        # machinery `_iter_training_batches` already uses
                        # (target_w, target_h, target_frames), with
                        # target_frames repurposed as the item's LATENT
                        # duration-window length (the audio analogue of a
                        # video frame count). Built + appended here and the
                        # pair skips every spatial/image code path below.
                        is_audio = bool(meta.get("is_audio"))
                        if is_audio:
                            self._append_audio_item(
                                inventory,
                                img_path=img_path,
                                img_rel=img_rel,
                                caption=caption,
                                meta=meta,
                                lyrics_content=pair.get("lyrics_content", "") or "",
                                ds_path=ds_path,
                                model_name=model_name,
                                ds_version=ds_version,
                                prefix=prefix,
                                repeats=repeats,
                                ds_config=ds_config,
                                ds_use_captions=ds_use_captions,
                                ds_use_model_aware=ds_use_model_aware,
                            )
                            continue

                        if not w and not is_video:
                            try:
                                with Image.open(img_path) as img:
                                    w, h = img.size
                            except (OSError, ValueError):
                                continue
                        if not w and is_video:
                            # Video without cached dims: fall back to meta or skip.
                            w, h = meta.get("width"), meta.get("height")
                            if not w:
                                continue

                        # ── Masked variant info ──
                        stem = os.path.splitext(os.path.basename(img_rel))[0]
                        masked_img_path = os.path.join(ds_path, "masked", f"{stem}.jpg")
                        has_masked = ds_masking_enabled and os.path.isfile(
                            masked_img_path
                        )

                        # Masked caption: read from masked/ alongside masked image
                        masked_caption = None
                        has_masked_caption = False
                        if has_masked:
                            masked_cap_path = os.path.join(
                                ds_path, "masked", f"{stem}.txt"
                            )
                            if os.path.isfile(masked_cap_path):
                                try:
                                    with open(
                                        masked_cap_path, "r", encoding="utf-8"
                                    ) as f:
                                        masked_caption = f.read().strip()
                                        has_masked_caption = True
                                except OSError:
                                    pass

                        bucketing_mode = self.config.get("bucketing_mode", "kohya")
                        if is_video:
                            # Temporal bucket: pick the largest frame bucket that
                            # the trimmed clip can supply, capped at this dataset's
                            # effective frame count (override or inherited). Falls
                            # back to a single requested frame count when no frame
                            # rule is set.
                            if ds_bucket_manager is not None:
                                vbucket = ds_bucket_manager.get_bucket_for_video(
                                    w,
                                    h,
                                    available_frames,
                                )
                                buckets = [vbucket]
                                vid_target_frames = vbucket["frames"]
                            else:
                                sbucket = self.bucket_manager.get_bucket(w, h)
                                vid_target_frames = max(
                                    min(
                                        ds_effective_frames or 1,
                                        available_frames or 1,
                                    ),
                                    1,
                                )
                                buckets = [{**sbucket, "frames": vid_target_frames}]
                        elif bucketing_mode == "multi":
                            buckets = self.still_bucket_manager.get_buckets_for_all_resolutions(
                                w, h
                            )
                        else:
                            buckets = [self.still_bucket_manager.get_bucket(w, h)]

                        for bucket in buckets:
                            target_w, target_h = bucket["width"], bucket["height"]
                            if is_video:
                                tgt_f = bucket.get("frames", vid_target_frames)
                                # Cache resolution string carries the temporal
                                # slice (F + fps) so image/video latents never
                                # collide. The per-clip trim window is folded
                                # into the FILENAME hash (extra_key) so two
                                # trims of the same source file under the same
                                # spatial/temporal bucket stay distinct even
                                # though they share a cache directory.
                                res_str = f"{target_w}x{target_h}x{tgt_f}f{vid_fps}"
                            else:
                                tgt_f = 1
                                res_str = f"{target_w}x{target_h}"
                            cache_dir = LatentManager.resolve_cache_dir(
                                ds_path, model_name, ds_version, res_str, "original"
                            )
                            item: dict[str, Any] = {
                                "path": img_path,
                                "id": img_rel,
                                "caption": caption,
                                "dataset_path": ds_path,
                                "prefix": prefix,
                                "dropout_rate": float(
                                    ds_config.get("caption_dropout_rate", 0.0)
                                ),
                                "use_captions": ds_use_captions,
                                "use_model_aware_captions": ds_use_model_aware,
                                "orig_w": w,
                                "orig_h": h,
                                "target_w": target_w,
                                "target_h": target_h,
                                "cache_dir": cache_dir,
                                "variant": "original",
                                "is_video": is_video,
                            }

                            # ── Video temporal fields ──
                            if is_video:
                                item["target_frames"] = tgt_f
                                item["target_fps"] = vid_target_fps
                                item["trim_start_s"] = vid_trim_start
                                item["trim_end_s"] = vid_trim_end

                            # Attach masked variant data for runtime selection
                            if has_masked:
                                masked_cache_dir = LatentManager.resolve_cache_dir(
                                    ds_path, model_name, ds_version, res_str, "masked"
                                )
                                item["masked_path"] = masked_img_path
                                item["masked_caption"] = masked_caption
                                item["has_masked_caption"] = has_masked_caption
                                item["masked_cache_dir"] = masked_cache_dir
                                item["has_masked"] = True
                                item["original_weight"] = ds_original_weight
                            else:
                                item["has_masked"] = False

                            # ── Paired control variants (edit runs) ──
                            if is_edit_run:
                                from app.engine.core.pipeline.edit_inventory import (
                                    build_control_fields,
                                )

                                def _cache_dir_for(
                                    rstr, variant, _p=ds_path, _v=ds_version
                                ):
                                    return LatentManager.resolve_cache_dir(
                                        _p, model_name, _v, rstr, variant
                                    )

                                def _bucket_for(bw, bh, base):
                                    return BucketManager(
                                        base_resolutions=[base]
                                    ).get_bucket(bw, bh)

                                control_fields = build_control_fields(
                                    pair.get("effective_controls") or [],
                                    ds_path,
                                    control_inputs,
                                    target_w,
                                    target_h,
                                    control_resolution,
                                    _cache_dir_for,
                                    _bucket_for,
                                )
                                if control_fields is None:
                                    # Partial pair — skip this bucket item.
                                    continue
                                item.update(control_fields)

                            if is_video:
                                # Seconds a single window must cover to supply
                                # tgt_f frames at the effective fps.
                                _span = (
                                    tgt_f / vid_target_fps
                                    if vid_target_fps > 0
                                    else 0.0
                                )
                                # Pass the FULL usable end (``end_s`` above is
                                # duration-aware — it uses the clip's duration_s
                                # for an untrimmed clip) so tiled mode enumerates
                                # windows across the WHOLE clip. A one-window-wide
                                # end here would silently collapse tiling to a
                                # single window for every untrimmed clip. first /
                                # fallback mode ignores end_s and clones the
                                # original trim window verbatim.
                                full_clip_frames = 0
                                if (
                                    self._temporal_coverage == "sliding"
                                    and self._sliding_full_bm is not None
                                ):
                                    full_clip_frames = (
                                        self._sliding_full_bm.frame_bucket_for(
                                            available_frames
                                        )
                                    )
                                inventory.extend(
                                    self._emit_temporal_items(
                                        base_item=item,
                                        trim_start_s=vid_trim_start,
                                        end_s=end_s,
                                        window_span_s=_span,
                                        repeats=repeats,
                                        full_clip_frames=full_clip_frames,
                                    )
                                )
                            else:
                                for _ in range(repeats):
                                    inventory.append(item)

                except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
                    self.logger.error("dataset_api_error", name=name, error=str(e))
                    continue

        # ── Run-start validation for edit/standard × dataset-kind ──
        from app.engine.core.edit_validation import validate_edit_config

        report = validate_edit_config(
            self.definition,
            self.config,
            dataset_kinds.get,
        )
        for warning in report.warnings:
            self.logger.warning("edit_config_warning", message=warning)
        if not report.ok:
            raise ValueError(
                "Edit-model configuration invalid: " + "; ".join(report.errors)
            )

        # Partial-pair guard: a few missing controls are skipped, but if most
        # pairs are incomplete the dataset layout is almost certainly wrong.
        if is_edit_run and edit_candidates > 0:
            self.logger.info(
                "edit_pair_coverage",
                candidates=edit_candidates,
                skipped_incomplete=edit_skipped,
                control_inputs=control_inputs,
            )
            if edit_skipped / edit_candidates > 0.5:
                raise ValueError(
                    f"{edit_skipped}/{edit_candidates} pairs are missing control "
                    "images — check the dataset's control/ folders. Refusing to "
                    "train on a mostly-unpaired edit dataset."
                )

        self.inventory = inventory
        if not inventory:
            raise ValueError("No training data found in datasets.")

        # Log masked coverage stats
        masked_count = sum(1 for i in inventory if i.get("has_masked"))
        if masked_count > 0:
            self.logger.info(
                "masked_training_coverage",
                total_items=len(inventory),
                items_with_masked=masked_count,
                items_without_masked=len(inventory) - masked_count,
            )

        self.bucket_manager.log_distribution()
        self.logger.info("data_prepared", total_items=len(inventory))

        # Initialize LatentManager early — needed by _validate_latent_cache()
        # and _pre_cache_latents() which run before prepare_for_training().
        vae = self.components.get("vae")
        self.latent_manager = LatentManager(
            vae,
            device=self.device,
            arch_params=getattr(self.definition, "architecture_params", None),
        )

    # ── Batch Construction ───────────────────────────────────────────────

    def _select_variant(self, item: dict) -> tuple[str, str, str]:
        """Pick image variant for this training step.

        Returns ``(path, caption, cache_dir)`` for the selected variant.
        When the item has a masked version (per-dataset masking),
        weighted random selection is applied using the item's own weight.

        Caption handling for masked variants:
        - Masked caption resolves via select_training_caption(masked=True):
          masked variant → masked/{stem}.txt → original caption.
        """
        from app.engine.core.pipeline.caption_selection import select_training_caption

        _def_id = getattr(getattr(self, "definition", None), "id", None)

        # Masked variant: weighted random selection using the item's own weight.
        if item.get("has_masked") and random.random() >= item.get(
            "original_weight", 0.70
        ):
            # Route masked through the same per-definition resolver as the general
            # branch. Defensive: no masked variant → masked_caption → original.
            cap = select_training_caption(item, _def_id, masked=True)
            return (item["masked_path"], cap, item["masked_cache_dir"])

        # General (non-masked) caption: per-definition variant overrides item["caption"].
        cap = select_training_caption(item, _def_id)
        return item["path"], cap, item["cache_dir"]

    def _get_batch(
        self, items: list[dict], decode_pixels: bool = True
    ) -> dict[str, Any]:
        """Build a training batch from inventory items.

        When augmentation is configured, applies per-sample variant
        selection (original / masked).  Flip augmentation is applied
        later to the latent tensor in the training loop.

        ``decode_pixels`` gates whether source pixels are decoded into
        ``batch["images"]``. A video clip's PyAV decode + resize is the heaviest
        per-step work, yet with a warm latent cache those pixels are immediately
        discarded for the cached latent — pure waste that starves the GPU. The
        train loop passes ``decode_pixels=False`` for video families and
        re-decodes on demand (:meth:`_decode_batch_images`) only on the rare
        cache miss, leaving ``batch["images"]`` ``None`` otherwise. Image/pixel
        families decode upfront (the default) so they stay byte-identical.
        """
        captions: list[str] = []
        ids: list[str] = []
        cache_dirs: list[str] = []
        batch_paths: list[str] = []
        extra_keys: list[str] = []

        trigger = self.config.get("global_triggerword", "")
        persist_trigger = bool(self.config.get("persist_triggerword_on_dropout", False))
        transform_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        is_video_family = bool(getattr(self, "is_video_family", False))

        # Pixels are only needed to ENCODE a latent; when the caller defers the
        # decode (warm cache, video families) we still build every other field
        # and leave images unbuilt — the train loop re-decodes only on a miss.
        images: list[torch.Tensor] | None = [] if decode_pixels else None

        for item in items:
            # ── Variant selection ──
            img_path, cap, cache_dir = self._select_variant(item)

            if images is not None:
                # Decode now (flip applied later on latents, matching the image
                # path's latent-space augmentation).
                images.append(
                    self._decode_item_image(
                        item, img_path, transform_norm, is_video_family
                    )
                )
            # Trim window folds into the cache-file hash (shared helper so
            # pre-caching computes the identical key); "" for images.
            extra_keys.append(video_trim_extra_key(item))

            # Caption construction
            is_dropped = random.random() < item["dropout_rate"]
            if is_dropped:
                cap = ""
            # Expand [triggerword] wildcard inline in caption text
            trigger_used_inline = "[triggerword]" in cap
            if trigger and trigger_used_inline:
                cap = cap.replace("[triggerword]", trigger)
            parts: list[str] = []
            # Triggerword: prepend only if NOT used inline (avoid duplication).
            # Always include if persist_triggerword_on_dropout,
            # otherwise only when caption is NOT dropped.
            if (
                trigger
                and not trigger_used_inline
                and (not is_dropped or persist_trigger)
            ):
                parts.append(trigger)
            if item["prefix"]:
                parts.append(item["prefix"])
            if cap:
                parts.append(cap)
            cap = ", ".join(parts)
            captions.append(cap)

            ids.append(item["id"])
            cache_dirs.append(cache_dir)
            batch_paths.append(img_path)

        batch = {
            "images": (
                torch.stack(images).to(self.device) if images is not None else None
            ),
            "captions": captions,
            "ids": ids,
            "cache_dirs": cache_dirs,
            "paths": batch_paths,
        }
        # Per-item cache discriminators (video trim window). Only attached when
        # at least one item carries one, so image batches are untouched.
        if any(extra_keys):
            batch["extra_keys"] = extra_keys

        # Effective fps for the model's frame-rate / RoPE conditioning. Items in
        # a batch share one temporal bucket (and thus one target_fps), so a
        # single scalar suffices. Only set for video items so image batches are
        # byte-identical to before.
        if items and items[0].get("is_video"):
            batch["target_fps"] = float(items[0].get("target_fps") or 0.0)

        # Lyrics sidecar text, one per item (audio families only — the
        # `<stem>.lyrics.txt` content the C0 dataset layer hydrates into
        # `lyrics_content`). Empty string when an audio item has no lyrics
        # sidecar; absent entirely for non-audio batches.
        if items and items[0].get("is_audio"):
            batch["lyrics"] = [it.get("lyrics_content", "") or "" for it in items]

        # ── Paired control images (edit runs) ──
        # All items in an edit batch carry the same control slot count
        # (partial pairs were skipped at inventory time), so we transpose into
        # per-slot stacked tensors the family forward can concat with the
        # target latents. Controls are loaded clean here and never flipped.
        if items and items[0].get("control_paths"):
            self._attach_control_images(batch, items, transform_norm)

        # Let families add extra data (e.g. SDXL time_ids)
        batch.update(self.build_batch_extra(items))
        return batch

    def _decode_item_image(
        self,
        item: dict,
        img_path: str,
        transform_norm,
        is_video_family: bool,
    ) -> torch.Tensor:
        """Decode one inventory item's source pixels to a training tensor.

        Video → ``[C, F, H, W]`` via the clip loader. Stills → ``[C, H, W]``
        (smart-resized + center-cropped + normalized), lifted to ``[C, 1, H, W]``
        in a video-family run so the whole batch collates 5D. Consumes no RNG,
        so deferring/repeating the decode never perturbs training determinism.
        """
        tw, th = item["target_w"], item["target_h"]
        if item.get("is_audio"):
            from app.engine.components.audio import AudioClipLoader

            return AudioClipLoader().load_clip(
                img_path,
                target_sample_rate=self._audio_target_sample_rate,
                target_channels=self._audio_target_channels,
                window_s=float(item.get("window_s") or self._audio_duration_cap),
            )

        if item.get("is_video"):
            from app.engine.components.video import VideoFrameLoader

            return VideoFrameLoader().load_clip(
                img_path,
                target_frames=int(item["target_frames"]),
                target_fps=float(item["target_fps"]),
                trim_start_s=float(item.get("trim_start_s") or 0.0),
                trim_end_s=item.get("trim_end_s"),
                target_w=tw,
                target_h=th,
                h_flip=False,
            )

        img = Image.open(img_path).convert("RGB")
        # Smart resize + center crop
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        still = transform_norm(img)  # [C, H, W]
        if is_video_family:
            still = still.unsqueeze(1)  # [C, 1, H, W]
        return still

    def _decode_batch_images(self, items: list[dict], paths: list[str]) -> torch.Tensor:
        """Decode pixels for a batch whose decode was deferred (cache miss).

        Uses the already variant-selected ``paths`` from :meth:`_get_batch` so
        the decoded pixels line up exactly with the ids/cache_dirs used for the
        latent-cache lookup. Mirrors the in-loop decode byte-for-byte.
        """
        transform_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        is_video_family = bool(getattr(self, "is_video_family", False))
        images = [
            self._decode_item_image(item, path, transform_norm, is_video_family)
            for item, path in zip(items, paths)
        ]
        return torch.stack(images).to(self.device)

    @staticmethod
    def _load_image_to(path: str, tw: int, th: int, transform) -> torch.Tensor:
        """Open, smart-resize + center-crop to (tw, th), normalize to [-1, 1]."""
        img = Image.open(path).convert("RGB")
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))
        return transform(img)

    def _load_control_video_clip(
        self,
        path: str,
        target_frames: int,
        target_fps: float,
        target_w: int,
        target_h: int,
        trim_start_s: float = 0.0,
    ) -> torch.Tensor:
        """Decode a control clip trimmed/padded to the TARGET's frame count.

        Decoding STARTS at ``trim_start_s`` — the paired target's own window
        start — so a user-trimmed target, or a tiled sub-window k>0 (whose
        ``trim_start_s`` is ``kΔ``), samples the temporally-aligned control
        segment instead of always sampling from the clip head. Default 0.0
        keeps the untrimmed head-window path byte-identical.

        ``VideoFrameLoader.load_clip`` only raises ``VideoClipTooShort`` when
        an explicit ``trim_end_s`` caps the usable window below what's asked
        for; called here with ``trim_end_s=None`` (an open-ended window from
        ``trim_start_s``), a control clip LONGER than the target's window is
        naturally trimmed (only ``target_frames`` timestamps from the start
        offset are ever sampled) and a control clip SHORTER than the window is
        naturally padded (each timestamp past the clip's last decoded frame
        resolves to that same nearest frame — repeated, never an out-of-range
        error). Either way the output is exactly ``target_frames`` long, so a
        control clip's duration mismatching its paired target never aborts
        training, and the encoded control latent's frame axis matches the
        target latent's (same VAE, same temporal downscale).
        """
        from app.engine.components.video import VideoFrameLoader

        clip = VideoFrameLoader().load_clip(
            path,
            target_frames=target_frames,
            target_fps=target_fps,
            trim_start_s=trim_start_s,
            trim_end_s=None,
            target_w=target_w,
            target_h=target_h,
            h_flip=False,
        )
        # Cheap production-code invariant (previously only asserted in
        # tests): the clip loader's own contract guarantees exactly
        # `target_frames` out (trim/pad, never a short read). A control
        # clip whose frame axis silently drifted from the target's own
        # frame count would misalign every downstream tensor op, so fail
        # loudly here rather than caching a mismatched control latent.
        if clip.shape[1] != target_frames:
            raise ValueError(
                f"control video clip '{path}' decoded to {clip.shape[1]} "
                f"frames, expected {target_frames} (the paired target's own "
                "frame count)"
            )
        return clip

    def _attach_control_images(self, batch, items, transform) -> None:
        """Transpose per-item control fields into per-slot batch tensors.

        A control slot may be a video (Bernini-R edit datasets): decoded via
        the VideoFrameLoader path into a ``[C, F, H, W]`` clip (F matches the
        paired target's own frame count — see :meth:`_load_control_video_clip`)
        instead of the still-image PIL round-trip. Per-slot video-ness is
        read off ``items[0]`` (mirrors the batch-level ``is_video``/``is_audio``
        flags elsewhere in this file); a batch with no video controls carries
        no ``control_is_video`` key and every slot decodes as an image,
        byte-identical to before this method learned about video.

        Two combinations are refused up front with a diagnostic error rather
        than reaching a confusing crash (or worse, a silent mismatch) deeper
        in the stack:

        * A video control slot paired with a still-image target (e.g. a
          same-stem video accidentally dropped into ``control/`` of a
          pre-existing image-edit dataset) — would otherwise die inside
          ``VideoFrameLoader`` with a generic ``target_fps must be > 0``.
        * A video control slot paired with a ``temporal_mode="sliding"``
          target — the target's cached latent holds the FULL clip
          (``cache_frames``) while the control decode targets the per-step
          window (``target_frames``), a frame-axis mismatch downstream.
          Unsupported in v1; not implemented here.
        """
        n_slots = len(items[0]["control_paths"])
        slot_is_video = items[0].get("control_is_video") or [False] * n_slots
        ctrl_images: list[torch.Tensor] = []
        ctrl_ids: list[list[str]] = []
        ctrl_paths: list[list[str]] = []
        ctrl_cache_dirs: list[list[str]] = []
        ctrl_extra_keys: list[list[str]] = []
        for slot in range(n_slots):
            is_video_slot = (
                bool(slot_is_video[slot]) if slot < len(slot_is_video) else False
            )
            slot_imgs: list[torch.Tensor] = []
            slot_ids: list[str] = []
            slot_paths: list[str] = []
            slot_cache: list[str] = []
            slot_extra: list[str] = []
            for item in items:
                path = item["control_paths"][slot]
                cw, ch = item["control_dims"][slot]
                if is_video_slot:
                    if not item.get("is_video"):
                        control_rel = item["control_rel_paths"][slot]
                        raise ValueError(
                            f"control slot '{control_rel}': video control "
                            "paired with a still-image target — video "
                            "controls require a video target"
                        )
                    if item.get("temporal_mode") == "sliding":
                        raise ValueError(
                            f"item {item.get('id')!r}: video control pairs "
                            "do not support temporal_coverage='sliding' yet"
                        )
                    tgt_f = int(item.get("target_frames") or 1)
                    tgt_fps = float(item.get("target_fps") or 0.0)
                    # Honor the paired target's OWN temporal window: a
                    # user-trimmed target, or a tiled sub-window k>0, carries a
                    # nonzero trim_start_s (``_emit_temporal_items`` overwrites
                    # it per window). Decode the control from the SAME start so
                    # window k trains target segment [kΔ,(k+1)Δ] against control
                    # segment [kΔ,(k+1)Δ] — not always the clip head.
                    tgt_start = float(item.get("trim_start_s") or 0.0)
                    slot_imgs.append(
                        self._load_control_video_clip(
                            path, tgt_f, tgt_fps, cw, ch, trim_start_s=tgt_start
                        )
                    )
                    # Mirrors the target video's own t{start}-{end} cache-key
                    # convention (:func:`video_trim_extra_key`) with the REAL
                    # window folded in, so a control source re-used at a
                    # different window (user trim, or a different tiled
                    # sub-window / temporal bucket) never collides in the cache
                    # — both the window START and END now discriminate. An
                    # untrimmed head window (start 0.0) formats t0.0-{end},
                    # byte-identical to before.
                    end_s = (tgt_start + tgt_f / tgt_fps) if tgt_fps > 0 else None
                    slot_extra.append(
                        video_trim_extra_key(
                            {
                                "is_video": True,
                                "trim_start_s": tgt_start,
                                "trim_end_s": end_s,
                            }
                        )
                    )
                else:
                    slot_imgs.append(self._load_image_to(path, cw, ch, transform))
                    slot_extra.append("")
                slot_ids.append(item["control_rel_paths"][slot])
                slot_paths.append(path)
                slot_cache.append(item["control_cache_dirs"][slot])
            ctrl_images.append(torch.stack(slot_imgs).to(self.device))
            ctrl_ids.append(slot_ids)
            ctrl_paths.append(slot_paths)
            ctrl_cache_dirs.append(slot_cache)
            ctrl_extra_keys.append(slot_extra)
        batch["control_images"] = ctrl_images
        batch["control_ids"] = ctrl_ids
        batch["control_paths"] = ctrl_paths
        batch["control_cache_dirs"] = ctrl_cache_dirs
        # Only attached when at least one slot carries a real discriminator —
        # an all-image control batch stays byte-identical (no new batch key,
        # no new kwarg reaching the latent manager in `_load_control_latents`).
        if any(any(keys) for keys in ctrl_extra_keys):
            batch["control_extra_keys"] = ctrl_extra_keys

    def _load_control_latents(self, batch: dict) -> None:
        """Encode/load clean control latents into ``batch['control_latents']``.

        Family-agnostic: runs in the train loop under ``no_grad`` when an edit
        batch carries control images. Control latents are NEVER noised or
        flipped — they're the clean conditioning the family forward concats
        with the noisy target tokens. No-op for non-edit batches. A video
        control slot yields a 5D latent (the VAE/LatentManager infer 5D
        purely from the input tensor's rank — no special-casing needed here
        beyond forwarding the per-slot cache-key discriminator).
        """
        slots_cache = batch.get("control_cache_dirs")
        if not slots_cache:
            return
        use_cache = self.config.get("cache_latents", True)
        slots_extra_keys = batch.get("control_extra_keys")
        control_latents: list[torch.Tensor] = []
        for slot_idx, cache_dirs in enumerate(slots_cache):
            ids = batch["control_ids"][slot_idx]
            paths = batch["control_paths"][slot_idx]
            extra_keys = slots_extra_keys[slot_idx] if slots_extra_keys else None
            # Conditional kwarg: an image-only batch never passes `extra_keys`
            # at all, so a latent-manager stub with the pre-BR1 signature
            # (positional ids/cache_dirs + source_paths) keeps working.
            key_kwargs = (
                {"extra_keys": extra_keys} if extra_keys and any(extra_keys) else {}
            )
            lat = None
            if use_cache:
                lat = self.latent_manager.load_cached_latents(
                    ids,
                    cache_dirs,
                    source_paths=paths,
                    **key_kwargs,
                )
            if lat is None:
                lat = self.latent_manager.encode_and_cache_batch(
                    batch["control_images"][slot_idx],
                    ids=ids,
                    cache_dirs=cache_dirs if use_cache else None,
                    source_paths=paths,
                    **key_kwargs,
                )
            control_latents.append(lat.to(self.device, dtype=self.autocast_dtype))
        batch["control_latents"] = control_latents
