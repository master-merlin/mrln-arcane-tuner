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
    return f"t{ts}-{te}"


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

    async def prepare_data(self):
        """Fetch datasets via API, build inventory with aspect-ratio bucketing."""
        self.logger.info("preparing_data")

        resolutions = self.config.get("resolutions", [1024])
        self.bucket_manager = BucketManager(base_resolutions=resolutions)

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
            raise ValueError(
                "Video configuration invalid: " + "; ".join(_vc.errors)
            )

        # ── Video bucketing config (fields land fully in phase B6) ──
        # Read defensively: config may not carry video knobs yet. When a frame
        # rule is present we build a temporal-aware BucketManager; otherwise
        # video items fall back to a single requested frame count.
        self._video_frame_rule = self.config.get("frame_rule") or None
        # Default target fps: native (None → use clip's own fps later).
        self._video_target_fps = self.config.get("target_fps") or None
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
        # Cache of frame-ladder BucketManagers keyed by max-frames cap, so a
        # per-dataset override builds its ladder once and is reused.
        self._video_bm_cache: dict[int, Any] = {}
        self._video_bucket_manager = self._video_bucket_manager_for(
            self._video_num_frames
        )

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
                            native_fps = float(meta.get("fps") or 0.0)
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
                            _base_fps = float(
                                self._video_target_fps or native_fps or 0.0
                            )
                            vid_target_fps = self._effective_fps(_base_fps)
                            available_frames = (
                                int(eff_dur * vid_target_fps)
                                if vid_target_fps > 0
                                else 0
                            )
                            vid_fps = vid_target_fps

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
                            buckets = (
                                self.bucket_manager.get_buckets_for_all_resolutions(
                                    w, h
                                )
                            )
                        else:
                            buckets = [self.bucket_manager.get_bucket(w, h)]

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

    def _get_batch(self, items: list[dict]) -> dict[str, Any]:
        """Build a training batch from inventory items.

        When augmentation is configured, applies per-sample variant
        selection (original / masked).  Flip augmentation is applied
        later to the latent tensor in the training loop.
        """
        images: list[torch.Tensor] = []
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

        for item in items:
            # ── Variant selection ──
            img_path, cap, cache_dir = self._select_variant(item)
            tw, th = item["target_w"], item["target_h"]

            if item.get("is_video"):
                # Decode a clip → [C, F, H, W] (flip applied later on latents,
                # matching the image path's latent-space augmentation).
                from app.engine.components.video import VideoFrameLoader

                clip = VideoFrameLoader().load_clip(
                    img_path,
                    target_frames=int(item["target_frames"]),
                    target_fps=float(item["target_fps"]),
                    trim_start_s=float(item.get("trim_start_s") or 0.0),
                    trim_end_s=item.get("trim_end_s"),
                    target_w=tw,
                    target_h=th,
                    h_flip=False,
                )
                images.append(clip)
                # Trim window folds into the cache-file hash (shared helper so
                # pre-caching computes the identical key).
                extra_keys.append(video_trim_extra_key(item))
            else:
                img = Image.open(img_path).convert("RGB")

                # Smart resize + center crop
                scale = max(tw / img.width, th / img.height)
                nw, nh = int(img.width * scale), int(img.height * scale)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                left = (nw - tw) // 2
                top = (nh - th) // 2
                img = img.crop((left, top, left + tw, top + th))

                still = transform_norm(img)  # [C, H, W]
                # In a video-family run, lift stills to a 1-frame clip so the
                # whole batch collates 5D. Pure image families keep [C, H, W].
                if is_video_family:
                    still = still.unsqueeze(1)  # [C, 1, H, W]
                images.append(still)
                extra_keys.append("")

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
            "images": torch.stack(images).to(self.device),
            "captions": captions,
            "ids": ids,
            "cache_dirs": cache_dirs,
            "paths": batch_paths,
        }
        # Per-item cache discriminators (video trim window). Only attached when
        # at least one item carries one, so image batches are untouched.
        if any(extra_keys):
            batch["extra_keys"] = extra_keys

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

    def _attach_control_images(self, batch, items, transform) -> None:
        """Transpose per-item control fields into per-slot batch tensors."""
        n_slots = len(items[0]["control_paths"])
        ctrl_images: list[torch.Tensor] = []
        ctrl_ids: list[list[str]] = []
        ctrl_paths: list[list[str]] = []
        ctrl_cache_dirs: list[list[str]] = []
        for slot in range(n_slots):
            slot_imgs: list[torch.Tensor] = []
            slot_ids: list[str] = []
            slot_paths: list[str] = []
            slot_cache: list[str] = []
            for item in items:
                path = item["control_paths"][slot]
                cw, ch = item["control_dims"][slot]
                slot_imgs.append(self._load_image_to(path, cw, ch, transform))
                slot_ids.append(item["control_rel_paths"][slot])
                slot_paths.append(path)
                slot_cache.append(item["control_cache_dirs"][slot])
            ctrl_images.append(torch.stack(slot_imgs).to(self.device))
            ctrl_ids.append(slot_ids)
            ctrl_paths.append(slot_paths)
            ctrl_cache_dirs.append(slot_cache)
        batch["control_images"] = ctrl_images
        batch["control_ids"] = ctrl_ids
        batch["control_paths"] = ctrl_paths
        batch["control_cache_dirs"] = ctrl_cache_dirs

    def _load_control_latents(self, batch: dict) -> None:
        """Encode/load clean control latents into ``batch['control_latents']``.

        Family-agnostic: runs in the train loop under ``no_grad`` when an edit
        batch carries control images. Control latents are NEVER noised or
        flipped — they're the clean conditioning the family forward concats
        with the noisy target tokens. No-op for non-edit batches.
        """
        slots_cache = batch.get("control_cache_dirs")
        if not slots_cache:
            return
        use_cache = self.config.get("cache_latents", True)
        control_latents: list[torch.Tensor] = []
        for slot_idx, cache_dirs in enumerate(slots_cache):
            ids = batch["control_ids"][slot_idx]
            paths = batch["control_paths"][slot_idx]
            lat = None
            if use_cache:
                lat = self.latent_manager.load_cached_latents(
                    ids,
                    cache_dirs,
                    source_paths=paths,
                )
            if lat is None:
                lat = self.latent_manager.encode_and_cache_batch(
                    batch["control_images"][slot_idx],
                    ids=ids,
                    cache_dirs=cache_dirs if use_cache else None,
                    source_paths=paths,
                )
            control_latents.append(lat.to(self.device, dtype=self.autocast_dtype))
        batch["control_latents"] = control_latents
