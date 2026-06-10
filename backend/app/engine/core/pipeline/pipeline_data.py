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


class PipelineDataMixin:
    """Dataset preparation, inventory building, and batch construction."""

    # ── Prepare Data (shared) ────────────────────────────────────────────

    async def prepare_data(self):
        """Fetch datasets via API, build inventory with aspect-ratio bucketing."""
        self.logger.info("preparing_data")

        resolutions = self.config.get("resolutions", [1024])
        self.bucket_manager = BucketManager(base_resolutions=resolutions)

        datasets_config = self.config.get("datasets", [])
        inventory: list[dict[str, Any]] = []

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
            timeout=60.0, headers=_internal_api_headers(),
        ) as client:
            for ds_config in datasets_config:
                if isinstance(ds_config, str):
                    ds_config = {"dataset_name": ds_config}

                name = ds_config.get("dataset_name")
                repeats = int(ds_config.get("num_repeats", 1))
                prefix = ds_config.get("caption_prefix", "")
                ignore_filter = ds_config.get("ignore_filter", False)

                # Per-dataset masking config
                ds_masking_enabled = bool(ds_config.get("masking_enabled", False))
                ds_original_weight = max(float(ds_config.get("original_weight", 0.70)), 0.50)

                try:
                    resp = await client.get(f"{api_url}/datasets/{name}/pairs")
                    if resp.status_code != 200:
                        self.logger.warning("dataset_fetch_failed", name=name, status=resp.status_code)
                        continue

                    pairs = resp.json()
                    ds_resp = await client.get(f"{api_url}/datasets/{name}")
                    ds_data = ds_resp.json()
                    ds_path = ds_data.get("path")
                    ds_version = ds_data.get("version", "1.0.0")

                    # Recreate masked images if requested
                    if ds_masking_enabled and bool(ds_config.get("recreate_masks", False)):
                        mask_opacity = float(ds_config.get("mask_opacity", 0.0))
                        self.logger.info(
                            "recreating_masks",
                            dataset=name,
                            opacity=mask_opacity,
                        )
                        from app.core.masking.masking_service import MaskingService
                        masking_svc = MaskingService()
                        result = masking_svc.mass_apply(
                            ds_path, mask_opacity, overwrite=True,
                        )
                        self.logger.info(
                            "masks_recreated",
                            dataset=name,
                            applied=result["applied"],
                            skipped=result["skipped"],
                        )

                    for pair in pairs:
                        img_rel = pair.get("media_file")
                        if not img_rel:
                            continue
                        if not ignore_filter:
                            meta = pair.get("metadata") or {}
                            if meta.get("enabled") is False:
                                continue

                        img_path = os.path.join(ds_path, img_rel)
                        caption = pair.get("caption_content", "")

                        meta = pair.get("metadata", {})
                        w, h = meta.get("width"), meta.get("height")
                        if not w:
                            try:
                                with Image.open(img_path) as img:
                                    w, h = img.size
                            except (OSError, ValueError):
                                continue

                        # ── Masked variant info ──
                        stem = os.path.splitext(os.path.basename(img_rel))[0]
                        masked_img_path = os.path.join(ds_path, "masked", f"{stem}.jpg")
                        has_masked = ds_masking_enabled and os.path.isfile(masked_img_path)

                        # Masked caption: read from masked/ alongside masked image
                        masked_caption = None
                        has_masked_caption = False
                        if has_masked:
                            masked_cap_path = os.path.join(ds_path, "masked", f"{stem}.txt")
                            if os.path.isfile(masked_cap_path):
                                try:
                                    with open(masked_cap_path, "r", encoding="utf-8") as f:
                                        masked_caption = f.read().strip()
                                        has_masked_caption = True
                                except OSError:
                                    pass

                        bucketing_mode = self.config.get("bucketing_mode", "kohya")
                        if bucketing_mode == "multi":
                            buckets = self.bucket_manager.get_buckets_for_all_resolutions(w, h)
                        else:
                            buckets = [self.bucket_manager.get_bucket(w, h)]

                        for bucket in buckets:
                            target_w, target_h = bucket["width"], bucket["height"]
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
                                "dropout_rate": float(ds_config.get("caption_dropout_rate", 0.0)),
                                "orig_w": w,
                                "orig_h": h,
                                "target_w": target_w,
                                "target_h": target_h,
                                "cache_dir": cache_dir,
                                "variant": "original",
                            }

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

                            for _ in range(repeats):
                                inventory.append(item)

                except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
                    self.logger.error("dataset_api_error", name=name, error=str(e))
                    continue

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
            vae, device=self.device,
            arch_params=getattr(self.definition, "architecture_params", None),
        )

    # ── Batch Construction ───────────────────────────────────────────────

    def _select_variant(self, item: dict) -> tuple[str, str, str]:
        """Pick image variant for this training step.

        Returns ``(path, caption, cache_dir)`` for the selected variant.
        When the item has a masked version (per-dataset masking),
        weighted random selection is applied using the item's own weight.

        Caption handling for masked variants:
        - If a dedicated ``masked/{stem}.txt`` exists → use it as-is
        - Otherwise → fall back to ``triggerword + prefix`` only,
          avoiding the full original caption which describes spatial
          context (backgrounds, viewing angles) absent from the
          masked image.
        """
        if (
            item.get("has_masked")
            and random.random() >= item.get("original_weight", 0.70)
        ):
            if item.get("has_masked_caption"):
                cap = item["masked_caption"]
            else:
                # No dedicated masked caption — return empty string so
                # _get_batch assembles trigger + prefix only (it always
                # prepends those).  This avoids the full original caption
                # which describes spatial context absent from the masked image.
                cap = ""
            return (item["masked_path"], cap, item["masked_cache_dir"])
        # General (non-masked) caption: route through the per-definition
        # variant resolver so a variant overrides item["caption"]. Mirrors
        # _build_caption_hints (TE pre-cache). Defensive: when no definition /
        # no variant / use_general / any error, select_training_caption returns
        # item["caption"] verbatim, so the per-step caption is byte-identical.
        from app.engine.core.pipeline.caption_selection import select_training_caption

        _def_id = getattr(getattr(self, "definition", None), "id", None)
        _cfg = getattr(self, "config", None)
        _use_general = bool(_cfg.get("use_general_captions", False)) if isinstance(_cfg, dict) else False
        cap = select_training_caption(item, _def_id, _use_general)
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

        trigger = self.config.get("global_triggerword", "")
        persist_trigger = bool(self.config.get("persist_triggerword_on_dropout", False))
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for item in items:
            # ── Variant selection ──
            img_path, cap, cache_dir = self._select_variant(item)

            img = Image.open(img_path).convert("RGB")
            tw, th = item["target_w"], item["target_h"]

            # Smart resize + center crop
            scale = max(tw / img.width, th / img.height)
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - tw) // 2
            top = (nh - th) // 2
            img = img.crop((left, top, left + tw, top + th))

            images.append(transform_norm(img))

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
            if trigger and not trigger_used_inline and (not is_dropped or persist_trigger):
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
        # Let families add extra data (e.g. SDXL time_ids)
        batch.update(self.build_batch_extra(items))
        return batch
