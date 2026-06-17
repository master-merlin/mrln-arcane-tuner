"""
Pipeline Caching Mixin — text embedding and latent pre-caching.
"""

import os

import structlog
import torch
from PIL import Image
from torchvision import transforms


logger = structlog.get_logger(__name__)


def _frames_to_encode(item: dict) -> int:
    """Frames the pre-cache should decode+encode for an item.

    Sliding caches the FULL clip (cache_frames); every other mode caches the
    per-step window (target_frames). Images default to 1.
    """
    if item.get("temporal_mode") == "sliding":
        return int(item.get("cache_frames") or item.get("target_frames", 1) or 1)
    return int(item.get("target_frames", 1) or 1)


class PipelineCachingMixin:
    """Text embedding cache building and latent pre-caching."""

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the text embedding cache before offloading TEs.

        No-op by default.  Override in families that support lazy text
        embedding caching and want to pre-fill the cache so TEs can be
        moved off GPU.
        """

    def _pre_cache_aux(self) -> None:
        """Cache auxiliary per-item latents after the video/image latent pass.

        No-op by default. Families with a SECOND cached modality (LTX-2's audio
        stream) override this to encode + cache it while their extra VAE is still
        resident — ``run_trainer`` offloads VAEs immediately after this runs.
        Unlike the video pre-cache, this is not gated on the video coverage count
        (``_latent_cache_missing``): a dataset can have complete video latents but
        no audio yet, so this hook always runs and does its own per-item check.
        """

    def _build_caption_hints(self) -> dict[str, str]:
        """Build the full set of captions needed for TE pre-caching.

        Returns a dict mapping each unique caption string to a
        human-readable *hint* used for disk-cache filenames.

        Covers:
        - Empty string for full caption dropout
        - Trigger-only variant (if ``persist_triggerword_on_dropout``)
        - Per-image composites: trigger + prefix + caption
          (with inline ``[triggerword]`` expansion)
        - Per-image dropout variants: trigger (if persist) + prefix
        - Sample prompts with wildcard expansion
        """
        trigger = self.config.get("global_triggerword", "")
        persist_trigger = bool(
            self.config.get("persist_triggerword_on_dropout", False)
        )

        caption_hints: dict[str, str] = {}

        # Empty string for full dropout (always needed)
        caption_hints[""] = "dropout_empty"

        # Trigger-only variant for persist_triggerword_on_dropout
        if persist_trigger and trigger:
            caption_hints[trigger] = "dropout_trigger"

        # Per-definition caption-variant resolution (computed once). Defensive:
        # when no definition / no variant / any error, select_training_caption
        # returns item["caption"] verbatim, so behavior is unchanged.
        from app.engine.core.pipeline.caption_selection import select_training_caption

        _definition = getattr(self, "definition", None)
        _def_id = getattr(_definition, "id", None)

        for item in self.inventory:
            cap = select_training_caption(item, _def_id)
            img_name = os.path.splitext(
                os.path.basename(item.get("path", ""))
            )[0]

            # Expand [triggerword] inline in caption text
            trigger_inline = "[triggerword]" in cap
            if trigger and trigger_inline:
                cap = cap.replace("[triggerword]", trigger)

            # Full caption: trigger (if not inline) + prefix + caption
            parts: list[str] = []
            if trigger and not trigger_inline:
                parts.append(trigger)
            if item.get("prefix"):
                parts.append(item["prefix"])
            if cap:
                parts.append(cap)
            composite = ", ".join(parts)
            if composite not in caption_hints:
                caption_hints[composite] = img_name

            # Dropout variant: trigger (if persist) + prefix
            dropout_parts: list[str] = []
            if trigger and persist_trigger:
                dropout_parts.append(trigger)
            if item.get("prefix"):
                dropout_parts.append(item["prefix"])
            dropout_cap = ", ".join(dropout_parts)
            if dropout_cap not in caption_hints:
                caption_hints[dropout_cap] = f"{img_name}_dropout"

            # Masked caption hint — route through the same resolver as
            # _select_variant so the pre-cached embedding matches the trained
            # caption. Gate on has_masked (not has_masked_caption) so masked
            # *variant* items without a masked/{stem}.txt are still cached.
            if item.get("has_masked"):
                m_cap = select_training_caption(item, _def_id, masked=True)
                m_inline = "[triggerword]" in m_cap
                if trigger and m_inline:
                    m_cap = m_cap.replace("[triggerword]", trigger)
                m_parts: list[str] = []
                if trigger and not m_inline:
                    m_parts.append(trigger)
                if item.get("prefix"):
                    m_parts.append(item["prefix"])
                if m_cap:
                    m_parts.append(m_cap)
                m_composite = ", ".join(m_parts)
                if m_composite not in caption_hints:
                    caption_hints[m_composite] = f"{img_name}_masked"

        # Sampling prompts — expand wildcards at pre-cache time
        for idx, sp in enumerate(self.config.get("sample_prompts", [])):
            prompt = (
                sp.get("prompt", "")
                if isinstance(sp, dict)
                else getattr(sp, "prompt", "")
            )
            if prompt:
                expanded = self._expand_wildcards_for_precache(prompt)
                if expanded not in caption_hints:
                    caption_hints[expanded] = f"sample_{idx}"

        return caption_hints

    def _expand_wildcards_for_precache(self, prompt: str) -> str:
        """Expand ``[triggerword]`` and ``[captionprefix]`` in a prompt.

        These are config-time constants, so their values are fully
        deterministic — safe to resolve at pre-cache time.
        """
        triggerword = self.config.get("global_triggerword", "")
        prompt = prompt.replace("[triggerword]", triggerword)

        datasets = self.config.get("datasets", [])
        if datasets:
            first = datasets[0]
            prefix = (
                first.get("caption_prefix", "")
                if isinstance(first, dict)
                else getattr(first, "caption_prefix", "")
            )
        else:
            prefix = ""
        prompt = prompt.replace("[captionprefix]", prefix)
        return prompt

    def _resolve_te_cache_dirs(self) -> list[str]:
        """Collect unique TE cache base directories from the inventory.

        Returns one directory per unique ``(dataset_path, model, version)``
        combination.  The returned paths do NOT include the ``te1``/``te2``
        suffix — families append the appropriate slot themselves.

        Path structure::

            {ds_path}/.cache/{model_name}/{ds_version}/
        """
        model_name = self.definition.id.split("/")[-1]
        seen: set[str] = set()
        dirs: list[str] = []

        for item in self.inventory:
            # Derive dataset_path and version from the latent cache_dir.
            # cache_dir = {ds_path}/.cache/{model}/{version}/latents/{variant}/{resolution}
            cache_dir = item["cache_dir"]
            # Walk upward: cache_dir → resolution → variant → latents → version → model → .cache → ds_path
            variant_dir = os.path.dirname(cache_dir)       # …/{variant}
            latents_dir = os.path.dirname(variant_dir)     # …/latents
            version_dir = os.path.dirname(latents_dir)     # …/{version}
            ds_version = os.path.basename(version_dir)
            model_dir = os.path.dirname(version_dir)       # …/{model}
            cache_root = os.path.dirname(model_dir)        # …/.cache
            ds_path = os.path.dirname(cache_root)          # dataset root

            # Base dir: {ds_path}/.cache/{model_name}/{ds_version}/
            base_dir = os.path.join(ds_path, ".cache", model_name, ds_version)
            key = f"{ds_path}|{model_name}|{ds_version}"
            if key not in seen:
                seen.add(key)
                dirs.append(base_dir)

        return dirs

    # ── Latent Cache ─────────────────────────────────────────────────────

    def _validate_latent_cache(self) -> None:
        """Log latent cache coverage stats.

        Runs only when ``cache_latents=True``.  Scans the full inventory
        and logs how many latent files exist vs are missing.  When
        masked training is enabled, also checks masked variant caches.
        """
        if not self.config.get("cache_latents", True):
            return

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        all_ids = [item["id"] for item in self.inventory]
        all_dirs = [item["cache_dir"] for item in self.inventory]
        all_paths = [item["path"] for item in self.inventory]
        # Video items key their cache file on the trim window; images → "".
        all_extra = [video_trim_extra_key(item) for item in self.inventory]

        # Include masked variant items (image-only — no trim key).
        for item in self.inventory:
            if item.get("has_masked"):
                all_ids.append(item["id"])
                all_dirs.append(item["masked_cache_dir"])
                all_paths.append(item["masked_path"])
                all_extra.append("")

        cached, missing, missing_ids = self.latent_manager.check_cache_coverage(
            all_ids, all_dirs, source_paths=all_paths, extra_keys=all_extra,
        )

        if missing == 0:
            self.logger.info(
                "latent_cache_complete",
                cached=cached,
                message="All latents found in cache — skipping pre-cache step.",
            )
        elif cached == 0:
            self.logger.info(
                "latent_cache_empty",
                total=missing,
                message="No cached latents found — will encode all during pre-cache.",
            )
        else:
            self.logger.warning(
                "latent_cache_partial",
                cached=cached,
                missing=missing,
                sample_missing=missing_ids[:20],
                message=f"{missing} latent file(s) missing from cache.",
            )

        self._latent_cache_missing = missing

    async def _pre_cache_latents(self) -> None:
        """Encode and cache ALL uncached latents before training starts.

        Iterates the full inventory individually, skipping already-cached
        items.  This prevents mid-training VAE fallback which can OOM when
        the GPU is already loaded with the transformer + LoRA.

        When masked training is enabled, also pre-caches masked variant
        latents alongside the originals.
        """
        if not self.config.get("cache_latents", True):
            return
        if getattr(self, "_latent_cache_missing", 0) == 0:
            return

        self.logger.info(
            "pre_caching_latents_start",
            total_missing=self._latent_cache_missing,
        )
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Latents (0%)")

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        # Build unique work items (deduplicate by id + cache_dir + trim key).
        # Video items carry the temporal fields + the trim cache-key so they
        # encode through the same VideoFrameLoader path (and under the same
        # filename) as the train loop; images stay byte-identical to before.
        seen: set[str] = set()
        work_items: list[dict] = []

        for item in self.inventory:
            extra_key = video_trim_extra_key(item)
            # Original variant
            key = f"{item['cache_dir']}/{item['id']}/{extra_key}"
            if key not in seen:
                seen.add(key)
                fname = self.latent_manager.latent_filename(
                    item["id"], item["path"], extra_key
                )
                path = os.path.join(item["cache_dir"], fname)
                if not os.path.exists(path):
                    work_items.append({
                        "path": item["path"],
                        "id": item["id"],
                        "cache_dir": item["cache_dir"],
                        "target_w": item["target_w"],
                        "target_h": item["target_h"],
                        "is_video": item.get("is_video", False),
                        "target_frames": item.get("target_frames", 1),
                        "target_fps": item.get("target_fps"),
                        "trim_start_s": item.get("trim_start_s"),
                        "trim_end_s": item.get("trim_end_s"),
                        "temporal_mode": item.get("temporal_mode"),
                        "cache_frames": item.get("cache_frames"),
                        "extra_key": extra_key,
                    })

            # Masked variant (image-only — no trim key).
            if item.get("has_masked"):
                m_key = f"{item['masked_cache_dir']}/{item['id']}"
                if m_key not in seen:
                    seen.add(m_key)
                    fname = self.latent_manager.latent_filename(item["id"], item["masked_path"])
                    path = os.path.join(item["masked_cache_dir"], fname)
                    if not os.path.exists(path):
                        work_items.append({
                            "path": item["masked_path"],
                            "id": item["id"],
                            "cache_dir": item["masked_cache_dir"],
                            "target_w": item["target_w"],
                            "target_h": item["target_h"],
                            "is_video": False,
                            "extra_key": "",
                        })

        total = len(work_items)
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for i, item in enumerate(work_items):
            try:
                tw, th = item["target_w"], item["target_h"]
                if item.get("is_video"):
                    # Decode the clip via the SAME loader the train loop uses →
                    # [C, F, H, W], already normalized to [-1, 1]. Encoding a
                    # video frame through PIL is what produced the original
                    # "cannot identify image file" failure.
                    from app.engine.components.video import VideoFrameLoader

                    clip = VideoFrameLoader().load_clip(
                        item["path"],
                        target_frames=_frames_to_encode(item),
                        target_fps=float(item.get("target_fps") or 0.0),
                        trim_start_s=float(item.get("trim_start_s") or 0.0),
                        trim_end_s=item.get("trim_end_s"),
                        target_w=tw,
                        target_h=th,
                        h_flip=False,
                    )
                    input_tensor = clip.unsqueeze(0).to(self.device)  # [1,C,F,H,W]
                else:
                    img = Image.open(item["path"]).convert("RGB")
                    scale = max(tw / img.width, th / img.height)
                    nw, nh = int(img.width * scale), int(img.height * scale)
                    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                    left = (nw - tw) // 2
                    top = (nh - th) // 2
                    img = img.crop((left, top, left + tw, top + th))
                    input_tensor = transform_norm(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.latent_manager.encode_and_cache_batch(
                        input_tensor,
                        ids=[item["id"]],
                        cache_dirs=[item["cache_dir"]],
                        source_paths=[item["path"]],
                        extra_keys=[item.get("extra_key", "")],
                    )

                pct = round((i + 1) / total * 100)
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Latents ({pct}%)")
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    self.logger.info(
                        "pre_caching_progress",
                        encoded=i + 1,
                        total=total,
                    )
            except (OSError, ValueError, RuntimeError) as e:
                self.logger.error(
                    "pre_cache_encode_failed",
                    id=item["id"],
                    error=str(e),
                )
                raise RuntimeError(
                    f"Failed to encode latent for '{item['id']}': {e}"
                ) from e

        self.logger.info("pre_caching_latents_done", encoded=total)

    # ── Cache Manifest ────────────────────────────────────────────────────

    def _build_cache_manifest(self) -> dict[str, list[str]] | None:
        """Build a cache manifest from current cache directories.

        Scans the latent and/or embedding cache directories used by the
        current inventory and collects all ``.safetensors`` filenames.

        Returns:
            Dict with ``"latents"`` and/or ``"embeddings"`` keys mapping
            to lists of filenames, or ``None`` if neither persist flag is on.
        """
        manifest: dict[str, list[str]] = {}

        if self.config.get("persist_latents"):
            dirs: set[str] = set()
            for item in self.inventory:
                if item.get("cache_dir"):
                    dirs.add(item["cache_dir"])
            latent_files: list[str] = []
            for d in sorted(dirs):
                if os.path.isdir(d):
                    latent_files.extend(
                        f for f in os.listdir(d) if f.endswith(".safetensors")
                    )
            manifest["latents"] = latent_files

        if self.config.get("persist_embeddings"):
            emb_dirs: set[str] = set()
            for item in self.inventory:
                # Embedding cache dirs live one level up from latent cache dirs
                # e.g. .cache/{model}/{version}/embeddings/te1/
                ds_path = item.get("dataset_path", "")
                model = self.config.get("model_name", "")
                version = item.get("dataset_version", "")
                if ds_path and model and version:
                    base = os.path.join(ds_path, ".cache", model, version, "embeddings")
                    if os.path.isdir(base):
                        for sub in os.listdir(base):
                            sub_path = os.path.join(base, sub)
                            if os.path.isdir(sub_path):
                                emb_dirs.add(sub_path)
            emb_files: list[str] = []
            for d in sorted(emb_dirs):
                if os.path.isdir(d):
                    emb_files.extend(
                        f for f in os.listdir(d) if f.endswith(".safetensors")
                    )
            manifest["embeddings"] = emb_files

        return manifest if manifest else None
