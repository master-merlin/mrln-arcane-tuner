"""MiniMax-H3-Image trainer: the family hooks of the generic training pipeline.

- **Setup** (``_setup_family``, the first family hook ``setup()`` calls, before
  any weight loads) refuses what this family cannot train, then gives a config
  without ``timestep_sampling`` the family default (``timestep.default_mode`` /
  ``timestep.fixed_sigmas``, ``setdefault``-style, the moss_image way: a mode
  the user set is never touched), then builds the driver and the loader with
  the DiT deferred. ``self.driver = MiniMaxH3ImageDriver(...)`` is written
  literally: ``tests/engine/test_hook_wiring_meta.py`` finds a family's driver
  by that assignment shape.
- **Clock**: the driver's CLOBBER hooks (``sample_timesteps``, ``add_noise``,
  ``compute_target``) are delegated explicitly, so the family never relies on
  the base pipeline's auto-delegation. The loss is the shared one.
- **Text**: every caption is encoded once while the encoder is resident and
  cached (memory and disk); the 66.7 GB encoder is then released without a
  host copy, before the 66.28 GB DiT loads. A caption missing from the cache
  after the release is an error naming it, never a silent reload.
- **Data**: video and audio files are skipped loudly, with a count; a dataset
  left without images fails. Every latent-cache dir is keyed on the driver's
  ``latent_cache_fingerprint`` (pixel adapter, VAE config, bucket rule), so a
  change to any of them never reuses stale latents.
"""

from __future__ import annotations

import os

import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.minimax_h3_shared import text as h3_text

from . import settings as h3i_settings
from .driver import MiniMaxH3ImageDriver

# The disk cache's filename hint: the entry's identity is the key (prefix, definition, tap,
# tokenizer, caption), never the image that first carried the caption.
_TE_CACHE_HINT = "minimax_h3_image"

_LIVE_TEXT_ENCODING_REFUSAL = (
    "minimax_h3_image requires cache_text_embeddings=True: the 66.7 GB Qwen3-VL encoder is "
    "released before the 66.28 GB transformer loads, so every caption is encoded once, up front"
)


class MiniMaxH3ImageTrainer(GenericTrainingPipeline):
    """MiniMax-H3-Image LoRA trainer (text-to-image)."""

    # The transformer's clock: t = 1 - sigma in [0, 1], t = 1 clean.
    timesteps_count_clean_fraction = True
    validation_timestep_domain = (0.0, 1.0)

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Refuse first, then the timestep default, then the driver and loader."""
        self._refuse_unsupported_settings()
        from .loader import MiniMaxH3ImageLoader

        mode, source = h3i_settings.apply_timestep_default(self.definition, self.config)
        self.config.setdefault("flux_shift_patchify_factor", 2)  # 2x2 latent patches per token
        self.driver = MiniMaxH3ImageDriver(self.definition, self.device)
        self.loader = MiniMaxH3ImageLoader(self.device, defer_transformer=True)
        self.logger.info(
            "minimax_h3_image_timestep_sampling",
            mode=mode,
            source=source,
            fixed_sigmas=self.config.get("fixed_sigmas") if mode == "fixed_sigmas" else None,
        )

    def _refuse_unsupported_settings(self) -> None:
        """Every refusal names the setting and what to set instead, before any weight loads."""
        cfg = self.config
        if not cfg.get("cache_text_embeddings", True):
            raise ValueError(_LIVE_TEXT_ENCODING_REFUSAL)
        if not cfg.get("cache_latents", True):
            raise ValueError(
                "minimax_h3_image: cache_latents=False is not supported; the 66.28 GB transformer "
                "loads only after every image is encoded and cached. Set cache_latents to true"
            )
        if not cfg.get("mixed_precision"):
            cfg["mixed_precision"] = "bf16"
        precision = cfg["mixed_precision"]
        if precision != "bf16":
            raise ValueError(
                f"minimax_h3_image: mixed_precision={precision!r} is not supported; every weight of this "
                "model ships bf16. Set mixed_precision to bf16"
            )
        if cfg.get("train_text_encoder"):
            raise ValueError(
                "minimax_h3_image: train_text_encoder is not supported; the text encoder is released "
                "before the transformer loads and only its cached embeddings are used"
            )
        te_quantization = str(cfg.get("te_quantization") or "none")
        if te_quantization != "none":
            raise ValueError(
                f"minimax_h3_image: te_quantization={te_quantization!r} is not supported; the text "
                "encoder runs once, unquantized, before it is released. Set te_quantization to none"
            )
        quantization = str(cfg.get("quantization") or "none")
        if quantization not in ("none", "int8"):
            raise ValueError(
                f"minimax_h3_image: quantization={quantization!r} is not supported; use none or int8"
            )
        control = max(int(cfg.get("control_inputs") or 0), int(getattr(self.definition, "control_inputs", 0) or 0))
        if control > 0:
            raise ValueError(
                f"minimax_h3_image: control_inputs={control} is not supported; this family trains "
                "text-to-image only (editing with reference images is not part of it)"
            )
        for ds in cfg.get("datasets") or []:
            if isinstance(ds, dict) and ds.get("masking_enabled"):
                raise ValueError(
                    f"minimax_h3_image: dataset {ds.get('dataset_name')!r} has masking_enabled=True; "
                    "masked training is not supported for this family. Turn masking off for this dataset"
                )

    def _create_sampler(self):
        """The 6-step preview sampler when sampling is on (``sample_every_n_steps > 0``)."""
        if int(self.config.get("sample_every_n_steps", 0) or 0) > 0:
            from .sampler import MiniMaxH3ImageSampler

            return MiniMaxH3ImageSampler(self)
        return None

    # ── CLOBBER hooks, delegated explicitly ───────────────────────────────

    def sample_timesteps(self, batch_size: int, latents: torch.Tensor | None = None) -> torch.Tensor:
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        return self.driver.sample_timesteps(batch_size, self.device, self.config, latents=latents, progress=progress)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.driver.add_noise(latents, noise, timesteps)

    def compute_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.driver.compute_target(latents, noise, timesteps)

    # ── Data ──────────────────────────────────────────────────────────────

    async def prepare_data(self):
        await super().prepare_data()
        self._skip_non_image_items()
        self._fingerprint_latent_cache_dirs()

    def _fingerprint_latent_cache_dirs(self) -> None:
        """Key every latent-cache dir on what its latents depend on (the driver's fingerprint):
        coverage, the pre-cache and the batches all read ``item["cache_dir"]`` from the
        inventory the base ``prepare_data`` wrote, so the key is applied here, before any of
        them. The key goes INSIDE the last path segment: ``_resolve_te_cache_dirs`` walks a
        fixed number of levels up from ``cache_dir``."""
        divisibility = int(getattr(getattr(self, "bucket_manager", None), "divisibility", 32) or 32)
        fingerprint = self.driver.latent_cache_fingerprint(divisibility)
        for item in self.inventory:
            for key in ("cache_dir", "masked_cache_dir"):
                path = item.get(key)
                if path:
                    head, tail = os.path.split(path.rstrip("/\\"))
                    item[key] = os.path.join(head, f"{tail}-h3i{fingerprint}")
        self.logger.info("minimax_h3_image_latent_cache_fingerprint", fingerprint=fingerprint)

    def _skip_non_image_items(self) -> None:
        """The shared ingestion turns video and audio files into inventory items
        for any family; this one trains on images only. Each skipped file is
        named once, the counts go on one line, and an empty result fails."""
        kinds = {"video": [], "audio": []}
        kept = []
        for item in self.inventory:
            if item.get("is_video"):
                kinds["video"].append(item)
            elif item.get("is_audio"):
                kinds["audio"].append(item)
            else:
                kept.append(item)
        if not kinds["video"] and not kinds["audio"]:
            return
        counts = {}
        for kind, items in kinds.items():
            paths = sorted({item["path"] for item in items})
            counts[kind] = len(paths)
            for path in paths:
                self.logger.warning(
                    "media_file_skipped",
                    media=os.path.basename(path),
                    message=f"a {kind} file; this model trains on images only",
                )
        self.inventory = kept
        self.logger.info(
            "minimax_h3_image_media_skipped",
            skipped_video_files=counts["video"],
            skipped_audio_files=counts["audio"],
            total_items=len(kept),
        )
        if not kept:
            raise ValueError(
                f"No training data found in datasets. {counts['video']} video file(s) and "
                f"{counts['audio']} audio file(s) were skipped; this model trains on images."
            )

    # ── Text embeddings ───────────────────────────────────────────────────

    def _te_cache_dir(self) -> str:
        """``{dataset cache}/embeddings/{te_quant}/{tap + tokenizer}/te1``, or "" with no dataset dir."""
        dirs = self._resolve_te_cache_dirs()
        if not dirs:
            return ""
        te_quant = self.config.get("te_quantization", "none")
        return os.path.join(dirs[0], "embeddings", te_quant, self.driver.te_cache_scope(), "te1")

    def _pre_cache_text_embeddings(self) -> None:
        """Encode every caption the run will ask for while the encoder is resident:
        the composites, the dropout variants and the empty prompt, and the
        sample prompts (``_build_caption_hints``). An entry is the caption's
        unpadded ``[L, D]`` tap; memory first, then disk, then the encoder."""
        if not self.config.get("cache_text_embeddings", True):
            raise ValueError(_LIVE_TEXT_ENCODING_REFUSAL)
        if self.driver.text_encoder is None:
            raise RuntimeError("minimax_h3_image: the text encoder is not loaded; nothing can be cached")

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_dir = self._te_cache_dir()
        dtype = self.driver.resolve_loading_dtype()
        from_disk = 0
        need: list[str] = []
        for caption in self._build_caption_hints():
            if caption in self.text_cache:
                continue
            found = TextEmbeddingCache.load(self.driver.te_cache_key(caption), te_dir, _TE_CACHE_HINT) if te_dir else None
            if found is not None and found.ndim == 2:
                self.text_cache[caption] = found
                from_disk += 1
            else:
                need.append(caption)
        with torch.no_grad():
            for i, caption in enumerate(need):
                out = self.driver.encode_text([caption], dtype)
                row = out.embeddings[0].detach().to("cpu")
                self.text_cache[caption] = row
                if te_dir:
                    TextEmbeddingCache.save(self.driver.te_cache_key(caption), row, te_dir, _TE_CACHE_HINT)
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Text Embeddings ({round((i + 1) / len(need) * 100)}%)")
        self.logger.info(
            "minimax_h3_image_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=from_disk,
            newly_encoded=len(need),
        )

    def encode_text(self, captions: list[str], dtype: torch.dtype, batch: dict | None = None) -> TextEncoderOutput:
        """Serve from the cache; a miss while the encoder is resident is encoded
        and cached, a miss after the release raises naming the caption."""
        rows = []
        for caption in captions:
            row = self.text_cache.get(caption)
            if row is None:
                if self.driver.text_encoder is None:
                    raise RuntimeError(
                        "minimax_h3_image encode_text: the text encoder was released and this caption "
                        f"was not pre-cached: {caption[:80]!r}"
                    )
                with torch.no_grad():
                    row = self.driver.encode_text([caption], dtype).embeddings[0].detach().to("cpu")
                self.text_cache[caption] = row
            rows.append(row)
        out = h3_text.pad_rows(rows, dtype)
        return TextEncoderOutput(
            embeddings=out.embeddings.to(self.device), attention_mask=out.attention_mask.to(self.device)
        )

    def _offload_text_encoders(self) -> None:
        """Release the encoder unconditionally and without a host copy (the shared release)."""
        h3_text.release_text_encoder(self)

    def _get_text_encoders(self) -> dict[str, torch.nn.Module]:
        return self.driver.get_text_encoders()

    # ── The deferred transformer ──────────────────────────────────────────

    def _materialise_transformer(self) -> None:
        """Load the deferred DiT, only once the encoder is gone."""
        self.driver.assert_text_encoder_released()
        model = self.loader.load_transformer(self.definition, self.driver.resolve_loading_dtype(), initial_device="cpu")
        self.components["transformer"] = model
        self.components["unet"] = model  # the base pipeline's primary key
        self.driver.assign_components(self.components)
        self.transformer = model
        self.logger.info("transformer loaded", subfolder="transformer_diffusers")

    async def prepare_for_training(self):
        # Freezing / wrapping the primary model starts here, so the deferred DiT must exist.
        if self.components.get("transformer") is None:
            self._materialise_transformer()
        await super().prepare_for_training()

    def _get_primary_model(self) -> torch.nn.Module:
        return self.driver.get_primary_model()

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep every alias on the one model after the PEFT wrap: the trainer's,
        both component keys, and the driver's (its forward runs this model)."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.components["transformer"] = new_model
        self.driver.transformer = new_model

    def _resolve_loading_dtype(self) -> torch.dtype:
        return self.driver.resolve_loading_dtype()
