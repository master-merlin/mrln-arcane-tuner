"""
Pipeline Loading Mixin — model loading, GPU placement, quantization, offloading.
"""

import gc
import os

import structlog
import torch

from app.engine.strategies.noise_interpolation import NoiseInterpolation

logger = structlog.get_logger(__name__)


class PipelineLoadingMixin:
    """Model loading, quantization, and component lifecycle management."""

    # ── Load Model (Phase A — all to CPU) ────────────────────────────────

    async def load_model(self):
        """Load model weights and metadata (Phase A).

        Places all components on **CPU** to minimize peak VRAM.  The
        orchestrator (``run_trainer.py``) then selectively moves
        components to GPU for caching phases before calling
        ``prepare_for_training()`` (Phase B).

        Steps:
        1. Load weights via family loader → CPU
        2. Enrich definition with introspection
        3. Init scheduler + noise interpolation
        """
        self.logger.info("loading_model")

        # 1. Family-specific loading — all components to CPU
        torch_dtype = self.driver.resolve_loading_dtype()
        self.components = await self.loader.load(
            self.definition, torch_dtype=torch_dtype, initial_device="cpu",
        )
        self._assign_components()

        # Surface loader diagnostics via the JobLogWriter warning channel
        _lw = getattr(self, "_log_writer", None)
        for warning in getattr(self.loader, "warnings", []):
            if _lw:
                _lw.warning(warning)

        # 2. Enrich definition with introspection
        from app.engine.models.registry import ModelRegistry
        root_path = getattr(self.loader, "_root_path", None)
        ModelRegistry.enrich_definition(
            self.definition.id, self.components, root_path=root_path
        )
        # Re-read: enrich_definition creates a new object via model_copy
        updated = ModelRegistry.get_definition(self.definition.id)
        if updated:
            self.definition = updated
            # Sync enriched definition to driver so architecture params
            # (te.output_layers, te.max_length, etc.) are current.
            # We update the definition ref AND refresh cached arch params
            # directly — calling _assign_components() is avoided because
            # it re-zeroes the guidance embedder and has other side-effects.
            if hasattr(self, "driver") and self.driver is not None:
                self.driver.definition = updated
                # Refresh TE arch params the driver may have cached
                arch = getattr(updated, "architecture_params", {}) or {}
                for attr, key in [
                    ("te_max_length", "te.max_length"),
                    ("te_concat_layers", "te.concat_layers"),
                    ("te_model_type", "te.model_type"),
                    ("te_output_layers", "te.output_layers"),
                ]:
                    if hasattr(self.driver, attr) and key in arch:
                        val = arch[key]
                        if attr == "te_concat_layers":
                            val = int(val)
                        setattr(self.driver, attr, val)
            self.logger.info(
                "definition_post_enrichment",
                arch_params=sorted(self.definition.architecture_params.keys()),
            )

        # 3. Init scheduler (family hook)
        self.scheduler = self.init_scheduler()
        self.components["scheduler"] = self.scheduler

        # 3b. Noise interpolation (configurable: linear, ddpm, cosine)
        interp_mode = self.config.get("noise_interpolation", "linear")
        self.noise_interpolation = NoiseInterpolation(
            mode=interp_mode,
            scheduler=self.scheduler,
        )

    # ── Component GPU Management ─────────────────────────────────────────

    def _move_component_to_gpu(self, *names: str) -> None:
        """Move named components from CPU to GPU.

        Looks up each name first as an instance attribute, then in
        ``self.components``.  Skips non-torch objects (tokenizers).
        """
        for name in names:
            comp = getattr(self, name, None) or self.components.get(name)
            if comp is None:
                continue
            if hasattr(comp, "to"):
                comp.to(self.device)
                self.logger.info("component_moved_to_gpu", name=name)

    # ── Quantization ─────────────────────────────────────────────────────

    def _quantize_text_encoders(self) -> None:
        """Quantize frozen text encoders (standalone, called before TE caching).

        Uses ``te_quantization`` config.  Skipped if TEs are being
        trained (``train_text_encoder=True``).  Sets
        ``self._te_quantization_applied`` to prevent double-quantization.

        When ``store_quantized_version`` is enabled, checks for a disk-
        cached version first and saves after fresh quantization.
        Cached models are stored under ``backend/models/.quantized/``.
        """
        if getattr(self, "_te_quantization_applied", False):
            return

        from app.engine.factories.quantization import QuantizationFactory

        te_quant = self.config.get("te_quantization", "none")
        te_backend = self.config.get("te_quantization_backend", "auto")
        train_te = self.config.get("train_text_encoder", False)
        store = self.config.get("store_quantized_version", True)
        definition_id = self.definition.id if hasattr(self, "definition") else None
        root_path = getattr(self.loader, "_root_path", None) if hasattr(self, "loader") else None

        if te_quant != "none" and not train_te:
            te_backend, te_quant = QuantizationFactory.validate_and_fallback(te_quant, te_backend)
            if te_quant == "none":
                original = self.config.get("te_quantization", "none")
                if getattr(self, "_log_writer", None):
                    self._log_writer.warning(
                        f"TE quantization '{original}' not available on this GPU — running without quantization"
                    )
            elif te_quant != "none":
                for name, te in self._get_text_encoders().items():
                    # Resolve source path for this TE component
                    source_path = os.path.join(root_path, name) if root_path else None
                    if source_path and not os.path.exists(source_path):
                        source_path = root_path  # flat layout fallback

                    # Try disk cache first
                    if store and definition_id:
                        cache_path = QuantizationFactory.resolve_cache_path(definition_id, name, te_quant)
                        cached = QuantizationFactory.load_quantized(te, cache_path, te_quant, source_path=source_path)
                        if cached is not None:
                            self.components[name] = cached
                            setattr(self, name, cached)
                            continue

                    te_vram = QuantizationFactory.estimate_vram(te, te_quant, te_backend)
                    self.logger.info("quantizing_te", name=name, backend=te_backend, scheme=te_quant, **te_vram)
                    quantized_te = QuantizationFactory.quantize(te, te_quant, backend_name=te_backend)
                    self.components[name] = quantized_te
                    setattr(self, name, quantized_te)

                    # Save to disk cache
                    if store and definition_id:
                        cache_path = QuantizationFactory.resolve_cache_path(definition_id, name, te_quant)
                        try:
                            QuantizationFactory.save_quantized(quantized_te, cache_path, te_quant, source_path=source_path)
                        except Exception as e:
                            self.logger.warning("quantized_cache_save_failed", name=name, error=str(e))

        self._te_quantization_applied = True

        # Re-sync family-specific aliases (e.g. clip_encoder, t5_encoder)
        # so downstream code uses quantized versions, not the originals.
        self._assign_components()

        # Free old BF16 weight tensors replaced by quantize_()
        gc.collect()
        torch.cuda.empty_cache()

    def _quantize_primary_model(self) -> None:
        """Quantize frozen primary model (UNet/Transformer, standalone).

        Uses ``quantization`` config.  Sets ``self._model_quantization_applied``
        to prevent double-quantization.

        When ``store_quantized_version`` is enabled, checks for a disk-
        cached version first and saves after fresh quantization.
        Cached models are stored under ``backend/models/.quantized/``.
        """
        if getattr(self, "_model_quantization_applied", False):
            return

        from app.engine.factories.quantization import QuantizationFactory

        model = self._get_primary_model()
        quant_scheme = self.config.get("quantization", "none")
        quant_backend = self.config.get("quantization_backend", "auto")
        store = self.config.get("store_quantized_version", True)
        definition_id = self.definition.id if hasattr(self, "definition") else None
        root_path = getattr(self.loader, "_root_path", None) if hasattr(self, "loader") else None

        if quant_scheme != "none" and model is not None:
            quant_backend, quant_scheme = QuantizationFactory.validate_and_fallback(quant_scheme, quant_backend)
            if quant_scheme == "none":
                original = self.config.get("quantization", "none")
                if getattr(self, "_log_writer", None):
                    self._log_writer.warning(
                        f"Model quantization '{original}' not available on this GPU — running without quantization"
                    )
            elif quant_scheme != "none":
                comp_name = "transformer" if hasattr(self, "transformer") else "unet"

                # Resolve source path for fingerprinting
                source_path = os.path.join(root_path, comp_name) if root_path else None
                if source_path and not os.path.exists(source_path):
                    source_path = root_path  # flat layout fallback

                # Blackwell FP8 training is a runtime module swap (nn.Linear →
                # Float8Linear), not a weight compression — caching doesn't apply.
                # Ada/Hopper FP8 is weight-only and benefits from disk cache.
                from app.engine.factories.quantization import _is_blackwell
                is_cacheable = not (quant_scheme == "fp8" and _is_blackwell())

                # Try disk cache first (weight-only schemes only)
                if is_cacheable and store and definition_id:
                    cache_path = QuantizationFactory.resolve_cache_path(definition_id, comp_name, quant_scheme)
                    cached = QuantizationFactory.load_quantized(model, cache_path, quant_scheme, source_path=source_path)
                    if cached is not None:
                        self._update_primary_model(cached)
                        self._model_quantization_applied = True
                        return

                vram_est = QuantizationFactory.estimate_vram(model, quant_scheme, quant_backend)
                self.logger.info("quantizing_model", backend=quant_backend, scheme=quant_scheme, **vram_est)
                quantized = QuantizationFactory.quantize(model, quant_scheme, backend_name=quant_backend)
                self._update_primary_model(quantized)

                # Save to disk cache (weight-only schemes only)
                if is_cacheable and store and definition_id:
                    cache_path = QuantizationFactory.resolve_cache_path(definition_id, comp_name, quant_scheme)
                    try:
                        QuantizationFactory.save_quantized(quantized, cache_path, quant_scheme, source_path=source_path)
                    except Exception as e:
                        self.logger.warning("quantized_cache_save_failed", error=str(e))

        self._model_quantization_applied = True

        # Free old BF16 weight tensors replaced by quantize_()
        gc.collect()
        torch.cuda.empty_cache()

    def _quantize_components(self) -> None:
        """Quantize frozen UNet/Transformer and text encoders.

        Backward-compatible wrapper — calls the targeted methods.
        Skipped if individual methods were already called by the
        orchestrator during early quantization phases.
        """
        self._quantize_primary_model()
        self._quantize_text_encoders()

    # ── Staged VRAM Management Hooks ─────────────────────────────────────

    def _offload_vae(self) -> None:
        """Move VAE to CPU after latent pre-caching — frees VRAM.

        Gated on ``low_vram`` (default: True).  All families benefit from
        this since VAE is unused after latents are cached.
        """
        if not self.config.get("low_vram", True):
            return
        vae = getattr(self, "vae", None)
        if vae is not None:
            vae.to("cpu")
            self.logger.info("vae_offloaded_to_cpu")
            torch.cuda.empty_cache()
            gc.collect()

    def _offload_text_encoders(self) -> None:
        """Move or unload text encoders after embedding caching.

        Uses :meth:`_get_text_encoders` to discover TE modules, then
        either **unloads** them (``unload_text_encoder=True``) for max
        VRAM savings, or **offloads** to CPU.

        **Critically**, this also removes TE entries from
        ``self.components`` to prevent stale references from keeping
        unloaded TEs in memory and from resurecting them during phased
        sampling.

        Subclasses may override for additional cleanup (e.g. tokenizers
        with non-standard attribute names), but should call ``super()``
        to get the component dict cleanup and gc.
        """
        if self._te_unloaded:
            return
        if not self.config.get("cache_text_embeddings", True):
            return

        te_dict = self._get_text_encoders()
        if not te_dict:
            return

        cache_count = len(getattr(self, "text_cache", {}))

        if self.config.get("unload_text_encoder", False):
            self.logger.info(
                "unloading_text_encoders",
                encoders=list(te_dict.keys()),
                cached_embeddings=cache_count,
            )
            # Remove from components dict — prevents stale refs & GC leak
            for name in te_dict:
                self.components.pop(name, None)
                # Zero out the instance attribute if it matches the
                # component dict key (works for Z-Image, QwenImage,
                # SDXL, Flux2; Flux1 overrides for its custom names)
                if hasattr(self, name):
                    setattr(self, name, None)
            self._te_unloaded = True
        else:
            self.logger.info(
                "text_encoders_offloaded_to_cpu",
                encoders=list(te_dict.keys()),
                cached_embeddings=cache_count,
            )
            for te in te_dict.values():
                if te is not None:
                    te.to("cpu")
            # Also clean components dict so _ensure_on_gpu won't
            # find stale entries after a future unload
            for name in te_dict:
                self.components.pop(name, None)

        torch.cuda.empty_cache()
        gc.collect()
