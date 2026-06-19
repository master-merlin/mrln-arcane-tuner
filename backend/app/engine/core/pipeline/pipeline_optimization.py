"""
Pipeline Optimization Mixin — PEFT/LoRA, optimizer, LR scheduler, EMA, resume.

Handles Phase B (prepare_for_training) and all optimization-related setup.
"""

import os
from typing import Any

import structlog
import torch

from app.core.naming import model_part_from_definition_id
from app.engine.components.checkpoints import CheckpointManager
from app.engine.strategies.ema import EMAHandler
from app.engine.components.latents import LatentManager
from app.engine.factories.optimizer import OptimizerFactory, LRSchedulerFactory
from app.engine.components.training_logger import TrainingLogger
from peft import LoraConfig, get_peft_model

logger = structlog.get_logger(__name__)


def _merge_te_caches(
    base: dict[str, dict] | None, overlay: dict[str, dict] | None
) -> dict[str, dict]:
    """Union two TE-cache dicts (``{subcache: {caption: emb}}``); overlay wins.

    Used on resume: ``base`` is the restored checkpoint cache, ``overlay`` is the
    freshly-warmed cache for THIS run (captions + sample prompts + the CFG
    unconditional). Merging overlay-wins means new/changed sample prompts survive
    instead of being clobbered by the checkpoint cache — otherwise sampling hits
    the offloaded text encoder ("caption not pre-cached"). Same-caption entries
    are identical (deterministic TE), so overlay-wins is safe.
    """
    base = base or {}
    overlay = overlay or {}
    merged: dict[str, dict] = {}
    for sub in set(base) | set(overlay):
        merged[sub] = {**(base.get(sub) or {}), **(overlay.get(sub) or {})}
    return merged


class PipelineOptimizationMixin:
    """PEFT, optimizer, gradient checkpointing, EMA, checkpoint resume."""

    # ── Prepare for Training (Phase B — UNet to GPU) ─────────────────────

    async def prepare_for_training(self):
        """Prepare model for training (Phase B).

        Called by ``run_trainer.py`` **after** TE and VAE caching phases
        have completed and those components are offloaded.  This method:

        1. Move primary model (UNet/Transformer) to GPU
        2. Freeze all remaining components
        3. Quantize frozen base model
        4. Enable gradient checkpointing
        5. Apply PEFT / LoRA adapters
        6. Configure optimizer, LR scheduler, GradScaler
        7. EMA
        8. Initialize managers (latent, logger, checkpoint)
        9. Resume from checkpoint
        10. Create sampler
        """
        self.logger.info("preparing_for_training")

        # Reset measured-VRAM snapshots for this run (estimation-wall calibration).
        self._vram_snapshots = {}

        # 1-2. Freeze ALL components before quantization
        self._freeze_all()

        # 3. Quantize frozen components ON CPU (cache load or fresh)
        #    before moving to GPU so we never allocate full bf16 on the card.
        self._quantize_components()

        # 4. Move (now quantized / FP8) primary model to GPU
        self._move_component_to_gpu("unet")

        # Snapshot resident VRAM after the base model lands on GPU but before
        # adapters/optimizer — isolates "model weights" for the measured wall.
        self._snapshot_vram("model")

        # 4. Gradient checkpointing (MUST be before PEFT — PeftModel lacks
        #    gradient_checkpointing_enable; only the raw model exposes it)
        self._configure_gradient_checkpointing()

        # 5a. Fuse QKV projections (Flux2: to_q/k/v → to_qkv) before PEFT
        #     so PEFT trains a single shared lora_A per fused QKV layer
        self._fuse_qkv_projections()

        # 5b. PEFT / LoRA (inject adapters into Float8Linear modules)
        self._apply_peft()

        # Snapshot after adapters inject — delta vs "model" = LoRA adapter VRAM.
        self._snapshot_vram("adapters")

        # 6. torch.compile (AFTER PEFT so compile sees full Float8Linear + LoRA graph)
        self._compile_if_quantized()

        # 6b. Block swapping (CPU↔GPU migration for VRAM savings)
        self._configure_block_swapping()

        # 6c. Targeted layer training (freeze non-selected layers)
        self._configure_targeted_training()

        # 7. Optimizer, LR scheduler, GradScaler
        max_train_steps = int(self.config.get("max_train_steps", 1000))
        self._configure_optimization(max_train_steps)

        # 7. EMA
        self._configure_ema()

        # 8. Managers
        self._configure_managers(max_train_steps)

        # 9. Resume
        self._resume_if_needed()

        # 10. Sampler (for generating sample images during training)
        self.sampler = self._create_sampler()
        if self.sampler:
            self.logger.info("sampling_pipeline_initialized")

    # ── PEFT / LoRA ──────────────────────────────────────────────────────

    def _fuse_qkv_projections(self) -> None:
        """Fuse separate Q/K/V projections into single ``to_qkv`` modules.

        Flux2 double-stream blocks have separate ``to_q/k/v`` and
        ``add_q/k/v_proj`` linear layers.  When PEFT wraps them
        independently, each gets its own ``lora_A`` matrix.  At save time,
        merging three independent rank-r LoRAs into one fused QKV requires
        SVD re-decomposition that loses 6-17% of learned information.

        Calling ``fuse_projections()`` before PEFT creates a single
        ``to_qkv`` (and ``to_added_qkv``) linear layer, so PEFT trains
        one shared ``lora_A`` — zero conversion loss at save time.

        Gated per family via ``driver.should_fuse_qkv_projections()`` (default
        ``False``).  Families that target the unfused ``to_q``/``to_k``/``to_v``
        (LTX-2, SDXL, …) must NOT fuse: it would leave their LoRA targets
        matching nothing, and LTX-2's audio↔video cross-modal attentions have
        asymmetric query vs key/value dims (4096 vs 2048) that crash diffusers'
        ``fuse_projections``.  Only FLUX.2 opts in.
        """
        if not self.driver.should_fuse_qkv_projections():
            self.logger.debug("qkv_fusion_skipped", reason="family_opts_out")
            return

        model = self._get_primary_model()

        fused_count = 0
        for module in model.modules():
            if hasattr(module, 'fuse_projections') and hasattr(module, '_supports_qkv_fusion'):
                if module._supports_qkv_fusion and not getattr(module, 'fused_projections', False):
                    module.fuse_projections()
                    fused_count += 1

        if fused_count > 0:
            self.logger.info("qkv_projections_fused", count=fused_count)

    def _apply_peft(self) -> None:
        """Apply PEFT/LoRA to primary model and optionally text encoders."""
        self.logger.info("initializing_peft")

        model = self._get_primary_model()
        rank = int(self.config.get("network_rank", 16))
        alpha = float(self.config.get("network_alpha", rank))

        # Primary model LoRA
        targets = self.get_lora_targets()
        exclude = self.get_lora_exclude_modules()
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            exclude_modules=exclude,
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        self._update_primary_model(model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            "lora_applied", trainable_params=trainable, total_params=total,
            rank=rank, alpha=alpha, targets=len(targets),
        )

        # Text encoder LoRA (if requested)
        if self.config.get("train_text_encoder", False):
            te_targets = self.get_te_lora_targets()
            if te_targets:
                te_dict = self._get_text_encoders()
                for name, te in te_dict.items():
                    te_lora = LoraConfig(
                        r=rank, lora_alpha=alpha,
                        target_modules=te_targets,
                        lora_dropout=0.0, bias="none",
                    )
                    wrapped_te = get_peft_model(te, te_lora)
                    self.components[name] = wrapped_te
                    setattr(self, name, wrapped_te)
                    self.logger.info("lora_te_applied", name=name, targets=len(te_targets))

    # ── Gradient Checkpointing ───────────────────────────────────────────

    def _configure_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing if configured."""
        if not self.config.get("gradient_checkpointing", False):
            return
        self.logger.info("enabling_gradient_checkpointing")
        model = self._get_primary_model()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

        if self.config.get("train_text_encoder", False):
            for te in self._get_text_encoders().values():
                if hasattr(te, "gradient_checkpointing_enable"):
                    te.gradient_checkpointing_enable()

    # ── torch.compile for Float8 Training ──────────────────────────────────

    def _compile_if_quantized(self) -> None:
        """torch.compile the primary model when FP8 training mode is active.

        On Blackwell GPUs, ``_quantize_fp8`` replaces ``nn.Linear`` with
        ``Float8Linear`` and sets ``_fp8_training_mode = True``.
        ``torch.compile`` is required to fuse the scaling + cast + GEMM
        into efficient kernels — without it, the dynamic scaling overhead
        negates the FP8 speedup.

        Uses ``mode="default"`` (not ``reduce-overhead``) to avoid CUDA
        graph issues with variable input shapes during sampling.

        On non-Blackwell GPUs, weight-only FP8 is used instead and
        compile is skipped (no benefit for dequant-only path).

        Must be called **after** PEFT wrapping so compile sees the full
        Float8Linear + LoRA graph.
        """
        model = self._get_primary_model()
        if model is None:
            return

        # Check if the Blackwell FP8 training path was used
        # (flag set by _quantize_fp8 in quantization.py).
        # After PEFT wrapping, the flag lives on the inner model:
        # PeftModel → base_model → model.  Walk through wrappers.
        fp8_training = getattr(model, "_fp8_training_mode", False)
        if not fp8_training:
            inner = getattr(model, "base_model", None)
            if inner is not None:
                inner = getattr(inner, "model", inner)
                fp8_training = getattr(inner, "_fp8_training_mode", False)
        if not fp8_training:
            return

        self.logger.info("compiling_fp8_training_model")
        try:
            compiled = torch.compile(model, mode="default")
            self._update_primary_model(compiled)
            self.logger.info("model_compiled", mode="default")
        except Exception as e:
            self.logger.warning(
                "compile_failed_continuing_eager",
                error=str(e),
                hint="Training will proceed without compile — FP8 may be slower.",
            )

    # ── Block Swapping ───────────────────────────────────────────────────

    def _configure_block_swapping(self) -> None:
        """Apply CPU↔GPU block swapping if configured.

        Uses ``driver.get_block_topology()`` to resolve block groups,
        then applies ``BlockSwappingManager`` to the configured number
        of blocks per group.

        Config keys (per block group name):
            ``blocks_to_swap.<group_name>`` — integer count of blocks to
            swap for that group (default 0 = no swap).
        """
        topology = self.driver.get_block_topology()
        if not topology:
            return

        swap_config = self.config.get("block_swap_config", {})
        if not swap_config:
            return

        from app.engine.core.optimization.block_swapping import BlockSwappingManager

        model = self._get_primary_model()
        if model is None:
            return

        for group in topology:
            pct = int(swap_config.get(group["name"], 0))
            if pct <= 0:
                continue

            attr = getattr(model, group["attr_path"], None)
            if attr is None:
                continue

            total_blocks = len(list(attr))
            count = round(total_blocks * pct / 100)
            if count <= 0:
                continue

            # Swap the first N blocks (deepest blocks first = most savings)
            blocks_to_swap = list(attr)[:count]
            manager = BlockSwappingManager(blocks_to_swap, device=self.device)
            manager.apply()

            # Store manager for cleanup on end
            if not hasattr(self, "_block_swap_managers"):
                self._block_swap_managers = []
            self._block_swap_managers.append(manager)

            self.logger.info(
                "block_swapping_configured",
                group=group["name"],
                pct=pct,
                swapped=count,
                total=total_blocks,
                approx_vram_saved_mb=count * group["approx_vram_mb"],
            )

    # ── Targeted Layer Training ──────────────────────────────────────────

    def _configure_targeted_training(self) -> None:
        """Apply selective layer freezing if configured.

        Config keys:
            ``targeted_layers`` — list of regex patterns.  Only parameters
            matching at least one pattern remain trainable.
        """
        patterns = self.config.get("targeted_layers", [])
        if not patterns:
            return

        from app.engine.core.optimization.targeted_training import TargetedLayerManager

        model = self._get_primary_model()
        if model is None:
            return

        manager = TargetedLayerManager(patterns)
        manager.apply(model)

    # ── Optimizer + Scheduler + Scaler ───────────────────────────────────

    def _configure_optimization(self, max_train_steps: int) -> None:
        """Set up optimizer, LR scheduler, and GradScaler."""
        lr = float(self.config.get("learning_rate", 1e-4))

        # Adaptive batch-size LR scaling (Goyal et al. 2017)
        scale_mode = self.config.get("lr_scale_mode", "none")
        if scale_mode != "none":
            batch_size = int(self.config.get("train_batch_size", 1))
            grad_accum = int(self.config.get("gradient_accumulation_steps", 1))
            effective_bs = batch_size * grad_accum
            if effective_bs > 1:
                if scale_mode == "batch":
                    lr = lr * effective_bs
                elif scale_mode == "sqrt":
                    import math
                    lr = lr * math.sqrt(effective_bs)
                self.logger.info(
                    "lr_scaled", mode=scale_mode,
                    effective_batch_size=effective_bs,
                    base_lr=float(self.config.get("learning_rate", 1e-4)),
                    scaled_lr=lr,
                )

        optimizer_type = self.config.get("optimizer_type", "AdamW8bit")
        weight_decay = float(self.config.get("weight_decay", 0.01))
        betas = (
            float(self.config.get("beta1", 0.9)),
            float(self.config.get("beta2", 0.999)),
        )

        # Collect trainable params
        params = [p for p in self._get_primary_model().parameters() if p.requires_grad]
        if self.config.get("train_text_encoder", False):
            for te in self._get_text_encoders().values():
                params += [p for p in te.parameters() if p.requires_grad]

        self.optimizer = OptimizerFactory.create(
            optimizer_type, params, lr, weight_decay,
            betas=betas,
            config=self.config,
        )

        # LR Scheduler (skip for adaptive)
        is_adaptive = OptimizerFactory.is_adaptive(optimizer_type, config=self.config)
        if is_adaptive:
            self.lr_scheduler = None
            self.logger.info("scheduler_skipped_adaptive_optimizer", optimizer=optimizer_type)
        else:
            warmup = int(self.config.get("lr_warmup_steps", 0))
            sched_type = self.config.get("lr_scheduler", "constant")
            self.lr_scheduler = LRSchedulerFactory.create(
                sched_type, self.optimizer, warmup, max_train_steps
            )

        # GradScaler — use driver precision spec
        mixed_prec = self.config.get("mixed_precision", "fp16")
        prec = self.driver.get_precision_spec(
            mixed_prec, is_adaptive_optimizer=is_adaptive,
        )

        self.autocast_dtype = prec.autocast_dtype
        self.use_amp = prec.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=prec.grad_scaler_enabled)

        self.logger.info(
            "precision_configured",
            mixed_precision=mixed_prec,
            autocast_dtype=str(self.autocast_dtype),
            scaler_enabled=self.scaler.is_enabled(),
        )

    # ── EMA ───────────────────────────────────────────────────────────────

    def _configure_ema(self) -> None:
        """Initialize EMA handler if configured."""
        if not self.config.get("ema", False):
            return
        decay = float(self.config.get("ema_decay", 0.999))
        self.logger.info("initializing_ema", decay=decay)
        self.ema_handler = EMAHandler(self._get_primary_model(), decay=decay)

    # ── Managers ──────────────────────────────────────────────────────────

    def _configure_managers(self, max_train_steps: int) -> None:
        """Set up TrainingLogger and CheckpointManager.

        Note: LatentManager is initialized earlier in prepare_data()
        so it is available for _validate_latent_cache / _pre_cache_latents.
        """
        # Re-create LatentManager only if it wasn't created yet (safety net)
        if not hasattr(self, "latent_manager") or self.latent_manager is None:
            vae = self.components.get("vae")
            self.latent_manager = LatentManager(
                vae, device=self.device,
                arch_params=getattr(self.definition, "architecture_params", None),
            )
        self.logger_component = TrainingLogger(max_steps=max_train_steps)

        # Checkpoint path
        output_root = self.config.get("output_dir", "outputs")
        lora_name = self.config.get("lora_name", "lora")
        model_part = model_part_from_definition_id(self.definition.id)
        run_name = f"{lora_name}_{model_part}"
        final_output_dir = os.path.join(output_root, run_name)

        saver = self.driver.get_saver()
        self.checkpoint_manager = CheckpointManager(
            output_dir=final_output_dir, saver_impl=saver
        )
        self.components["config"] = self.config
        self.logger.info("checkpoint_path_configured", path=final_output_dir)

    # ── Resume ────────────────────────────────────────────────────────────

    def _resume_if_needed(self) -> None:
        """Resume from checkpoint if configured."""
        resume_path = self.config.get("resume_from_checkpoint")
        if not resume_path:
            return

        self.logger.info("resuming_from_checkpoint", path=resume_path)
        try:
            from peft import PeftModel

            peft_comps: dict[str, Any] = {}
            model = self._get_primary_model()
            if isinstance(model, PeftModel):
                peft_comps["unet"] = model
            for name, te in self._get_text_encoders().items():
                if isinstance(te, PeftModel):
                    peft_comps[name] = te

            checkpoint_state = self.checkpoint_manager.load_checkpoint(
                resume_path,
                peft_components=peft_comps,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                scaler=self.scaler,
                ema_handler=self.ema_handler,
                current_config=self.config,
            )
            self.global_step = checkpoint_state.global_step
            self.logger_component.elapsed_offset = checkpoint_state.elapsed_time
            self.logger_component.step_offset = checkpoint_state.global_step
            # Apply merged config (checkpoint base + user overrides like
            # max_train_steps, learning_rate, etc.) so subsequent saves
            # persist the correct values.
            if checkpoint_state.config:
                self.config.update(checkpoint_state.config)
            # Restore text embedding cache from checkpoint, MERGED over the
            # freshly-warmed cache so NEW/changed sample prompts (warmed this run)
            # survive. A plain restore replaces text_cache → a resumed run with a
            # different sample prompt hits the offloaded TE at sample time
            # ("caption not pre-cached").
            if checkpoint_state.te_cache:
                self.set_te_cache(
                    _merge_te_caches(checkpoint_state.te_cache, self.get_te_cache())
                )
            self.logger.info(
                "resumed_at_step",
                step=self.global_step,
                elapsed_offset=checkpoint_state.elapsed_time,
                adapters=checkpoint_state.adapters_loaded,
                components=checkpoint_state.components_loaded,
            )

            # Recreate LR scheduler for remaining steps so cosine/linear
            # decay correctly over the remaining training period.
            # Without this, the scheduler has num_training_steps=total but
            # only steps through remaining, traversing only a fraction of
            # the curve (e.g. 25% of cosine → barely decays).
            if self.global_step > 0 and self.lr_scheduler is not None:
                max_steps = int(self.config.get("max_train_steps", 1000))
                remaining = max(max_steps - self.global_step, 1)
                warmup = int(self.config.get("lr_warmup_steps", 0))
                sched_type = self.config.get("lr_scheduler", "constant")
                self.lr_scheduler = LRSchedulerFactory.create(
                    sched_type, self.optimizer, warmup, remaining,
                )
                self.logger.info(
                    "scheduler_recreated_for_resume",
                    scheduler=sched_type,
                    remaining_steps=remaining,
                    resumed_step=self.global_step,
                )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            self.logger.error("failed_to_resume_checkpoint", error=str(e))
            raise

    # ── Trainable Components ─────────────────────────────────────────────

    def _build_trainable_components(self) -> dict[str, Any]:
        """Collect all trainable model components for checkpointing."""
        comps: dict[str, Any] = {"unet": self._get_primary_model()}
        if self.config.get("train_text_encoder", False):
            for name, te in self._get_text_encoders().items():
                comps[name] = te
        return comps
