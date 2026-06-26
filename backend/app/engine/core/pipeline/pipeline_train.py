"""
Pipeline Train Mixin — the main training loop with gradient accumulation.
"""

import os
import random
import time
import uuid
from typing import Any

import structlog
import torch

from app.engine.factories.optimizer import OptimizerFactory

logger = structlog.get_logger(__name__)


def _sliding_window_frames(target_frames: int, temporal_downscale: int) -> int:
    """Per-step sliding window length in LATENT frames for a target frame count."""
    from app.engine.components.latents import LatentManager

    return LatentManager.latent_frames(int(target_frames), int(temporal_downscale))


class _RegionTimer:
    """Wall-clock + CUDA-event timer for one named per-step region.

    Records host wall time and a CUDA start/end event pair (resolved with a
    single ``synchronize`` at window end) so a region's GPU-busy time and its
    host-side idle (wall - gpu) can be apportioned without kineto/torch.profiler
    (whose stop()/export deadlocks on cu130 + Windows). Accumulates into the
    owner's ``_region_wall`` / ``_region_count`` / ``_region_gpu_events``.
    """

    def __init__(self, owner: "PipelineTrainMixin", name: str) -> None:
        self._owner = owner
        self._name = name
        self._t0 = 0.0
        self._e0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            self._e0 = torch.cuda.Event(enable_timing=True)
            self._e0.record()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self._t0
        o = self._owner
        o._region_wall[self._name] = o._region_wall.get(self._name, 0.0) + wall
        o._region_count[self._name] = o._region_count.get(self._name, 0) + 1
        if self._e0 is not None:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            o._region_gpu_events.append((self._name, self._e0, e1))
        return False


class PipelineTrainMixin:
    """Main training loop with gradient accumulation, checkpointing, and sampling."""

    # ── Optional lightweight step profiler (diagnostic; inert unless armed) ──
    #
    # Gated by ``profile_steps`` in the run config. When unset/0 every hook is a
    # no-op and the loop is byte-identical. When > 0 it times the per-step regions
    # ("data_prep" = cached-latent load + H2D + text encode; "forward_loss" = the
    # autocast compute) over ``profile_steps`` steps after a short warmup, using
    # wall-clock + CUDA events (NOT kineto/torch.profiler, which deadlocks on
    # stop()/export with cu130 on Windows). Writes a region breakdown — avg wall
    # vs GPU vs idle — so the per-step GPU-idle is apportioned, then stops.

    def _maybe_init_profiling(self) -> None:
        """Arm the step profiler from config (fully inert when ``profile_steps`` <= 0)."""
        self._profiling_live = False
        self._profiling_active = int(self.config.get("profile_steps", 0) or 0)
        self._region_wall: dict[str, float] = {}
        self._region_count: dict[str, int] = {}
        self._region_gpu_events: list = []  # (name, start_event, end_event)
        if self._profiling_active <= 0:
            return
        self._profiling_warmup = int(self.config.get("profile_warmup", 3) or 0)
        import os

        self._profile_dir = self.config.get("profile_dir") or os.path.join(
            str(self.checkpoint_manager.output_dir), "profile"
        )
        os.makedirs(self._profile_dir, exist_ok=True)
        self.logger.info(
            "profiling_armed",
            active=self._profiling_active,
            warmup=self._profiling_warmup,
            dir=self._profile_dir,
        )

    def _prof_region(self, name: str):
        """A wall+CUDA-event timing span when profiling is live, else a no-op context."""
        if not getattr(self, "_profiling_live", False):
            from contextlib import nullcontext

            return nullcontext()
        return _RegionTimer(self, name)

    def _profiling_maybe_begin(self, step: int) -> None:
        """Go live once the warmup boundary is reached (no-op otherwise)."""
        if (
            getattr(self, "_profiling_active", 0) <= 0
            or getattr(self, "_profiling_live", False)
            or step < self._profiling_warmup
        ):
            return
        self._profiling_live = True
        self.logger.info("profiling_started", at_step=step)

    def _profiling_maybe_end(self, step: int) -> bool:
        """Write the region report once the active window is captured.

        Returns True when profiling just finished so the caller can break the
        training loop (a profile run does not need to train to completion).
        """
        if not getattr(self, "_profiling_live", False):
            return False
        if step + 1 < self._profiling_warmup + self._profiling_active:
            return False
        self._profiling_live = False
        self._write_profile_report(step)
        return True

    def _write_profile_report(self, step: int) -> None:
        import os

        # Resolve per-region GPU time from the recorded CUDA events (single sync).
        region_gpu: dict[str, float] = {}
        if torch.cuda.is_available() and self._region_gpu_events:
            torch.cuda.synchronize()
            for name, e0, e1 in self._region_gpu_events:
                region_gpu[name] = region_gpu.get(name, 0.0) + e0.elapsed_time(e1) / 1000.0

        header = (
            f"# Step profile: {self._profiling_active} steps after "
            f"{self._profiling_warmup} warmup - family={self.__class__.__name__}"
        )
        cols = "# region            avg_wall_ms   avg_gpu_ms   avg_idle_ms   count"
        lines = [header, cols]
        for name in self._region_wall:
            cnt = max(self._region_count.get(name, 1), 1)
            wall_ms = self._region_wall[name] / cnt * 1000.0
            gpu_ms = region_gpu.get(name, 0.0) / cnt * 1000.0
            lines.append(
                f"{name:<18}{wall_ms:>11.1f}{gpu_ms:>13.1f}{wall_ms - gpu_ms:>13.1f}{cnt:>8}"
            )
        body = "\n".join(lines) + "\n"

        path = os.path.join(self._profile_dir, "profile_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        self.logger.info("profiling_report_written", path=path)
        for ln in lines:
            self.logger.info("profile_row", row=ln)
        lw = getattr(self, "_log_writer", None)
        if lw:
            lw.log("Step profile:\n" + body)

    async def train(self):
        """Execute the main training loop with gradient accumulation."""
        self.logger.info("starting_training_loop", family=self.__class__.__name__)
        # Surface the active allocator config + VRAM safety valve so a job log
        # makes it obvious whether the anti-fragmentation settings are in effect
        # (they only take effect when the trainer process was launched with them
        # — e.g. after a backend restart for the plugin-injected path).
        if torch.cuda.is_available():
            try:
                self.logger.info(
                    "vram_config",
                    alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
                    allocator_backend=torch.cuda.get_allocator_backend(),
                )
            except Exception:  # noqa: BLE001
                pass
        self._get_primary_model().train()

        max_steps = int(self.config.get("max_train_steps", 1000))
        self.max_train_steps = max_steps
        batch_size = int(self.config.get("train_batch_size", 1))

        # Sigma distribution tracker — accumulates timestep histogram for diagnostics.
        try:
            from app.engine.strategies.sigma_tracker import SigmaTracker
            self._sigma_tracker = SigmaTracker()
        except Exception:  # noqa: BLE001
            self._sigma_tracker = None
        grad_accum = int(self.config.get("gradient_accumulation_steps", 1))
        noise_offset_strength = float(self.config.get("noise_offset", 0.0))

        # Bucket-aware infinite iterator: groups items by target resolution
        # so torch.stack works when batch_size > 1.
        #
        # VRAM-safe warmup (default on): training order stays FULLY RANDOM, but
        # the FIRST batch of each epoch — and the first batch after each sample —
        # is forced to the largest bucket. Rationale: the CUDA caching allocator
        # grows its reserved pool in allocation-arrival order; with pure random
        # order, small buckets grow it incrementally and a later large bucket
        # strands those blocks → ~20-30 GB of order-dependent fragmentation that
        # varies run-to-run and can spill past the card (a "freeze"). Reserving
        # the largest bucket's peak segment FIRST means every later (smaller)
        # step just reuses it, no stranding — so only one warmup step per epoch
        # is needed and the rest of the order is untouched. The sampler calls
        # empty_cache (to fit the TE/denoise), which releases the pool, so we
        # re-warm on the first step after each sample via ``_vram_rewarm_pending``.
        from collections import defaultdict

        vram_safe_order = bool(self.config.get("vram_safe_bucket_order", True))

        def _bucket_key(item: dict) -> tuple[int, int, int]:
            return (
                item["target_w"],
                item["target_h"],
                item.get("target_frames", 1),
            )

        # Items of the single largest bucket (by pixel/voxel count) — the peak.
        _largest_items: list = []
        if self.inventory:
            _bk = defaultdict(list)
            for _it in self.inventory:
                _bk[_bucket_key(_it)].append(_it)
            _largest_key = max(_bk, key=lambda k: k[0] * k[1] * k[2])
            _largest_items = _bk[_largest_key]

        def _warm_batch() -> list:
            """One batch from the largest bucket (reserves the peak segment)."""
            return random.sample(
                _largest_items, min(batch_size, len(_largest_items))
            )

        def _random_batches() -> list[list]:
            """One epoch of batches, fully random. BS>1 groups by bucket so
            collated shapes stay uniform; BS<=1 is a flat shuffle."""
            if batch_size <= 1:
                pool = list(self.inventory)
                random.shuffle(pool)
                return [[it] for it in pool]
            random.shuffle(self.inventory)
            buckets: dict[tuple[int, int, int], list] = defaultdict(list)
            for it in self.inventory:
                buckets[_bucket_key(it)].append(it)
            batches: list[list] = []
            for items in buckets.values():
                for i in range(0, len(items), batch_size):
                    batches.append(items[i : i + batch_size])
            random.shuffle(batches)
            return batches

        def get_iterator():
            if not vram_safe_order:
                while True:
                    yield from _random_batches()
                return
            self._vram_rewarm_pending = True  # warm once at the very start
            while True:
                for batch in _random_batches():
                    # Warm the largest bucket first at epoch start / after a
                    # sample's empty_cache, so the peak segment is reserved
                    # before any smaller batch can fragment the pool.
                    if getattr(self, "_vram_rewarm_pending", False) and _largest_items:
                        self._vram_rewarm_pending = False
                        yield _warm_batch()
                    yield batch
                self._vram_rewarm_pending = True  # re-warm at the next epoch

        data_iter = get_iterator()

        # Virtual epoch tracking
        steps_per_epoch = max(1, len(self.inventory) // batch_size)
        self._steps_per_epoch = steps_per_epoch

        # Signal Manager
        from app.engine.components.signal_manager import TrainingSignalManager

        self.signal_manager = TrainingSignalManager(self.checkpoint_manager.output_dir)
        self.logger_component.signal_manager = self.signal_manager

        # Inject the file-based log writer into the logger component
        _lw = getattr(self, "_log_writer", None)
        if _lw:
            self.logger_component.log_writer = _lw
            # Cascade to the sampler so its progress messages reach the UI
            # via the file-based IPC channel (see
            # docs/superpowers/specs/2026-05-17-ipc-migration-design.md).
            if getattr(self, "sampler", None) is not None:
                self.sampler._log_writer = _lw

        # Write initial training log
        trainable_comps = self._build_trainable_components()
        self.checkpoint_manager._write_training_log(
            step=self.global_step,
            components=trainable_comps,
            optimizer=self.optimizer,
            config=self.config,
            is_final=False,
            elapsed_time=0.0,
            lora_filename="(not yet saved)",
        )

        # ── Create job history record in SQLite ──
        self._job_history_id = self._init_job_history(max_steps, grad_accum)
        self.logger_component._job_id = self._job_history_id

        # ── Log trainable vs total parameter counts ──
        primary = self._get_primary_model()
        trainable_params = sum(
            p.numel() for p in primary.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in primary.parameters())
        self._trainable_params = trainable_params
        self._total_params = total_params
        if _lw:
            _lw.log(
                f"Trainable {trainable_params:,} of {total_params:,} params "
                f"({trainable_params / max(total_params, 1) * 100:.2f}%)"
            )

        is_adaptive = OptimizerFactory.is_adaptive(
            self.config.get("optimizer_type", "AdamW8bit"),
            config=self.config,
        )

        # Step-0 baseline sample: generate a sample BEFORE any training
        # to show the model's starting point.  Controlled by config.
        if (
            self.sampler
            and self.config.get("sample_before_training", True)
            and self.global_step == 0
        ):
            self.logger.info("generating_step0_baseline_sample")
            self._emit_status("Sampling")
            try:
                self.sampler.generate_samples(step=-1)  # step -1 → displayed as step 0
            except Exception as e:
                self.logger.error("step0_sampling_failed", error=str(e))
                self._emit_warning(f"Step-0 sampling failed: {e}")
            finally:
                self._emit_status("Training")

        # Tell the logger how often we save so it can project save overhead in ETA
        save_every_cfg = int(self.config.get("save_every_n_steps", 0))
        if save_every_cfg > 0:
            self.logger_component._save_every = save_every_cfg

        start_step = self.global_step + 1 if self.global_step > 0 else 0
        sps_window: list[float] = []  # Sliding window for Samples/s smoothing

        # Snapshot the caching/load peak, then reset CUDA peak stats so the loop
        # below measures the training-phase peak (calibrates the VRAM wall).
        self._begin_training_vram_window()

        # Arm the optional profiler window (no-op unless profile_steps > 0).
        self._maybe_init_profiling()

        for step in range(start_step, max_steps):
            self.global_step = step

            # Start the profiler once warmup steps have elapsed (no-op otherwise).
            self._profiling_maybe_begin(step)

            # ── Signal check ──
            signal_action = self.signal_manager.handle_signals()
            if signal_action == "soft_stop":
                self.logger.info("soft_stop_saving_checkpoint", step=step)
                self._emit_status("Saving Checkpoint")
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    components=self._build_trainable_components(),
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    scaler=self.scaler,
                    config=self.config,
                    ema_handler=self.ema_handler,
                    elapsed_time=self.logger_component.get_total_elapsed(),
                    te_cache=self.get_te_cache(),
                )
                self.logger.info("soft_stop_checkpoint_saved", step=step)
                break

            # ── Gradient accumulation loop ──
            self.optimizer.zero_grad()
            accumulated_loss = 0.0
            grad_norm = torch.tensor(0.0)

            for accum_idx in range(grad_accum):
                batch_items = next(data_iter)
                # Video families pay a heavy per-clip PyAV decode that a warm
                # latent cache turns into pure waste (the pixels are discarded
                # for the cached latent), starving the GPU. Defer their decode
                # and re-run it only on a cache miss; image/pixel families (incl.
                # the pixel-space ones that read batch["images"] in forward)
                # decode upfront, byte-identical to before.
                defer_decode = bool(getattr(self, "is_video_family", False))
                batch = self._get_batch(batch_items, decode_pixels=not defer_decode)

                # 1. Encode Latents
                #    Profiler region "data_prep": cached-latent disk load + the
                #    host→device copy + text encode — the per-step data cost that
                #    scales with clip size (the suspected video-vs-image util gap).
                with torch.no_grad(), self._prof_region("data_prep"):
                    use_cache = self.config.get("cache_latents", True)
                    # Per-item cache discriminators (e.g. a video clip's trim
                    # window) — present only for video batches. Splat only when
                    # set so pixel-space families whose passthrough overrides
                    # don't accept the kwarg are unaffected, and image batches
                    # keep byte-identical legacy cache paths.
                    extra_keys = batch.get("extra_keys")
                    ek_kwarg = {"extra_keys": extra_keys} if extra_keys else {}
                    latents = None
                    if use_cache:
                        bi0 = batch_items[0] if batch_items else {}
                        if bi0.get("temporal_mode") == "sliding":
                            window_lf = _sliding_window_frames(
                                int(bi0.get("target_frames", 1)),
                                self.latent_manager.temporal_downscale(),
                            )
                            latents = (
                                self.latent_manager.load_cached_latent_windows(
                                    batch["ids"],
                                    batch["cache_dirs"],
                                    source_paths=batch["paths"],
                                    window_frames=window_lf,
                                    **ek_kwarg,
                                )
                            )
                        else:
                            latents = self.latent_manager.load_cached_latents(
                                batch["ids"],
                                batch["cache_dirs"],
                                source_paths=batch["paths"],
                                **ek_kwarg,
                            )
                    if latents is None:
                        # Cache miss — should only happen if pre-cache was
                        # skipped or new items were added mid-run. Decode the
                        # pixels now if we deferred them above (video families),
                        # then encode. Re-decode uses the same variant-selected
                        # paths so the latent it writes is keyed identically.
                        if batch.get("images") is None:
                            batch["images"] = self._decode_batch_images(
                                batch_items, batch["paths"]
                            )
                        self._uncached_encode_count = (
                            getattr(self, "_uncached_encode_count", 0) + 1
                        )
                        self.logger.warning(
                            "latent_cache_miss",
                            step=step,
                            ids=batch["ids"],
                            miss_count=self._uncached_encode_count,
                        )
                        latents = self.latent_manager.encode_and_cache_batch(
                            batch["images"],
                            ids=batch["ids"],
                            cache_dirs=batch["cache_dirs"] if use_cache else None,
                            source_paths=batch["paths"],
                            **ek_kwarg,
                        )
                    latents = latents.to(self.device, dtype=self.autocast_dtype)

                    # ── Flip augmentation (applied on latent tensor) ──
                    if self._aug_h_flip and random.random() < 0.5:
                        latents = torch.flip(latents, dims=[-1])
                    if self._aug_v_flip and random.random() < 0.5:
                        latents = torch.flip(latents, dims=[-2])

                    # ── Clean control latents (paired edit runs) ──
                    # Loaded after the target's flip aug so controls are never
                    # flipped. No-op for non-edit batches; the family forward
                    # consumes batch["control_latents"] when present.
                    self._load_control_latents(batch)

                    # Pre-noise conditioning (e.g. WAN/LTX i2v first-frame latent).
                    self._attach_conditioning(batch, latents)

                    # 2. Encode Text (family hook). Pass the batch so paired-edit
                    # trainers can condition the text encoder on the control image
                    # + key the TE cache by (caption, control). Ignored otherwise.
                    text_emb = self.encode_text(
                        batch["captions"],
                        self.autocast_dtype,
                        batch=batch,
                    )

                # 3. Forward + Loss (under autocast)
                #    Profiler region "forward_loss": the compute (noise/pack/
                #    timestep/forward/loss). Backward + optimizer get their own
                #    regions below so the full step is apportioned.
                with torch.autocast(
                    "cuda", dtype=self.autocast_dtype, enabled=self.use_amp
                ), self._prof_region("forward_loss"):
                    # Sample noise in SPATIAL space [B,C,H,W] (before any packing).
                    # This is critical: noise offset must be per-channel in spatial
                    # space, not in packed [B,L,D] space where D interleaves
                    # 2×2 patch positions and would amplify the offset ~4×.
                    noise = torch.randn_like(latents)

                    # Noise offset (https://www.crosslabs.org/blog/diffusion-with-offset-noise)
                    if noise_offset_strength > 0:
                        # Per-channel offset in spatial space: [B, C, 1, 1]
                        noise_offset = noise_offset_strength * torch.randn(
                            latents.shape[0],
                            latents.shape[1],
                            *([1] * (latents.ndim - 2)),
                            device=self.device,
                            dtype=latents.dtype,
                        )
                        noise = noise + noise_offset

                    # Pack/reshape latents AND noise together (family hook).
                    # Both must go through the same packing to stay aligned.
                    prepared_latents = self.prepare_latents_for_training(latents)
                    prepared_noise = self.prepare_noise_for_training(noise)

                    # Timesteps (family hook)
                    timesteps = self.sample_timesteps(
                        prepared_latents.shape[0], latents
                    )

                    # Track sigma distribution (non-fatal diagnostic).
                    try:
                        if self._sigma_tracker is not None:
                            self._sigma_tracker.update(timesteps)
                    except Exception:  # noqa: BLE001
                        pass

                    # Add noise (family hook)
                    noisy_input = self.add_noise(
                        prepared_latents, prepared_noise, timesteps
                    )

                    # Forward pass (family hook)
                    pred = self.forward_pass(noisy_input, timesteps, text_emb, batch)

                    # Target (family hook)
                    target = self.compute_target(
                        prepared_latents, prepared_noise, timesteps
                    )

                    # Loss — family hook allows full override (e.g. pixel-space
                    # families like HiDream-O1 that bypass the latent/noise path
                    # and compute their own recipe loss in forward_pass).
                    loss = self._compute_step_loss(
                        pred,
                        target,
                        timesteps,
                        batch,
                        grad_accum,
                    )

                # 4. Backward
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count = getattr(self, "nan_count", 0) + 1
                    self.logger.warning(
                        "nan_loss_detected", step=step, consecutive_nan=self.nan_count
                    )
                    if self.nan_count >= 10:
                        self.logger.error(
                            "training_aborted_nan",
                            message="10 consecutive NaN losses — aborting training",
                        )
                        raise RuntimeError(
                            "Training aborted: 10 consecutive NaN losses"
                        )
                    continue  # Skip this accum step
                else:
                    self.nan_count = 0
                    accumulated_loss += loss.item() * grad_accum  # Unscale for logging

                    with self._prof_region("backward"):
                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

            # 5. Optimizer step (after all accumulation steps)
            #    Profiler region "optimizer": unscale + grad-norm clip + step.
            with self._prof_region("optimizer"):
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                # Compute grad norm for ALL optimizers (monitoring).
                # Only clip for non-adaptive; adaptive optimizers manage their own.
                if is_adaptive:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._get_primary_model().parameters(),
                        max_norm=float("inf"),
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._get_primary_model().parameters(), max_norm=1.0
                    )

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 6. EMA step
            if self.ema_handler:
                self.ema_handler.step()

            # 6a. Driver per-step hook (default no-op). Multi-model families
            # (WAN 2.2 dual-expert MoE) advance their router here → set the
            # active expert for the NEXT step (and swap it onto the GPU). Wrapped
            # in getattr so this stays safe for any driver/trainer shape.
            driver = getattr(self, "driver", None)
            if driver is not None and hasattr(driver, "on_optimizer_step"):
                driver.on_optimizer_step(step)

            # 6b. First completed optimizer step: optimizer states are now
            # allocated and gradients are live — snapshot the resident set so
            # _compute_vram_measured can isolate optimizer + activations.
            if "after_first_step" not in getattr(self, "_vram_snapshots", {}):
                self._snapshot_vram("after_first_step")

            # 7. Logging  — emit rich per-step diagnostics
            extra: dict[str, Any] = {}

            # Grad norm (always available now)
            extra["grad_norm"] = round(float(grad_norm), 6)

            # Learning rate (d_estimate for Prodigy, raw LR otherwise)
            if (
                hasattr(self.optimizer, "param_groups")
                and "d" in self.optimizer.param_groups[0]
            ):
                d_est = float(self.optimizer.param_groups[0]["d"])
                current_lr = d_est * self.optimizer.param_groups[0]["lr"]
                extra["d_estimate"] = round(d_est, 8)
            else:
                raw_lr = self.optimizer.param_groups[0].get("lr")
                if raw_lr is not None:
                    current_lr = float(raw_lr)
                elif is_adaptive:
                    import math

                    t = max(step + 1, 1)
                    current_lr = min(1e-6 * t, 1.0 / math.sqrt(t))
                else:
                    current_lr = 0.0

            # Timestep distribution (populates the previously-dead DB column)
            extra["timestep_mean"] = round(float(timesteps.float().mean()), 3)

            # Epoch progress
            extra["epoch"] = round((step + 1) / self._steps_per_epoch, 2)

            # Batch resolution (from last accumulation batch)
            if batch.get("target_w") and batch.get("target_h"):
                extra["resolution"] = f"{batch['target_w']}x{batch['target_h']}"

            # NaN event counter (early warning before abort)
            nan_count = getattr(self, "nan_count", 0)
            if nan_count > 0:
                extra["nan_count"] = nan_count

            # GradScaler scale factor (detects skipped steps / instability)
            if self.scaler.is_enabled():
                extra["amp_scale"] = float(self.scaler.get_scale())

            # Throughput: samples per second (SMA smoothed)
            step_time = time.time() - self.logger_component.last_step_time
            if step_time > 0:
                raw_sps = batch_size * grad_accum / max(step_time, 0.001)
                sps_window.append(raw_sps)
                if len(sps_window) > 10:  # 10-step simple moving average
                    sps_window.pop(0)

                extra["samples_per_sec"] = round(sum(sps_window) / len(sps_window), 2)

            # Live VRAM usage
            if torch.cuda.is_available():
                extra["vram_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / 1024**2
                )
                extra["vram_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / 1024**2
                )
                # Diagnostic: cumulative PEAK *allocated* (live tensors) vs the
                # reserved pool. Distinguishes a genuine large working set from
                # caching-allocator fragmentation: if peak_alloc plateaus far
                # below reserved, the gap is fragmentation; if it tracks reserved,
                # the workload truly needs that much. Monotonic (no reset) so it
                # doesn't disturb the end-of-run peak telemetry.
                extra["vram_peak_alloc_mb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**2
                )

            self.logger_component.log_step(
                step,
                accumulated_loss / grad_accum if grad_accum > 1 else accumulated_loss,
                current_lr,
                extra=extra,
            )

            # Persist current step to job_history every step so the UI's
            # "Step X / Y" counter ticks live. Previously this only ran
            # inside the periodic-save block, which meant the DB lagged by
            # up to `save_every_n_steps - 1` steps. The actual write is a
            # single integer update — cheap enough to run every step.
            self._update_job_progress(step)

            # 8. Periodic save
            save_every = int(self.config.get("save_every_n_steps", 0))
            if save_every > 0 and step > 0 and step % save_every == 0:
                self.logger.info("periodic_checkpoint_saving", step=step)
                self._emit_status("Saving Checkpoint")
                self.logger_component.pause_step_timer()
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    components=self._build_trainable_components(),
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    scaler=self.scaler,
                    config=self.config,
                    ema_handler=self.ema_handler,
                    elapsed_time=self.logger_component.get_total_elapsed(),
                    te_cache=self.get_te_cache(),
                    cache_manifest=self._build_cache_manifest(),
                )
                self.logger_component.resume_step_timer()
                self.logger.info(
                    "periodic_checkpoint_saved",
                    step=step,
                    save_time=round(self.logger_component._save_times[-1], 2),
                    avg_save_time=round(self.logger_component.avg_save_time, 2),
                )
                # Emit sigma distribution summary alongside the checkpoint log.
                try:
                    if self._sigma_tracker is not None:
                        self.logger.info(
                            "sigma_distribution", **self._sigma_tracker.summary()
                        )
                except Exception:  # noqa: BLE001
                    pass
                # Record checkpoint in DB
                self._record_checkpoint(step, is_final=False)
                # Flush metrics buffer + update progress
                self.logger_component.flush_metrics()
                self._update_job_progress(step)
                self._emit_status("Training")

            # 8b. Sample generation (independent from checkpoint save interval)
            if self.sampler and self.sampler.should_sample(step):
                self.logger.info("sampling_triggered", step=step)
                # Broadcast status via file-based IPC (LogTailer reads job_log.jsonl)
                self._emit_status("Sampling")
                try:
                    self.sampler.generate_samples(step)
                except Exception as e:
                    self.logger.error("sampling_failed", step=step, error=str(e))
                    self._emit_warning(f"Sampling failed at step {step + 1}: {e}")
                finally:
                    self._emit_status("Training")
                    # Sampling ran empty_cache → re-warm the largest bucket on the
                    # next step so random order can't re-fragment the freed pool.
                    self._vram_rewarm_pending = True

            # 9. Virtual epoch boundary
            if step > 0 and step % steps_per_epoch == 0:
                epoch_num = step // steps_per_epoch
                self.on_epoch_end(epoch_num)

            # 10. Profiler window complete → write report and stop early
            #     (a profile run does not need to train to completion).
            if self._profiling_maybe_end(step):
                self.logger.info("profiling_window_complete_stopping", step=step)
                self._emit_status("Profiling complete")
                return

        # ── Training complete ──
        self.logger.info("training_finished")

        # Capture training-phase VRAM peaks NOW — before the final checkpoint
        # save allocates extra memory (EMA copy, safetensors buffers) that would
        # otherwise inflate the measured "training" peak.
        self._capture_training_peaks()
        self._vram_measured = self._compute_vram_measured()

        self.logger_component.save_loss_history(self.checkpoint_manager.output_dir)

        self.logger.info("saving_final_checkpoint")
        self._emit_status("Saving Final Checkpoint")
        self.checkpoint_manager.save_checkpoint(
            step=max_steps,
            components=self._build_trainable_components(),
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            scaler=self.scaler,
            config=self.config,
            ema_handler=self.ema_handler,
            is_final=True,
            elapsed_time=self.logger_component.get_total_elapsed(),
            te_cache=self.get_te_cache(),
            cache_manifest=self._build_cache_manifest(),
        )
        self._emit_status("Training")

        # Record final checkpoint + complete job history
        self._record_checkpoint(max_steps, is_final=True)
        self._complete_job_history(max_steps)

        # Final sample — generate unless the last step already produced one
        if self.sampler:
            last_step = max_steps - 1
            already_sampled = self.sampler.should_sample(last_step)
            if already_sampled:
                self.logger.info(
                    "skipping_final_sample_already_generated",
                    step=max_steps,
                )
            else:
                self.logger.info("generating_final_sample", step=max_steps)
                self._emit_status("Sampling")
                try:
                    self.sampler.generate_samples(last_step, final=True)
                except Exception as e:
                    self.logger.error(
                        "final_sampling_failed", step=max_steps, error=str(e)
                    )
                    self._emit_warning(f"Final sampling failed: {e}")
                finally:
                    self._emit_status("Training")

    # ── File-based IPC helpers ────────────────────────────────────────

    def _emit_status(self, label: str) -> None:
        """Emit a status label via the file-based JobLogWriter IPC channel."""
        _lw = getattr(self, "_log_writer", None)
        if _lw:
            _lw.status(label)

    def _emit_warning(self, message: str) -> None:
        """Emit a warning via the file-based JobLogWriter IPC channel."""
        _lw = getattr(self, "_log_writer", None)
        if _lw:
            _lw.warning(message)

    # ── Job History Helpers ──────────────────────────────────────────

    def _init_job_history(self, max_steps: int, grad_accum: int) -> str | None:
        """Create a job_history record at training start."""
        try:
            from app.core.db import DatabaseEngine
            from app.core.db.repositories.job_repo import JobHistoryRepository

            db = DatabaseEngine.get_instance()
            db.initialize()

            config = dict(self.config) if self.config else {}
            job_id = config.get("job_id")
            repo = JobHistoryRepository()

            payload = {
                "status": "running",
                "started_at": time.time(),
                "total_steps": max_steps,
                "output_dir": self.checkpoint_manager.output_dir,
                "network_rank": int(config.get("network_rank", 0)) or None,
                "network_alpha": int(config.get("network_alpha", 0)) or None,
                "optimizer_type": config.get("optimizer_type"),
                "learning_rate": float(config.get("learning_rate", 0)) or None,
                "lr_scheduler": config.get("lr_scheduler"),
                "timestep_sampling": config.get("timestep_sampling"),
                "batch_size": int(config.get("train_batch_size", 1)),
                "grad_accum": grad_accum,
                "quantization": config.get("quantization"),
                "mixed_precision": config.get("mixed_precision"),
                "ema_enabled": bool(config.get("use_ema", False)),
                "targeted_layers": config.get("targeted_layers"),
            }

            if job_id:
                repo.update_status(job_id, **payload)
            else:
                job_id = str(uuid.uuid4())
                payload["id"] = job_id
                payload["lora_name"] = config.get("lora_name", "")
                payload["definition_id"] = config.get("definition_id", "")
                payload["config"] = config
                payload["created_at"] = time.time()
                payload["datasets_config"] = [
                    {
                        "dataset_name": ds.get("dataset_name", ""),
                        "num_repeats": ds.get("num_repeats", 1),
                        "masking_enabled": ds.get("masking_enabled", False),
                        "caption_dropout": float(ds.get("caption_dropout_rate", 0)),
                    }
                    for ds in config.get("datasets", [])
                ]
                repo.create(payload)

            logger.info("job_history_running", job_id=job_id)
            return job_id
        except Exception as e:
            logger.warning("job_history_init_failed", error=str(e))
            return None

    def _record_checkpoint(self, step: int, is_final: bool = False) -> None:
        """Record a checkpoint save in the DB."""
        if not getattr(self, "_job_history_id", None):
            return
        try:
            from app.core.db.repositories.checkpoint_repo import CheckpointRepository

            repo = CheckpointRepository()
            repo.add(
                {
                    "job_id": self._job_history_id,
                    "step": step,
                    "path": self.checkpoint_manager.output_dir,
                    "is_final": is_final,
                    "loss_at_step": self.logger_component._loss_history[-1]["loss"]
                    if self.logger_component._loss_history
                    else None,
                    "lr_at_step": self.logger_component._loss_history[-1]["lr"]
                    if self.logger_component._loss_history
                    else None,
                }
            )
        except Exception as e:
            logger.warning("checkpoint_record_failed", step=step, error=str(e))

    def _update_job_progress(self, step: int) -> None:
        """Update job_history with current progress."""
        if not getattr(self, "_job_history_id", None):
            return
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository

            repo = JobHistoryRepository()
            repo.update_progress(self._job_history_id, completed_steps=step + 1)
        except Exception as e:
            logger.warning("job_progress_update_failed", error=str(e))

    def _complete_job_history(self, max_steps: int) -> None:
        """Finalize the job_history record."""
        if not getattr(self, "_job_history_id", None):
            return
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository

            repo = JobHistoryRepository()

            elapsed = self.logger_component.get_total_elapsed()
            history = self.logger_component._loss_history
            losses = [h["loss"] for h in history] if history else []

            # Measured VRAM (captured pre-save) + total on-disk footprint.
            vram_measured = getattr(self, "_vram_measured", None)
            peak_train = getattr(self, "_peak_vram_train_mb", None)
            peak_cache = getattr(self, "_peak_vram_cache_mb", None)
            total_bytes = self._measure_run_disk_bytes()

            # Persist the full per-component breakdown alongside the LoRA, in
            # training_log.json (written during the final save just above). The
            # DB keeps only the compact peak scalars for fast calibration reads.
            if vram_measured:
                self._write_vram_measured(vram_measured)

            repo.complete(
                self._job_history_id,
                completed_steps=max_steps,
                completed_epochs=round(max_steps / self._steps_per_epoch, 2)
                if getattr(self, "_steps_per_epoch", 0)
                else None,
                duration_seconds=elapsed,
                training_seconds=elapsed - self.logger_component._total_save_time,
                avg_loss=sum(losses) / len(losses) if losses else None,
                min_loss=min(losses) if losses else None,
                avg_step_time=elapsed / max_steps if max_steps > 0 else None,
                avg_save_time=self.logger_component.avg_save_time or None,
                peak_vram_train_mb=peak_train,
                peak_vram_cache_mb=peak_cache,
                total_run_bytes=total_bytes,
            )
            logger.info("job_history_completed", job_id=self._job_history_id)

            # Refresh this definition's estimation coefficients (best-effort).
            try:
                from app.core.stats import definition_stats_service

                definition_stats_service.recompute(
                    self.config.get("definition_id") or None
                )
            except Exception as e:
                logger.warning("definition_stats_recompute_failed", error=str(e))
        except Exception as e:
            logger.warning("job_history_complete_failed", error=str(e))

    # ── Measured VRAM (per-component, for estimation-wall calibration) ──

    def _snapshot_vram(self, label: str) -> None:
        """Record current allocated VRAM (MB) under a lifecycle label."""
        if not torch.cuda.is_available():
            return
        if not hasattr(self, "_vram_snapshots"):
            self._vram_snapshots: dict[str, float] = {}
        try:
            self._vram_snapshots[label] = torch.cuda.memory_allocated() / 1024**2
        except Exception:
            pass

    def _begin_training_vram_window(self) -> None:
        """Snapshot the caching/load peak, then reset CUDA peak stats.

        After this, ``max_memory_*`` reflect only the training phase.
        """
        self._peak_vram_cache_mb = None
        if not torch.cuda.is_available():
            return
        try:
            self._peak_vram_cache_mb = round(torch.cuda.max_memory_reserved() / 1024**2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._peak_vram_cache_mb = None

    def _capture_training_peaks(self) -> None:
        """Record training-phase peaks (call BEFORE the final save)."""
        self._peak_vram_train_mb = None
        self._peak_vram_train_alloc_mb = None
        if not torch.cuda.is_available():
            return
        try:
            self._peak_vram_train_mb = round(torch.cuda.max_memory_reserved() / 1024**2)
            self._peak_vram_train_alloc_mb = round(
                torch.cuda.max_memory_allocated() / 1024**2
            )
        except Exception:
            pass

    def _compute_vram_measured(self) -> dict | None:
        """Decompose measured GPU memory into Training VRAM Wall components.

        Uses lifecycle snapshots (model load → adapter inject → first optimizer
        step) plus the training-phase allocated/reserved peaks. Components are
        deltas; activations are the spike above the resident set; overhead is
        the allocator/context gap (reserved − allocated). Approximate by design
        — it calibrates the analytic estimate, it is not an accounting ledger.
        """
        snaps = getattr(self, "_vram_snapshots", None)
        peak_reserved = getattr(self, "_peak_vram_train_mb", None)
        peak_alloc = getattr(self, "_peak_vram_train_alloc_mb", None)
        if not snaps or peak_alloc is None:
            return None
        model = snaps.get("model")
        if model is None:
            return None
        resident_adapters = snaps.get("adapters", model)
        after_first = snaps.get("after_first_step")

        model_weights = round(model)
        adapters = round(max(resident_adapters - model, 0))
        # Gradients ≈ trainable params at adapter precision (bf16/fp16 = 2 bytes)
        trainable = getattr(self, "_trainable_params", 0) or 0
        gradients = round(trainable * 2 / 1024**2)
        if after_first is not None:
            optimizer_states = round(
                max(after_first - resident_adapters - gradients, 0)
            )
            resident_set = after_first
        else:
            optimizer_states = 0
            resident_set = resident_adapters + gradients
        activations = round(max(peak_alloc - resident_set, 0))
        overhead = round(max((peak_reserved or peak_alloc) - peak_alloc, 0))

        return {
            "model_weights_mb": model_weights,
            "lora_adapters_mb": adapters,
            "optimizer_states_mb": optimizer_states,
            "gradients_mb": gradients,
            "activations_mb": activations,
            "overhead_mb": overhead,
            "training_peak_mb": peak_reserved or peak_alloc,
            "caching_peak_mb": getattr(self, "_peak_vram_cache_mb", None),
            "peak_allocated_mb": peak_alloc,
        }

    def _write_vram_measured(self, measured: dict) -> None:
        """Merge the measured VRAM breakdown into the run's training_log.json."""
        try:
            import json
            import os

            out_dir = getattr(self.checkpoint_manager, "output_dir", None)
            if not out_dir:
                return
            path = os.path.join(out_dir, "training_log.json")
            data = {}
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["vram_measured"] = measured
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("vram_measured_write_failed", error=str(e))

    def _measure_run_disk_bytes(self) -> int | None:
        """Total bytes written under the run's output directory."""
        try:
            import os

            out_dir = getattr(self.checkpoint_manager, "output_dir", None)
            if not out_dir or not os.path.isdir(out_dir):
                return None
            total = 0
            for root, _dirs, files in os.walk(out_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        continue
            return total or None
        except Exception:
            return None
