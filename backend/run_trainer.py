
# --------------------------------------------------------------------------------------------------
# CRITICAL: This script is the entry point for all training jobs.
# Output is written to {output_dir}/job_log.jsonl via the JobLogWriter.
# The backend's LogTailer reads that file to bridge logs to the UI.
# --------------------------------------------------------------------------------------------------

# Prevent WinError 1314: Windows symlink permission errors in HF Hub cache.
# Must be set BEFORE any huggingface_hub imports.
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# CUDA caching-allocator config — MUST be set before torch initialises CUDA
# (read lazily at first CUDA use, so setting it here, before any app/torch
# import below, is in time). Set HERE in the trainer entry point (not only in
# the launching plugin) so it takes effect on every trainer launch even when
# the backend server wasn't restarted to pick up new plugin code.
#   - expandable_segments:True → one growable segment per stream reused across
#     tensor shapes, instead of a fresh fixed segment per shape. Aspect-ratio
#     bucketing trains many distinct latent shapes (e.g. 448x576 … 1888x1056);
#     without this the reserved pool fragments per new shape until it spills
#     past physical VRAM into Windows/WDDM shared memory (a slow "freeze").
# NOTE: garbage_collection_threshold was REMOVED — measured peak-allocated is
# only ~67 GB (fits the card) but it fired at 75 GB, below the ~88 GB reserved
# plateau, doing synchronous mid-step GC that freed + re-fragmented cached
# blocks every step. That churn caused the run-to-run plateau variance (78↔95 GB
# on the same checkpoint) and fought expandable_segments' stable single segment.
# setdefault → an explicit user/parent value is respected.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import traceback
import argparse
import json
import asyncio

# FORCE UNBUFFERED OUTPUT IMMEDIATELY
# Keeps stdout working for local debug even though IPC now uses files.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# safety-net print: pre-init, _log_writer is not constructed until main()
print("Trainer process started...", flush=True)

try:
    import logging  # noqa: F401
    from app.engine.models.registry import registry
    # Use the centralized logging setup to ensure JSON output and consistent formatting
    from app.core.logger import setup_logging, get_logger
    
    # Configure logging immediately
    # Critical: Disable file handler (`include_file_handler=False`) to avoid file locking contentions on Windows.
    # The worker logs only to STDOUT, which the JobManager captures and bridges to the main server log.
    setup_logging("INFO", include_file_handler=False)
    logger = get_logger("trainer")

    # Apply Hugging Face auth so gated-model downloads in this subprocess
    # authenticate. An inherited env token (HF_TOKEN) wins; otherwise the
    # token saved in Server → Models is used. Never logs the token value.
    from app.core.hf_auth import apply_hf_auth
    from app.engine.utils.model_override_manager import ModelOverrideManager
    apply_hf_auth(ModelOverrideManager.get_all().hf_token)
except ImportError:
    # safety-net print: pre-init, import-failure handler runs before logging setup completes
    print("CRITICAL: Failed to import dependencies. Check PYTHONPATH and venv.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception:
    # safety-net print: pre-init, unexpected-import-error handler runs before logging setup completes
    print("CRITICAL: Unexpected error during imports.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


# ── JobLogWriter integration ─────────────────────────────────────────────

_log_writer = None  # Module-level reference for the logging handler bridge


def _resolve_output_dir(config: dict, definition_id: str) -> str:
    """Resolve the training output directory from config.

    Must match the path logic used by the backend's JobManager
    (``_get_job_output_dir``) and the trainers' CheckpointManager.
    """
    from app.core.naming import model_part_from_definition_id

    output_root = config.get("output_dir", "outputs")
    lora_name = config.get("lora_name", "untitled")
    model_part = model_part_from_definition_id(definition_id)
    run_name = f"{lora_name}_{model_part}"
    return os.path.join(output_root, run_name)


class _LogWriterHandler(logging.Handler):
    """Bridge: forwards Python logging records to the JobLogWriter.

    Attached to the root logger so that all structlog output appearing
    on stdout is *also* written to the job_log.jsonl file.
    """

    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:
                self.writer.log(msg)
        except Exception:
            self.handleError(record)


# Note: We do NOT use logging.basicConfig anymore as setup_logging handles it.

async def run_async_trainer(trainer, log_writer=None):
    """
    Helper to run async trainer methods properly.
    
    Phased orchestration for minimal peak VRAM.
    
    The flow supports two quantization strategies (configurable via
    ``quantization_strategy`` config):
    
    **fastest** (default, best for >= 24 GB VRAM):
        1. Load all → CPU
        2. TE → GPU → quantize → stay on GPU → cache embeddings → offload
        3. VAE → GPU → cache latents → offload
        4. Model → GPU → freeze → quantize → PEFT → optimizer → train
    
    **vram_safe** (for <= 16 GB VRAM):
        1. Load all → CPU
        2. TE → GPU → quantize → offload to CPU
        3. TE → GPU (quantized, small) → cache embeddings → offload
        4. VAE → GPU → cache latents → offload
        5. Model → GPU → freeze → quantize → PEFT → optimizer → train
    """
    try:
        trainer.config.get("quantization_strategy", "fastest")

        # Inject the log writer into the trainer so components can use it
        if log_writer:
            trainer._log_writer = log_writer

        # ── Phase A: Load all components to CPU ──────────────────────
        _emit_status("Checking Model", log_writer)
        await trainer.setup()
        
        _emit_status("Loading Model", log_writer)
        await trainer.load_model()
        
        _emit_status("Preparing Data", log_writer)
        await trainer.prepare_data()

        # ── TE Quantization + Caching Phase ──────────────────────────
        te_names = list(trainer._get_text_encoders().keys())
        te_quant = trainer.config.get("te_quantization", "none")
        has_te_quant = te_quant != "none" and not trainer.config.get("train_text_encoder", False)

        if has_te_quant and te_names:
            _emit_status("Quantizing Text Encoders", log_writer)
            trainer._quantize_text_encoders()
            for name in te_names:
                trainer._move_component_to_gpu(name)
        elif te_names:
            for name in te_names:
                trainer._move_component_to_gpu(name)

        if trainer.get_te_cache():
            _emit_status("TE Cache Restored", log_writer)
        trainer._pre_cache_text_embeddings()

        trainer._offload_text_encoders()

        # ── Latent (VAE) caching phase ───────────────────────────────
        trainer._move_component_to_gpu("vae")

        trainer._validate_latent_cache()
        await trainer._pre_cache_latents()
        # Second-modality cache (LTX-2 audio latents) — runs while the audio VAE
        # is still resident; no-op for every other family.
        trainer._pre_cache_aux()

        trainer._offload_vae()

        # Notify parent process that cache directories exist
        try:
            ds_configs = trainer.config.get("datasets", [])
            ds_names = list({
                (d if isinstance(d, str) else d.get("dataset_name", ""))
                for d in ds_configs
            } - {""})
            if ds_names:
                if log_writer:
                    log_writer.emit("cache_ready", ds_names)
        except Exception:
            pass  # Non-critical

        # ── Phase B: Prepare UNet for training ───────────────────────
        _emit_status("Preparing Training", log_writer)
        await trainer.prepare_for_training()

        # ── Train ────────────────────────────────────────────────────
        _emit_status("Training", log_writer)
        await trainer.train()
    except Exception as e:
        # safety-net print: fallback, async-trainer wrapper exception; _log_writer may not exist
        print(f"CRITICAL: Async Trainer Exception: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _emit_status(label: str, log_writer=None):
    """Emit status via the JobLogWriter file-based IPC channel."""
    if log_writer:
        log_writer.status(label)


def _finalize_before_exit(log_writer=None) -> None:
    """Drain pending GPU work and report leaked children before exit.

    Two failure modes this addresses:

    1. **Background CUDA stream still running**: an exception that fires
       while async kernels are in flight can race the ``_log_writer.exit``
       write — the parent reports exit code 0 while the GPU is still busy
       and the JobManager removes the job from the active queue even
       though work is ongoing.  ``torch.cuda.synchronize()`` blocks until
       all pending kernels complete so the reported exit reflects reality.

    2. **Orphaned child processes** (DataLoader workers, HF download
       helpers, multiprocessing pools): on Windows the trainer is launched
       with ``CREATE_NEW_PROCESS_GROUP``; any surviving child keeps the
       GPU pinned but the JobManager's PID watchdog only follows the
       parent.  We log a warning listing surviving children so the next
       repro shows exactly which subsystem leaked, then attempt to
       terminate them so the GPU actually frees.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass  # Never block exit on diagnostics

    try:
        import psutil
        children = psutil.Process().children(recursive=True)
        if children:
            child_info = [
                {"pid": c.pid, "name": c.name(), "status": c.status()}
                for c in children
            ]
            msg = f"leaked_child_processes_at_exit: {child_info}"
            if log_writer:
                log_writer.warning(msg)
            for c in children:
                try:
                    c.terminate()
                except Exception:
                    pass
            gone, alive = psutil.wait_procs(children, timeout=3)
            for c in alive:
                try:
                    c.kill()
                except Exception:
                    pass
    except Exception:
        pass


def main():
    logger.info("trainer_entry_point_reached")
    
    parser = argparse.ArgumentParser(description="MRLN Arcane Tuner Training Entry Point")
    parser.add_argument("--definition_id", type=str, required=True, help="Model Definition ID")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration string")
    
    args = parser.parse_args()
    
    global _log_writer
    
    try:
        config = json.loads(args.config)
        logger.info(f"loaded_config: definition_id={args.definition_id} config_keys={list(config.keys())}")
        
        # ── Initialise file-based IPC log writer ─────────────────────
        output_dir = _resolve_output_dir(config, args.definition_id)
        
        from app.engine.components.job_log_writer import JobLogWriter
        _log_writer = JobLogWriter(output_dir)
        
        # Bridge Python logging → JobLogWriter so structlog output
        # also appears in job_log.jsonl for the LogTailer.
        handler = _LogWriterHandler(_log_writer)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(handler)
        
        _log_writer.log(f"Trainer started: definition_id={args.definition_id}")
        
        # Initialize Registry
        registry.initialize()
        
        definition = registry.get_definition(args.definition_id)
        
        if not definition:
             logger.error(f"definition_not_found: {args.definition_id}")
             _log_writer.exit(1, error=f"Definition ID '{args.definition_id}' not found")
             sys.exit(1)
             
        logger.info(f"resolved_family: {definition.family}")
        
        # Registry-driven dispatch: every family's ``family.py`` registers a
        # ``ModelFamily`` subclass with ``family_name`` matching the YAML
        # ``family`` field.  ``ModelRegistry.discover_families`` (already
        # run via ``registry.initialize()`` above) imports those modules so
        # this lookup succeeds for any family present in
        # ``app/engine/models/families/``.
        try:
            family_cls = registry.get_family_class(definition.family)
        except ValueError as exc:
            logger.error("trainer_dispatch_failed", family=definition.family, error=str(exc))
            _log_writer.exit(1, error=str(exc))
            sys.exit(1)

        family_instance = family_cls(definition, config)
        TrainerClass = family_instance.get_trainer_class()
        logger.info(
            "instantiating_trainer",
            family=definition.family,
            trainer_class=TrainerClass.__name__,
        )
        trainer = TrainerClass(definition=definition, run_config=config)
        logger.info("starting_async_training", family=definition.family)
        asyncio.run(run_async_trainer(trainer, _log_writer))

        logger.info("training_completed_successfully")
        _finalize_before_exit(_log_writer)
        _log_writer.exit(0)
        sys.exit(0)

    except Exception as e:
        # safety-net print: fallback, outermost exception handler; _log_writer may not exist
        print("CRITICAL: Unhandled exception in main execution block.", file=sys.stderr)
        traceback.print_exc()
        # Bridge the FULL traceback into the job log so it surfaces in the UI.
        # The trainer is a detached subprocess: traceback.print_exc() above only
        # reaches trainer_stdout.log on disk, while the UI tails job_log.jsonl —
        # without this the operator sees just the one-line exit error.
        if _log_writer:
            _log_writer.log_exception(e)
        _finalize_before_exit(_log_writer)
        if _log_writer:
            _log_writer.exit(1, error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
