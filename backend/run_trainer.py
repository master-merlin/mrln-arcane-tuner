
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

import sys
import traceback
import argparse
import json
import asyncio

# FORCE UNBUFFERED OUTPUT IMMEDIATELY
# Keeps stdout working for local debug even though IPC now uses files.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
except ImportError:
    print("CRITICAL: Failed to import dependencies. Check PYTHONPATH and venv.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception:
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
    output_root = config.get("output_dir", "outputs")
    lora_name = config.get("lora_name", "untitled")
    model_part = definition_id.split("/")[-1].replace(":", "_")
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
                # Keep stdout marker for backward compat with any pipe readers
                import json as _json
                print(f"[CACHE_READY:{_json.dumps(ds_names)}]", flush=True)
        except Exception:
            pass  # Non-critical

        # ── Phase B: Prepare UNet for training ───────────────────────
        _emit_status("Preparing Training", log_writer)
        await trainer.prepare_for_training()

        # ── Train ────────────────────────────────────────────────────
        _emit_status("Training", log_writer)
        await trainer.train()
    except Exception as e:
        print(f"CRITICAL: Async Trainer Exception: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _emit_status(label: str, log_writer=None):
    """Emit status to both the log writer and stdout (backward compat)."""
    if log_writer:
        log_writer.status(label)
    print(f"[STATUS:{label}]", flush=True)


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
             print(f"CRITICAL: Definition ID '{args.definition_id}' not found in registry.", file=sys.stderr)
             _log_writer.exit(1, error=f"Definition ID '{args.definition_id}' not found")
             sys.exit(1)
             
        logger.info(f"resolved_family: {definition.family}")
        
        if definition.family == "sdxl":
            from app.engine.models.families.sdxl.trainer import SDXLTrainer as TrainerClass
            
            logger.info("instantiating_sdxl_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            logger.info("starting_async_v2_training")
            asyncio.run(run_async_trainer(trainer, _log_writer))
            
        elif definition.family == "flux2":
            from app.engine.models.families.flux2.trainer import Flux2Trainer as TrainerClass
            
            logger.info("instantiating_flux2_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            logger.info("starting_async_flux2_training")
            asyncio.run(run_async_trainer(trainer, _log_writer))
            
        elif definition.family == "flux1":
            from app.engine.models.families.flux1.trainer import Flux1Trainer as TrainerClass
            
            logger.info("instantiating_flux1_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            logger.info("starting_async_flux1_training")
            asyncio.run(run_async_trainer(trainer, _log_writer))
            
        elif definition.family == "zimage":
            from app.engine.models.families.zimage.trainer import ZImageTrainer as TrainerClass
            
            logger.info("instantiating_zimage_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            logger.info("starting_async_zimage_training")
            asyncio.run(run_async_trainer(trainer, _log_writer))
            
        elif definition.family == "qwen_image":
            from app.engine.models.families.qwen_image.trainer import QwenImageTrainer as TrainerClass
            
            logger.info("instantiating_qwen_image_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            logger.info("starting_async_qwen_image_training")
            asyncio.run(run_async_trainer(trainer, _log_writer))
            
        else:
            logger.info("using_legacy_trainer")
            logger.warning("legacy_trainer_path_incomplete_aborting", family=definition.family)
            print(f"CRITICAL: No trainer registered for family '{definition.family}'. Aborting.", file=sys.stderr)
            _log_writer.exit(1, error=f"No trainer for family '{definition.family}'")
            sys.exit(1)

        logger.info("training_completed_successfully")
        _log_writer.exit(0)
        sys.exit(0)

    except Exception as e:
        print("CRITICAL: Unhandled exception in main execution block.", file=sys.stderr)
        traceback.print_exc()
        if _log_writer:
            _log_writer.exit(1, error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
