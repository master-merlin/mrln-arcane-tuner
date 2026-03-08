
# --------------------------------------------------------------------------------------------------
# CRITICAL: This script is the entry point for all training jobs.
# Any print() statement here is captured by the JobManager and bridged to the server log.
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
# This ensures that prints are flushed to the parent process (JobManager) instantly.
# Without this, prints might be buffered and lost if the process crashes.
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

# Note: We do NOT use logging.basicConfig anymore as setup_logging handles it.

async def run_async_trainer(trainer):
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

        # ── Phase A: Load all components to CPU ──────────────────────
        print("[STATUS:Checking Model]", flush=True)
        await trainer.setup()
        
        print("[STATUS:Loading Model]", flush=True)
        await trainer.load_model()
        
        print("[STATUS:Preparing Data]", flush=True)
        await trainer.prepare_data()

        # ── TE Quantization + Caching Phase ──────────────────────────
        #
        # Key: quantize / load FP8 cache BEFORE moving to GPU.
        # This avoids the wasteful bf16→GPU→FP8 transition that
        # doubles peak VRAM (bf16 + FP8 coexist on GPU).
        # Flow:  CPU bf16 → CPU FP8 (cache or fresh) → GPU FP8
        te_names = list(trainer._get_text_encoders().keys())
        te_quant = trainer.config.get("te_quantization", "none")
        has_te_quant = te_quant != "none" and not trainer.config.get("train_text_encoder", False)

        if has_te_quant and te_names:
            # Quantize on CPU first (cache load or fresh quantization)
            print("[STATUS:Quantizing Text Encoders]", flush=True)
            trainer._quantize_text_encoders()
            # Now move the (already quantized / FP8) TEs to GPU
            for name in te_names:
                trainer._move_component_to_gpu(name)
        elif te_names:
            # No quantization — move bf16 TEs straight to GPU
            for name in te_names:
                trainer._move_component_to_gpu(name)

        if trainer.get_te_cache():
            print("[STATUS:TE Cache Restored]", flush=True)
        trainer._pre_cache_text_embeddings()

        # TEs no longer needed — move to CPU or unload
        trainer._offload_text_encoders()

        # ── Latent (VAE) caching phase ───────────────────────────────
        # Move VAE to GPU for latent encoding, then offload.
        trainer._move_component_to_gpu("vae")

        trainer._validate_latent_cache()
        await trainer._pre_cache_latents()

        # VAE no longer needed after latents are cached
        trainer._offload_vae()

        # Notify parent process that cache directories exist,
        # so it can flag datasets as cache-bearing in the UI.
        try:
            ds_configs = trainer.config.get("datasets", [])
            ds_names = list({
                (d if isinstance(d, str) else d.get("dataset_name", ""))
                for d in ds_configs
            } - {""})
            if ds_names:
                import json as _json
                print(f"[CACHE_READY:{_json.dumps(ds_names)}]", flush=True)
        except Exception:
            pass  # Non-critical; don't abort training for this

        # ── Phase B: Prepare UNet for training ───────────────────────
        # Model quantization happens inside prepare_for_training()
        # via _quantize_components() → _quantize_primary_model()
        # (after model is moved to GPU and frozen).
        print("[STATUS:Preparing Training]", flush=True)
        await trainer.prepare_for_training()

        # ── Train ────────────────────────────────────────────────────
        print("[STATUS:Training]", flush=True)
        await trainer.train()
    except Exception as e:
        print(f"CRITICAL: Async Trainer Exception: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        raise e

def main():
    logger.info("trainer_entry_point_reached")
    
    parser = argparse.ArgumentParser(description="MRLN Arcane Tuner Training Entry Point")
    parser.add_argument("--definition_id", type=str, required=True, help="Model Definition ID")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration string")
    
    args = parser.parse_args()
    
    try:
        config = json.loads(args.config)
        logger.info(f"loaded_config: definition_id={args.definition_id} config_keys={list(config.keys())}")
        
        # Initialize Registry
        registry.initialize()
        
        definition = registry.get_definition(args.definition_id)
        
        if not definition:
             logger.error(f"definition_not_found: {args.definition_id}")
             print(f"CRITICAL: Definition ID '{args.definition_id}' not found in registry.", file=sys.stderr)
             sys.exit(1)
             
        logger.info(f"resolved_family: {definition.family}")
        
        if definition.family == "sdxl":
            from app.engine.models.families.sdxl.trainer import SDXLTrainer as TrainerClass
            
            # Instantiate V2 Trainer
            logger.info("instantiating_sdxl_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            # Execute Async Wrapper
            logger.info("starting_async_v2_training")
            asyncio.run(run_async_trainer(trainer))
            
        elif definition.family == "flux2":
            from app.engine.models.families.flux2.trainer import Flux2Trainer as TrainerClass
            
            # Instantiate Flux2 Trainer
            logger.info("instantiating_flux2_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            # Execute Async Wrapper
            logger.info("starting_async_flux2_training")
            asyncio.run(run_async_trainer(trainer))
            
        elif definition.family == "flux1":
            from app.engine.models.families.flux1.trainer import Flux1Trainer as TrainerClass
            
            # Instantiate Flux1 Trainer
            logger.info("instantiating_flux1_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            
            # Execute Async Wrapper
            logger.info("starting_async_flux1_training")
            asyncio.run(run_async_trainer(trainer))
            
        elif definition.family == "zimage":
            from app.engine.models.families.zimage.trainer import ZImageTrainer as TrainerClass
            
            logger.info("instantiating_zimage_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            logger.info("starting_async_zimage_training")
            asyncio.run(run_async_trainer(trainer))
            
        elif definition.family == "qwen_image":
            from app.engine.models.families.qwen_image.trainer import QwenImageTrainer as TrainerClass
            
            logger.info("instantiating_qwen_image_trainer")
            trainer = TrainerClass(definition=definition, run_config=config)
            logger.info("starting_async_qwen_image_training")
            asyncio.run(run_async_trainer(trainer))
            
        else:
            # Fallback to Legacy for everything else for now
            logger.info("using_legacy_trainer")
            logger.warning("legacy_trainer_path_incomplete_aborting", family=definition.family)
            print(f"CRITICAL: No trainer registered for family '{definition.family}'. Aborting.", file=sys.stderr)
            sys.exit(1)

        logger.info("training_completed_successfully")
        sys.exit(0)

    except Exception:
        print("CRITICAL: Unhandled exception in main execution block.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
