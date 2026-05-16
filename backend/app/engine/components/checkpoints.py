"""
Checkpoint Manager — saves and loads full training state for resume.

Handles:
- Model weights via ModelSaver interface (distribution LoRA for inference)
- PEFT adapter saving/loading (save_pretrained / load_adapter)
- Optimizer, scheduler, scaler, EMA state dicts
- Training metadata (step, config, timestamp)
- Checkpoint manifest (component inventory + file sizes)
- Config override validation on resume
"""

import gc
import json
import os
import shutil
import time

import structlog
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from safetensors.torch import load_file as st_load
from typing import Any

from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)


# ── Pydantic Models ──────────────────────────────────────────────────────


class CheckpointState(BaseModel):
    """Result of loading a checkpoint — everything the trainer needs to resume."""

    model_config = {"arbitrary_types_allowed": True}

    global_step: int = 0
    elapsed_time: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0
    components_loaded: list[str] = Field(default_factory=list)
    adapters_loaded: list[str] = Field(default_factory=list)
    te_cache: dict[str, dict[str, torch.Tensor]] | None = None
    cache_manifest: dict[str, list[str]] | None = None


# ── Config Override Categories ───────────────────────────────────────────

# Safe: always allowed, no side-effects on compatibility
SAFE_OVERRIDES = {
    "learning_rate", "lr_scheduler", "lr_warmup_steps",
    "max_train_steps", "save_every_n_steps", "keep_last_checkpoints",
    "train_batch_size", "output_dir", "lora_name",
}

# Warning: allowed but the user should know
WARNING_OVERRIDES = {
    "gradient_accumulation_steps", "noise_offset", "min_snr_gamma",
    "ema", "ema_decay", "gradient_checkpointing", "offload_to_cpu",
    "mixed_precision", "save_precision",
}

# Blocked: would break checkpoint compatibility
BLOCKED_OVERRIDES = {
    "network_rank", "network_alpha",
}


# ── Config Override Logic ────────────────────────────────────────────────


def apply_overrides(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge current config overrides into checkpoint config for resume.

    Safe overrides are applied silently. Warning overrides log a warning.
    Blocked overrides raise ``ValueError``.

    Args:
        checkpoint_config: The config saved in the checkpoint.
        current_config: The config the user is resuming with.

    Returns:
        Merged config (checkpoint base + allowed overrides).

    Raises:
        ValueError: If a blocked override is attempted.
    """
    merged = dict(checkpoint_config)

    for key, new_val in current_config.items():
        old_val = checkpoint_config.get(key)
        if old_val == new_val:
            continue  # no change

        if key in BLOCKED_OVERRIDES:
            raise ValueError(
                f"Cannot override '{key}' on resume: "
                f"checkpoint={old_val}, requested={new_val}. "
                f"This would break checkpoint compatibility."
            )

        if key in WARNING_OVERRIDES:
            logger.warning(
                "config_override_warning",
                key=key,
                old_value=old_val,
                new_value=new_val,
            )
            merged[key] = new_val

        elif key in SAFE_OVERRIDES:
            logger.info(
                "config_override_applied",
                key=key,
                old_value=old_val,
                new_value=new_val,
            )
            merged[key] = new_val

        else:
            # Unknown keys — apply silently (custom family-specific keys)
            merged[key] = new_val

    return merged


# ── Compatibility Validation ─────────────────────────────────────────────


def validate_compatibility(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
) -> list[str]:
    """
    Check that a checkpoint is compatible with the current training config.

    Returns a list of warning messages. Raises ``ValueError`` on fatal
    incompatibility.

    Args:
        checkpoint_config: Config from the saved checkpoint.
        current_config: Config for the new run.

    Returns:
        List of warning strings (empty if fully compatible).

    Raises:
        ValueError: If rank or alpha mismatch detected.
    """
    warnings: list[str] = []

    # Fatal: rank/alpha mismatch would corrupt weight loading
    for key in ("network_rank", "network_alpha"):
        ckpt_val = checkpoint_config.get(key)
        curr_val = current_config.get(key)
        if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
            raise ValueError(
                f"Checkpoint incompatible: {key} was {ckpt_val}, "
                f"current config has {curr_val}. Cannot resume."
            )

    # Warning: model definition changed
    ckpt_def = checkpoint_config.get("model_definition")
    curr_def = current_config.get("model_definition")
    if ckpt_def and curr_def and ckpt_def != curr_def:
        warnings.append(
            f"Model definition changed: checkpoint={ckpt_def}, current={curr_def}"
        )

    # Warning: TE training flag changed
    ckpt_te = checkpoint_config.get("train_text_encoder")
    curr_te = current_config.get("train_text_encoder")
    if ckpt_te is not None and curr_te is not None and ckpt_te != curr_te:
        warnings.append(
            f"train_text_encoder changed: checkpoint={ckpt_te}, current={curr_te}"
        )

    for w in warnings:
        logger.warning("checkpoint_compatibility_warning", message=w)

    return warnings


# ── LoRA Name Resolution ─────────────────────────────────────────────────


def resolve_lora_name(config: dict[str, Any]) -> str:
    """Resolve ``{placeholder}`` tokens in ``lora_name`` using config values.

    This is a backend safety-net: the frontend normally resolves
    placeholders before submission, but API consumers that bypass
    the UI may send raw template strings.

    Args:
        config: Full training config dict.

    Returns:
        Resolved LoRA name with all ``{key}`` tokens replaced.
    """
    import re

    raw = config.get("lora_name", "lora")

    def _replacer(match: re.Match) -> str:  # type: ignore[type-arg]
        key = match.group(1)
        val = config.get(key, "")
        return str(val) if val else ""

    return re.sub(r"\{(\w+)\}", _replacer, raw)


# ── CheckpointManager ────────────────────────────────────────────────────


class CheckpointManager:
    """
    Manages saving and loading of training state for resume.

    Handles model weights (via saver), PEFT adapters, optimizer,
    scheduler, scaler, EMA, and training metadata.
    """

    def __init__(self, output_dir: str, saver_impl: Any = None):
        """
        Args:
            output_dir: Root directory for checkpoint output.
            saver_impl: Object with ``save(components, path, metadata)`` method
                        for producing distribution-format LoRA files.
        """
        self.output_dir = output_dir
        self.saver = saver_impl
        self.last_save_time = 0.0

    # ── Save ─────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        step: int,
        components: dict[str, Any],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        config: dict[str, Any] | None = None,
        ema_handler: Any | None = None,
        is_final: bool = False,
        elapsed_time: float = 0.0,
        te_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        cache_manifest: dict[str, list[str]] | None = None,
    ) -> str:
        """
        Save a full checkpoint (distribution LoRA + resume state).

        Args:
            step: Current global training step.
            components: Dict of model components (UNet, TEs, etc.).
            optimizer: Optimizer to save state_dict from.
            scheduler: LR scheduler to save state_dict from.
            scaler: GradScaler to save state_dict from.
            config: Training configuration dict.
            ema_handler: EMA handler to save shadow weights from.
            is_final: True for the final save at training end.

        Returns:
            Path to the checkpoint directory.
        """
        config = config or {}

        # Determine paths
        resolved_name = resolve_lora_name(config)
        if is_final:
            folder_name = "final"
            lora_filename = f"{resolved_name}_final.safetensors"
        else:
            folder_name = f"checkpoint-{step:06d}"
            lora_filename = f"{resolved_name}_{step:06d}.safetensors"

        save_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        logger.info("saving_checkpoint", step=step, path=save_path, is_final=is_final)

        # 1. Save distribution LoRA (for inference)
        if self.saver:
            if ema_handler:
                logger.debug("swapping_ema_weights_for_save")
                ema_handler.store_and_swap()

            metadata = {
                "ss_session_id": config.get("job_id", "unknown"),
                "step": step,
            }

            # Inject config into components so savers can access training params
            save_components = dict(components)
            save_components["config"] = config

            # To root output dir (filenames are step-unique, e.g. lora_001050.safetensors)
            dist_path = Path(self.output_dir) / lora_filename
            try:
                self.saver.save(save_components, dist_path, metadata=metadata)
                logger.info("saved_distribution_lora", path=str(dist_path))
            except (OSError, RuntimeError) as e:
                logger.error("failed_to_save_distribution_lora", error=str(e))

            if ema_handler:
                ema_handler.restore()

        # 2. Save resume state
        self._save_train_state(save_path, components, optimizer, scheduler, scaler, config, step, ema_handler, elapsed_time, te_cache=te_cache, cache_manifest=cache_manifest)

        # 2b. Also save step-numbered checkpoint when final (rollback safety)
        if is_final:
            step_folder = f"checkpoint-{step:06d}"
            step_save_path = os.path.join(self.output_dir, step_folder)
            os.makedirs(step_save_path, exist_ok=True)
            logger.info("saving_step_numbered_final", step=step, path=step_save_path)
            self._save_train_state(step_save_path, components, optimizer, scheduler, scaler, config, step, ema_handler, elapsed_time, te_cache=te_cache, cache_manifest=cache_manifest)

        # 3. Write verbose training log (to root output dir for easy access)
        self._write_training_log(
            step=step,
            components=components,
            optimizer=optimizer,
            config=config,
            is_final=is_final,
            elapsed_time=elapsed_time,
            lora_filename=lora_filename,
        )

        # 4. Cleanup old checkpoint folders (never LoRA files in root)
        keep_last = int(config.get("keep_last_checkpoints", 0))
        if keep_last > 0:
            self._cleanup_old_checkpoints(keep_last)

        self.last_save_time = time.time()
        return save_path

    def _cleanup_old_checkpoints(self, keep_last: int) -> None:
        """Delete checkpoint folders older than the retention window.

        Only removes ``checkpoint-NNNNNN/`` directories.  ``final/`` and
        LoRA ``.safetensors`` files in the root output dir are never touched.
        """
        checkpoint_dirs = sorted([
            d for d in os.listdir(self.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(self.output_dir, d))
        ])

        if len(checkpoint_dirs) <= keep_last:
            return

        to_delete = checkpoint_dirs[:-keep_last]
        for d in to_delete:
            dir_path = os.path.join(self.output_dir, d)
            try:
                shutil.rmtree(dir_path)
                logger.info("deleted_old_checkpoint", path=dir_path)
            except OSError as e:
                logger.warning("failed_to_delete_checkpoint", path=dir_path, error=str(e))

    def _write_training_log(
        self,
        step: int,
        components: dict[str, Any],
        optimizer: Any | None,
        config: dict[str, Any],
        is_final: bool,
        elapsed_time: float,
        lora_filename: str,
    ) -> None:
        """Write comprehensive training log to the output directory.

        Captures everything needed to debug issues like rank mismatches,
        wrong alpha values, or weight shape problems. Written on every
        checkpoint so even intermediate saves are fully debuggable.

        Args:
            step: Current global training step.
            components: Dict of model components.
            optimizer: Optimizer instance (for type/LR info).
            config: Full training configuration.
            is_final: Whether this is the final save.
            elapsed_time: Total elapsed training time in seconds.
            lora_filename: Name of the saved LoRA file.
        """
        try:
            log: dict[str, Any] = {
                "software": "Arcane Tuner",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "step": step,
                "is_final": is_final,
                "elapsed_seconds": round(elapsed_time, 2),
                "elapsed_human": self._format_elapsed(elapsed_time),
                "output_dir": self.output_dir,
                "lora_filename": lora_filename,
            }

            # ── Full training config ──
            log["config"] = config

            # ── PEFT / LoRA details (from actual model, not just config) ──
            peft_info = {}
            for name, comp in components.items():
                if name == "config":
                    continue
                if hasattr(comp, "peft_config"):
                    comp_peft = {}
                    for adapter_name, peft_cfg in comp.peft_config.items():
                        adapter_info = {
                            "adapter_name": adapter_name,
                            "peft_type": str(getattr(peft_cfg, "peft_type", "unknown")),
                            "r": getattr(peft_cfg, "r", None),
                            "lora_alpha": getattr(peft_cfg, "lora_alpha", None),
                            "lora_dropout": getattr(peft_cfg, "lora_dropout", None),
                            "bias": getattr(peft_cfg, "bias", None),
                            "target_modules": sorted(peft_cfg.target_modules)
                                if hasattr(peft_cfg, "target_modules") and peft_cfg.target_modules
                                else None,
                            "modules_to_save": getattr(peft_cfg, "modules_to_save", None),
                        }
                        comp_peft[adapter_name] = adapter_info
                    peft_info[name] = comp_peft
            if peft_info:
                log["peft_config"] = peft_info

            # ── Trainable parameter counts ──
            param_counts = {}
            for name, comp in components.items():
                if name == "config":
                    continue
                if isinstance(comp, torch.nn.Module):
                    trainable = sum(p.numel() for p in comp.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in comp.parameters())
                    param_counts[name] = {
                        "trainable": trainable,
                        "total": total,
                        "trainable_pct": round(trainable / total * 100, 4) if total > 0 else 0,
                    }
            if param_counts:
                log["parameter_counts"] = param_counts

            # ── Weight shape verification (per-tensor snapshot) ──
            weight_shapes = {}
            for name, comp in components.items():
                if name == "config":
                    continue
                if hasattr(comp, "peft_config"):
                    try:
                        from peft import get_peft_model_state_dict
                        sd = get_peft_model_state_dict(comp)
                        shapes = {}
                        for key, tensor in sd.items():
                            if isinstance(tensor, torch.Tensor):
                                shapes[key] = {
                                    "shape": list(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                    "numel": tensor.numel(),
                                }
                        weight_shapes[name] = shapes
                    except (ImportError, RuntimeError) as e:
                        weight_shapes[name] = {"error": str(e)}
            if weight_shapes:
                log["weight_shapes"] = weight_shapes

            # ── Optimizer info ──
            if optimizer:
                opt_info = {"type": type(optimizer).__name__}
                if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                    pg = optimizer.param_groups[0]
                    opt_info["lr"] = pg.get("lr")
                    opt_info["weight_decay"] = pg.get("weight_decay")
                    opt_info["betas"] = pg.get("betas")
                log["optimizer"] = opt_info

            # ── LoRA file size ──
            dist_path = os.path.join(self.output_dir, lora_filename)
            if os.path.exists(dist_path):
                size_bytes = os.path.getsize(dist_path)
                log["lora_file_size_mb"] = round(size_bytes / (1024 * 1024), 2)

            # ── Write ──
            log_path = os.path.join(self.output_dir, "training_log.json")
            os.makedirs(self.output_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, default=str)

            logger.info("training_log_written", path=log_path)

        except (OSError, ValueError, TypeError) as e:
            logger.error("failed_to_write_training_log", error=str(e))

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed seconds as human-readable string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    def _save_train_state(
        self,
        path: str,
        components: dict[str, Any],
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        config: dict[str, Any],
        step: int,
        ema_handler: Any,
        elapsed_time: float = 0.0,
        te_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        cache_manifest: dict[str, list[str]] | None = None,
    ) -> None:
        """Save all training state files for resume."""
        logger.info("saving_train_state", path=path)
        manifest: dict[str, int] = {}

        # 1. Components (PEFT adapters + standard modules)
        for name, comp in components.items():
            try:
                if hasattr(comp, "save_pretrained"):
                    # PEFT model → save adapter via save_pretrained
                    comp_subfolder = os.path.join(path, name)
                    os.makedirs(comp_subfolder, exist_ok=True)
                    # Pre-clear existing safetensors to avoid mmap locks on resume
                    for existing in os.listdir(comp_subfolder):
                        if existing.endswith(".safetensors"):
                            try:
                                os.remove(os.path.join(comp_subfolder, existing))
                            except OSError:
                                pass  # locked — save_pretrained will overwrite
                    comp.save_pretrained(comp_subfolder)
                    # Record adapter files in manifest
                    for f in os.listdir(comp_subfolder):
                        fpath = os.path.join(comp_subfolder, f)
                        if os.path.isfile(fpath):
                            manifest[f"{name}/{f}"] = os.path.getsize(fpath)
                    logger.info("saved_peft_component", component=name, path=comp_subfolder)

                elif isinstance(comp, torch.nn.Module) or hasattr(comp, "state_dict"):
                    comp_path = os.path.join(path, f"{name}.pt")
                    torch.save(comp.state_dict(), comp_path)
                    manifest[f"{name}.pt"] = os.path.getsize(comp_path)

                else:
                    logger.debug("skipping_component_no_state_dict", component=name)

            except Exception as e:
                logger.error("failed_to_save_component", component=name, error=str(e))

        # 2. Optimizer
        if optimizer:
            try:
                opt_path = os.path.join(path, "optimizer.pt")
                torch.save(optimizer.state_dict(), opt_path)
                manifest["optimizer.pt"] = os.path.getsize(opt_path)
            except (OSError, RuntimeError) as e:
                logger.error("failed_to_save_optimizer", error=str(e))
        else:
            logger.warning("optimizer_is_none_skipping_save")

        # 3. Scheduler
        if scheduler:
            try:
                sch_path = os.path.join(path, "scheduler.pt")
                torch.save(scheduler.state_dict(), sch_path)
                manifest["scheduler.pt"] = os.path.getsize(sch_path)
            except (OSError, RuntimeError) as e:
                logger.error("failed_to_save_scheduler", error=str(e))

        # 4. Scaler (AMP)
        if scaler:
            try:
                sc_path = os.path.join(path, "scaler.pt")
                torch.save(scaler.state_dict(), sc_path)
                manifest["scaler.pt"] = os.path.getsize(sc_path)
            except (OSError, RuntimeError) as e:
                logger.error("failed_to_save_scaler", error=str(e))

        # 5. EMA Shadow
        if ema_handler:
            try:
                ema_path = os.path.join(path, "ema_shadow.pt")
                torch.save(ema_handler.state_dict(), ema_path)
                manifest["ema_shadow.pt"] = os.path.getsize(ema_path)
            except (OSError, RuntimeError) as e:
                logger.error("failed_to_save_ema", error=str(e))

        # 6. Text Embedding Cache
        if te_cache:
            try:
                flat_tensors: dict[str, torch.Tensor] = {}
                index: dict[str, list[str]] = {}  # cache_name → [caption, ...]
                for cache_name, cap_dict in te_cache.items():
                    captions_list = list(cap_dict.keys())
                    index[cache_name] = captions_list
                    for i, cap in enumerate(captions_list):
                        value = cap_dict[cap]
                        if isinstance(value, torch.Tensor):
                            flat_tensors[f"{cache_name}::{i}"] = value
                        elif isinstance(value, (tuple, list)):
                            # Multi-tensor entry (e.g. qwen_image's
                            # (embedding, attention_mask) pairs).
                            for j, t in enumerate(value):
                                if isinstance(t, torch.Tensor):
                                    flat_tensors[f"{cache_name}::{i}::{j}"] = t
                        else:
                            logger.warning(
                                "skipping_non_tensor_cache_entry",
                                cache=cache_name, index=i,
                                type=type(value).__name__,
                            )
                if flat_tensors:
                    te_path = os.path.join(path, "te_cache.safetensors")
                    safe_save_file(flat_tensors, te_path)
                    manifest["te_cache.safetensors"] = os.path.getsize(te_path)
                    idx_path = os.path.join(path, "te_cache_index.json")
                    with open(idx_path, "w", encoding="utf-8") as f:
                        json.dump(index, f, ensure_ascii=False)
                    manifest["te_cache_index.json"] = os.path.getsize(idx_path)
                    logger.info(
                        "saved_te_cache",
                        caches=list(index.keys()),
                        total_entries=sum(len(v) for v in index.values()),
                    )
            except Exception as e:
                logger.error("failed_to_save_te_cache", error=str(e))

        # 7. Metadata — use atomic write to handle locked files on resume
        state = {
            "global_step": step,
            "elapsed_time": elapsed_time,
            "config": config,
            "timestamp": time.time(),
        }
        if cache_manifest:
            state["cache_manifest"] = cache_manifest
        state_path = os.path.join(path, "training_state.json")
        tmp_state_path = state_path + ".tmp"
        try:
            with open(tmp_state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_state_path, state_path)
            manifest["training_state.json"] = os.path.getsize(state_path)
        except OSError as e:
            logger.error("failed_to_save_training_state", error=str(e))
            if os.path.exists(tmp_state_path):
                try:
                    os.remove(tmp_state_path)
                except OSError:
                    pass

        # 7. Manifest
        manifest_path = os.path.join(path, "checkpoint_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "train_state_saved",
            path=path,
            files_saved=len(manifest),
            total_bytes=sum(manifest.values()),
        )

    # ── Load ─────────────────────────────────────────────────────────

    def load_checkpoint(
        self,
        path: str,
        components: dict[str, Any] | None = None,
        peft_components: dict[str, Any] | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        ema_handler: Any | None = None,
        current_config: dict[str, Any] | None = None,
        skip_scheduler: bool = False,
    ) -> CheckpointState:
        """
        Load training state from a checkpoint directory.

        Args:
            path: Path to the checkpoint directory.
            components: Standard PyTorch modules to load state_dicts into.
            peft_components: PEFT models — will auto-detect adapter subdirs.
            optimizer: Optimizer to restore state into.
            scheduler: LR scheduler to restore state into.
            scaler: GradScaler to restore state into.
            ema_handler: EMA handler to restore shadow weights into.
            current_config: If provided, validates compatibility and applies overrides.
            skip_scheduler: If True, skip loading scheduler state (e.g. LR override).

        Returns:
            CheckpointState with step, config, loaded components list.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            ValueError: If checkpoint is incompatible with current config.
        """
        logger.info("loading_checkpoint", path=path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = CheckpointState()

        # 1. Read metadata
        state_path = os.path.join(path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                meta = json.load(f)
                state.global_step = meta.get("global_step", 0)
                state.elapsed_time = meta.get("elapsed_time", 0.0)
                state.config = meta.get("config", {})
                state.timestamp = meta.get("timestamp", 0.0)
                state.cache_manifest = meta.get("cache_manifest")

        # 2. Validate compatibility & apply overrides
        if current_config and state.config:
            validate_compatibility(state.config, current_config)

            # Detect scheduler change BEFORE merging (merged config won't show the diff)
            ckpt_sched = state.config.get("lr_scheduler")
            curr_sched = current_config.get("lr_scheduler")
            if ckpt_sched and curr_sched and ckpt_sched != curr_sched:
                skip_scheduler = True
                logger.info("skipping_scheduler_load_due_to_override",
                            checkpoint_scheduler=ckpt_sched, new_scheduler=curr_sched)

            merged = apply_overrides(state.config, current_config)
            state.config = merged

        # 3. Standard components (.pt files)
        if components:
            for name, comp in components.items():
                pt_path = os.path.join(path, f"{name}.pt")
                if os.path.exists(pt_path) and hasattr(comp, "load_state_dict"):
                    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
                    comp.load_state_dict(sd)
                    state.components_loaded.append(name)
                    logger.debug("loaded_component", name=name)

        # 4. PEFT adapters (auto-detect subdirs with adapter_config.json)
        if peft_components:
            for name, model in peft_components.items():
                adapter_dir = os.path.join(path, name)
                adapter_config = os.path.join(adapter_dir, "adapter_config.json")
                if os.path.exists(adapter_config):
                    model.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
                    state.adapters_loaded.append(name)
                    logger.info("loaded_peft_adapter", component=name, path=adapter_dir)

            # Release mmap file locks from PEFT adapter safetensors
            # by cloning trainable parameters into regular memory
            for name, model in peft_components.items():
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = param.data.clone()
            gc.collect()

        # 5. Optimizer
        if optimizer:
            opt_path = os.path.join(path, "optimizer.pt")
            if os.path.exists(opt_path):
                sd = torch.load(opt_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(sd)
                state.components_loaded.append("optimizer")
                logger.debug("loaded_optimizer")

        # 6. Scheduler
        if scheduler and not skip_scheduler:
            sch_path = os.path.join(path, "scheduler.pt")
            if os.path.exists(sch_path):
                sd = torch.load(sch_path, map_location="cpu", weights_only=True)
                scheduler.load_state_dict(sd)
                state.components_loaded.append("scheduler")
                logger.debug("loaded_scheduler")

        # 7. Scaler
        if scaler:
            sc_path = os.path.join(path, "scaler.pt")
            if os.path.exists(sc_path):
                sd = torch.load(sc_path, map_location="cpu", weights_only=True)
                scaler.load_state_dict(sd)
                state.components_loaded.append("scaler")
                logger.debug("loaded_scaler")

        # 8. EMA
        if ema_handler:
            ema_path = os.path.join(path, "ema_shadow.pt")
            if os.path.exists(ema_path):
                sd = torch.load(ema_path, map_location="cpu", weights_only=True)
                ema_handler.load_state_dict(sd)
                state.components_loaded.append("ema")
                logger.debug("loaded_ema")

        # 9. Text Embedding Cache
        te_cache_path = os.path.join(path, "te_cache.safetensors")
        te_index_path = os.path.join(path, "te_cache_index.json")
        if os.path.exists(te_cache_path) and os.path.exists(te_index_path):
            try:
                flat = st_load(te_cache_path, device="cpu")
                # Clone tensors to release mmap file lock
                flat = {k: v.clone() for k, v in flat.items()}
                gc.collect()
                with open(te_index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                caches: dict[str, dict[str, torch.Tensor]] = {}
                for cache_name, captions_list in index.items():
                    cap_dict: dict[str, Any] = {}
                    for i, cap in enumerate(captions_list):
                        single_key = f"{cache_name}::{i}"
                        if single_key in flat:
                            # Single-tensor entry (original format)
                            cap_dict[cap] = flat[single_key]
                        else:
                            # Multi-tensor entry — collect sub-parts
                            parts: list[torch.Tensor] = []
                            j = 0
                            while True:
                                part_key = f"{cache_name}::{i}::{j}"
                                if part_key not in flat:
                                    break
                                parts.append(flat[part_key])
                                j += 1
                            if parts:
                                cap_dict[cap] = tuple(parts)
                    caches[cache_name] = cap_dict
                state.te_cache = caches
                state.components_loaded.append("te_cache")
                logger.info(
                    "loaded_te_cache",
                    caches=list(caches.keys()),
                    total_entries=sum(len(v) for v in caches.values()),
                )
            except (OSError, RuntimeError, json.JSONDecodeError) as e:
                logger.warning("failed_to_load_te_cache", error=str(e))

        logger.info(
            "checkpoint_loaded",
            step=state.global_step,
            components=state.components_loaded,
            adapters=state.adapters_loaded,
        )

        return state
