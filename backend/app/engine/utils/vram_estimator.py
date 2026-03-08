"""Pre-training VRAM estimation utility.

Calculates expected GPU memory consumption for a training run *before*
any weights are loaded.  The estimate is intentionally conservative
so users see a worst-case budget.

Usage::

    from app.engine.utils.vram_estimator import VRAMEstimator
    report = VRAMEstimator.estimate(definition, config)
    # report.fits  → True / False
    # report.peak_mb  → worst-case peak VRAM in MB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Well-known model sizes (B params) — used when introspection is unavailable
# ---------------------------------------------------------------------------
_FAMILY_PARAMS: dict[str, dict[str, float]] = {
    "sdxl": {
        "unet": 2.6,
        "text_encoder_1": 0.12,
        "text_encoder_2": 0.35,
        "vae": 0.08,
    },
    "flux1": {
        "transformer": 12.0,
        "text_encoder_clip": 0.12,
        "text_encoder_t5": 4.8,
        "vae": 0.08,
    },
    "flux2": {
        "transformer": 32.0,       # FLUX.2-dev default
        "text_encoder": 24.0,      # Mistral3 (dev)
        "vae": 0.17,
    },
    "zimage": {
        "transformer": 6.2,
        "text_encoder": 4.0,        # Qwen3
        "vae": 0.08,
    },
    "qwen_image": {
        "transformer": 20.4,
        "text_encoder": 8.3,        # Qwen2.5-VL
        "vae": 0.17,
    },
}

# Bytes-per-param for common dtypes
_DTYPE_BYTES: dict[str, int] = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.int8": 1,
}

# Bits-per-param after quantization (mirrors QuantizationFactory.bits_map)
_QUANT_BITS: dict[str, float] = {
    "none": 0,  # sentinel — uses native dtype
    "bf16": 16,
    "nvfp4": 4,
    "fp8": 8,
    "nf4": 4,
    "int4": 4,
    "int5": 5,
    "int6": 6,
    "int7": 7,
    "int8": 8,
}


# ---------------------------------------------------------------------------
# Result data-class
# ---------------------------------------------------------------------------

@dataclass
class VRAMReport:
    """Structured VRAM estimation result."""

    # --- Per-category breakdown (MB) ---
    model_weights_mb: float = 0.0
    lora_adapters_mb: float = 0.0
    optimizer_states_mb: float = 0.0
    gradients_mb: float = 0.0
    activations_mb: float = 0.0
    overhead_mb: float = 1024.0      # CUDA context + kernels (~1 GB)

    # --- Phase peaks ---
    caching_peak_mb: float = 0.0     # TE caching phase
    training_peak_mb: float = 0.0    # training phase (model + adapters + optim + grads + acts)

    # --- Summary ---
    peak_mb: float = 0.0             # max(caching, training)
    available_mb: float = 0.0        # from GPU query
    fits: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_weights_mb": round(self.model_weights_mb),
            "lora_adapters_mb": round(self.lora_adapters_mb),
            "optimizer_states_mb": round(self.optimizer_states_mb),
            "gradients_mb": round(self.gradients_mb),
            "activations_mb": round(self.activations_mb),
            "overhead_mb": round(self.overhead_mb),
            "caching_peak_mb": round(self.caching_peak_mb),
            "training_peak_mb": round(self.training_peak_mb),
            "peak_mb": round(self.peak_mb),
            "available_mb": round(self.available_mb),
            "fits": self.fits,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class VRAMEstimator:
    """Estimate peak VRAM for a model + config before loading weights.

    The estimator works from:
    - ``ModelDefinition`` metadata (family, detected_precision, architecture_params)
    - Training config dict (quantization, lora_rank, train_text_encoder, resolution, batch_size, etc.)
    - Live GPU info (optional)
    """

    @staticmethod
    def estimate(
        definition: Any,
        config: dict[str, Any],
    ) -> VRAMReport:
        """Build a VRAM report for the given model + training config.

        Args:
            definition: ``ModelDefinition`` instance (or dict-like with
                        ``family``, ``detected_precision``, ``architecture_params``).
            config:     Training config dict.

        Returns:
            ``VRAMReport`` with per-category breakdown and fit assessment.
        """
        report = VRAMReport()

        family = getattr(definition, "family", None) or "unknown"
        precision = getattr(definition, "detected_precision", {}) or {}
        arch = getattr(definition, "architecture_params", {}) or {}
        size_mb = getattr(definition, "model_size_mb", {}) or {}

        # ── 1. Model weight size (primary trainable component) ───────────
        primary_key = "unet"
        native_bpp = _bytes_per_param(precision.get(primary_key, "torch.bfloat16"))

        quant_scheme = config.get("quantization", "none")
        if quant_scheme != "none" and quant_scheme in _QUANT_BITS:
            effective_bpp = _QUANT_BITS[quant_scheme] / 8
        else:
            effective_bpp = native_bpp

        # Use model_size_mb if available (most accurate), else fall back to param estimate
        primary_disk_mb = _get_component_disk_mb(size_mb, primary_key)
        if primary_disk_mb > 0:
            # Scale on-disk size by quantization ratio
            report.model_weights_mb = primary_disk_mb * (effective_bpp / native_bpp)
            primary_params_b = (primary_disk_mb * 1024 * 1024) / (native_bpp * 1e9)
        else:
            primary_params_b = _get_primary_params(family, arch, primary_key)
            model_bytes = primary_params_b * 1e9 * effective_bpp
            report.model_weights_mb = model_bytes / (1024 * 1024)

        # ── 2. LoRA adapters ─────────────────────────────────────────────
        lora_rank = config.get("lora_rank", config.get("rank", 16))
        # Rough estimate: each target module gets rank×in + rank×out params
        # Typical ratio: ~1-3% of model params at rank 16
        lora_ratio = min(lora_rank / 16 * 0.015, 0.10)  # cap at 10%
        lora_params = primary_params_b * 1e9 * lora_ratio
        lora_bpp = 2  # adapters always bf16/fp16
        report.lora_adapters_mb = (lora_params * lora_bpp) / (1024 * 1024)

        # ── 3. Optimizer states (AdamW: 2 fp32 moments per trainable param) ──
        trainable_params = lora_params
        train_te = config.get("train_text_encoder", False)
        if train_te:
            te_params_b = _get_te_params(family)
            trainable_params += te_params_b * 1e9

        optimizer = config.get("optimizer", "adamw")
        if optimizer in ("adamw", "adam", "adam8bit", "adamw8bit"):
            # 2 moments × fp32 (4 bytes) = 8 bytes per trainable param
            # 8-bit optimizers halve this
            moment_bytes = 4 if "8bit" not in optimizer else 2
            report.optimizer_states_mb = (trainable_params * 2 * moment_bytes) / (1024 * 1024)
        elif optimizer in ("prodigy", "prodigyopt"):
            # Prodigy stores ~3× fp32 states per param
            report.optimizer_states_mb = (trainable_params * 12) / (1024 * 1024)
        else:
            # SGD / other — 1 momentum buffer
            report.optimizer_states_mb = (trainable_params * 4) / (1024 * 1024)

        # ── 4. Gradients ─────────────────────────────────────────────────
        report.gradients_mb = (trainable_params * lora_bpp) / (1024 * 1024)

        # ── 5. Activations (highly dependent on resolution + batch) ──────
        resolution = config.get("resolution", config.get("width", 1024))
        batch_size = config.get("batch_size", 1)
        grad_checkpointing = config.get("gradient_checkpointing", True)
        # gradient_accumulation_steps doesn't affect peak VRAM (same batch in memory)

        # Rough activation estimate:
        # Without grad checkpointing: ~resolution² × depth × hidden × batch × 2 bytes
        # With grad checkpointing: ~1/3 of above
        hidden_size = arch.get("hidden_size", 3072)
        depth = arch.get("depth", 19) + arch.get("depth_single_blocks", 38)
        pixels = (resolution // 8) ** 2  # latent space
        act_bytes = pixels * depth * hidden_size * batch_size * 2  # bf16
        act_factor = 0.33 if grad_checkpointing else 1.0
        report.activations_mb = (act_bytes * act_factor) / (1024 * 1024)

        # Cap activations at a reasonable max (empirical)
        max_act_mb = 8192 if not grad_checkpointing else 4096
        report.activations_mb = min(report.activations_mb, max_act_mb)

        # ── 6. Training peak ─────────────────────────────────────────────
        report.training_peak_mb = (
            report.model_weights_mb
            + report.lora_adapters_mb
            + report.optimizer_states_mb
            + report.gradients_mb
            + report.activations_mb
            + report.overhead_mb
        )

        # ── 7. Caching peak (TE on GPU during embedding generation) ─────
        te_bpp = _bytes_per_param(precision.get("text_encoder", "torch.bfloat16"))
        te_quant = config.get("te_quantization", "none")
        if te_quant != "none" and te_quant in _QUANT_BITS:
            te_effective_bpp = _QUANT_BITS[te_quant] / 8
        else:
            te_effective_bpp = te_bpp

        te_disk_mb = _get_component_disk_mb(size_mb, "text_encoder")
        if te_disk_mb > 0:
            te_mb = te_disk_mb * (te_effective_bpp / te_bpp)
        else:
            te_total_params_b = _get_te_params(family)
            te_mb = (te_total_params_b * 1e9 * te_effective_bpp) / (1024 * 1024)

        # During caching: TE on GPU + VAE might be loaded too
        vae_disk_mb = _get_component_disk_mb(size_mb, "vae")
        if vae_disk_mb > 0:
            vae_mb = vae_disk_mb
        else:
            vae_mb = (_get_vae_params(family) * 1e9 * 2) / (1024 * 1024)
        report.caching_peak_mb = te_mb + vae_mb + report.overhead_mb

        # ── 8. Overall peak ──────────────────────────────────────────────
        report.peak_mb = max(report.training_peak_mb, report.caching_peak_mb)

        # ── 9. GPU availability ──────────────────────────────────────────
        try:
            from app.core.system_monitor import system_monitor
            snap = system_monitor.snapshot()
            if snap.gpus:
                report.available_mb = snap.gpus[0].vram_total_mb
                report.fits = report.peak_mb < report.available_mb * 0.95  # 5% headroom
        except Exception:
            report.warnings.append("Could not query GPU — fit check skipped")

        # ── 10. Warnings ─────────────────────────────────────────────────
        if report.peak_mb > 0 and report.available_mb > 0:
            ratio = report.peak_mb / report.available_mb
            if ratio > 1.0:
                overshoot_mb = round(report.peak_mb - report.available_mb)
                report.warnings.append(
                    f"Estimated peak VRAM ({round(report.peak_mb)} MB) exceeds "
                    f"available ({round(report.available_mb)} MB) by {overshoot_mb} MB. "
                    f"Consider quantization, lower resolution, or smaller batch size."
                )
            elif ratio > 0.85:
                report.warnings.append(
                    f"Estimated VRAM usage is {round(ratio * 100)}% of available — "
                    f"tight fit. May OOM under activation spikes."
                )

        if not config.get("gradient_checkpointing", True):
            report.warnings.append(
                "Gradient checkpointing is OFF — activation memory will be significantly higher."
            )

        # ── Quantization arch compatibility ──────────────────────────
        _check_quant_compat(quant_scheme, "Model quantization", report, config)
        _check_quant_compat(te_quant, "TE quantization", report, config)

        logger.info(
            "vram_estimate",
            family=family,
            peak_mb=round(report.peak_mb),
            available_mb=round(report.available_mb),
            fits=report.fits,
        )

        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_component_disk_mb(size_mb: dict, key: str) -> float:
    """Look up on-disk component size in MB from model_size_mb dict.

    Tries *key* first, then common aliases (transformer/unet).
    Returns 0 when not found.
    """
    for k in (key, "transformer", "unet"):
        val = size_mb.get(k, 0)
        if val > 0:
            return float(val)
    return 0.0


def _get_primary_params(family: str, arch: dict, key: str = "unet") -> float:
    """Get primary model param count in billions (fallback only)."""
    # Try architecture params first (from introspection)
    total_params = arch.get("total_params", 0)
    if total_params > 0:
        return total_params / 1e9

    # Fall back to well-known sizes
    family_data = _FAMILY_PARAMS.get(family, {})
    for k in (key, "transformer", "unet", "model"):
        if k in family_data:
            return family_data[k]
    return 2.0  # conservative default


def _get_te_params(family: str) -> float:
    """Get total text encoder param count in billions (fallback only)."""
    family_data = _FAMILY_PARAMS.get(family, {})
    total = 0.0
    for k, v in family_data.items():
        if "text_encoder" in k:
            total += v
    return total or 0.35  # default ~350M


def _get_vae_params(family: str) -> float:
    """Get VAE param count in billions (fallback only)."""
    return _FAMILY_PARAMS.get(family, {}).get("vae", 0.08)


def _bytes_per_param(dtype_str: str) -> int:
    """Get bytes per parameter for a dtype string."""
    return _DTYPE_BYTES.get(dtype_str, 2)  # default bf16


def _check_quant_compat(scheme: str, label: str, report: VRAMReport, config: dict[str, Any]) -> None:
    """Add a warning to *report* if *scheme* isn't supported on this GPU."""
    if scheme in ("none", "bf16"):
        return

    try:
        import torch
        if not torch.cuda.is_available():
            return
        cap = torch.cuda.get_device_capability()
        sm = cap[0] * 10 + cap[1]
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        return

    # NVFP4 requires SM >= 100
    if scheme == "nvfp4" and sm < 100:
        from app.engine.factories.quantization import QuantizationFactory
        backend_name = config.get("te_quantization_backend", "auto")
        fallback, scheme = QuantizationFactory.validate_and_fallback(scheme, backend_name)
        report.warnings.append(
            f"{label} '{scheme}' requires Blackwell (SM ≥ 100) but {gpu_name} has SM {sm}. "
            f"Will fall back to '{fallback}' at runtime."
        )

    # FP8 requires SM >= 89
    elif scheme == "fp8" and sm < 89:
        from app.engine.factories.quantization import QuantizationFactory
        backend_name = config.get("quantization_backend", "auto")
        fallback, scheme = QuantizationFactory.validate_and_fallback(scheme, backend_name)
        report.warnings.append(
            f"{label} '{scheme}' requires Ada/Hopper/Blackwell (SM ≥ 89) but {gpu_name} has SM {sm}. "
            f"Will fall back to '{fallback}' at runtime."
        )
