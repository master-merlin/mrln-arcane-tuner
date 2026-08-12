"""FLUX.2 LoRA saver — export PEFT weights to BFL-format for ComfyUI.

Converts diffusers-style PEFT keys (``transformer_blocks.N.attn.to_q``)
to BFL-format keys (``diffusion_model.double_blocks.N.img_attn.qkv``)
that ComfyUI expects.  Handles QKV fusion, module name remapping,
and dtype casting.

Key mapping derived from diffusers ``lora_conversion_utils.py``
``_convert_non_diffusers_flux2_lora_to_diffusers`` (reverse direction).
"""

import re
from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict
from app.engine.utils.lora_metadata import trigger_metadata
from app.engine.utils.safe_save import safe_save_file

from app.engine.core.interfaces import ModelSaver

logger = structlog.get_logger(__name__)


def _diffusers_to_bfl(diffusers_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a diffusers-format LoRA state dict to BFL-format.

    Handles:
        - Double blocks: fuse separate Q/K/V lora_B into single ``qkv``,
          rename proj and MLP keys.
        - Single blocks: rename ``to_qkv_mlp_proj`` → ``linear1``,
          ``to_out`` → ``linear2``.
    """
    bfl_sd: dict[str, torch.Tensor] = {}
    remaining = dict(diffusers_sd)

    # ── Double blocks ────────────────────────────────────────────────
    # Detect how many double blocks exist
    double_block_indices = set()
    for key in remaining:
        m = re.match(r"transformer_blocks\.(\d+)\.", key)
        if m:
            double_block_indices.add(int(m.group(1)))

    for dl in sorted(double_block_indices):
        tb = f"transformer_blocks.{dl}"
        db = f"double_blocks.{dl}"

        # img_attn QKV: prefer fused to_qkv (simple rename), fall back to
        # SVD fusion of separate to_q/k/v for backward compat
        if f"{tb}.attn.to_qkv.lora_A.weight" in remaining:
            _rename_pair(remaining, bfl_sd, f"{tb}.attn.to_qkv", f"{db}.img_attn.qkv")
        else:
            _fuse_qkv(
                remaining, bfl_sd,
                q_prefix=f"{tb}.attn.to_q",
                k_prefix=f"{tb}.attn.to_k",
                v_prefix=f"{tb}.attn.to_v",
                out_prefix=f"{db}.img_attn.qkv",
            )

        # txt_attn QKV: same pattern
        if f"{tb}.attn.to_added_qkv.lora_A.weight" in remaining:
            _rename_pair(remaining, bfl_sd, f"{tb}.attn.to_added_qkv", f"{db}.txt_attn.qkv")
        else:
            _fuse_qkv(
                remaining, bfl_sd,
                q_prefix=f"{tb}.attn.add_q_proj",
                k_prefix=f"{tb}.attn.add_k_proj",
                v_prefix=f"{tb}.attn.add_v_proj",
                out_prefix=f"{db}.txt_attn.qkv",
            )

        # Proj (1:1 rename)
        _rename_pair(remaining, bfl_sd, f"{tb}.attn.to_out.0", f"{db}.img_attn.proj")
        _rename_pair(remaining, bfl_sd, f"{tb}.attn.to_add_out", f"{db}.txt_attn.proj")

        # MLP (1:1 rename)
        _rename_pair(remaining, bfl_sd, f"{tb}.ff.linear_in", f"{db}.img_mlp.0")
        _rename_pair(remaining, bfl_sd, f"{tb}.ff.linear_out", f"{db}.img_mlp.2")
        _rename_pair(remaining, bfl_sd, f"{tb}.ff_context.linear_in", f"{db}.txt_mlp.0")
        _rename_pair(remaining, bfl_sd, f"{tb}.ff_context.linear_out", f"{db}.txt_mlp.2")

    # ── Single blocks ────────────────────────────────────────────────
    single_block_indices = set()
    for key in remaining:
        m = re.match(r"single_transformer_blocks\.(\d+)\.", key)
        if m:
            single_block_indices.add(int(m.group(1)))

    for sl in sorted(single_block_indices):
        stb = f"single_transformer_blocks.{sl}"
        sb = f"single_blocks.{sl}"

        # linear1 = to_qkv_mlp_proj (direct rename)
        _rename_pair(remaining, bfl_sd, f"{stb}.attn.to_qkv_mlp_proj", f"{sb}.linear1")

        # linear2 = to_out (direct rename)
        _rename_pair(remaining, bfl_sd, f"{stb}.attn.to_out", f"{sb}.linear2")

    # Anything left gets passed through with a warning
    if remaining:
        logger.warning(
            "flux2_saver_unconverted_keys",
            keys=list(remaining.keys()),
        )
        bfl_sd.update(remaining)

    return bfl_sd


def _fuse_qkv(
    src: dict[str, torch.Tensor],
    dst: dict[str, torch.Tensor],
    q_prefix: str,
    k_prefix: str,
    v_prefix: str,
    out_prefix: str,
) -> None:
    """Fuse separate Q/K/V LoRA weights into a single fused QKV weight.

    BFL format uses a **shared** ``lora_A`` for Q/K/V, but PEFT trains
    independent ``lora_A`` per projection.  A naïve fusion that discards
    K/V ``lora_A`` corrupts the LoRA at inference (wrong attention K/V).

    Fix: merge each pair into a full-rank delta (``lora_B @ lora_A``),
    concatenate ``[ΔQ; ΔK; ΔV]``, then re-decompose via truncated SVD
    to recover a shared ``lora_A`` and fused ``lora_B``.
    """
    a_suffix = "lora_A.weight"
    b_suffix = "lora_B.weight"

    q_a_key = f"{q_prefix}.{a_suffix}"
    if q_a_key not in src:
        return

    # Gather all six tensors
    q_A = src.pop(f"{q_prefix}.{a_suffix}")
    k_A = src.pop(f"{k_prefix}.{a_suffix}")
    v_A = src.pop(f"{v_prefix}.{a_suffix}")
    q_B = src.pop(f"{q_prefix}.{b_suffix}")
    k_B = src.pop(f"{k_prefix}.{b_suffix}")
    v_B = src.pop(f"{v_prefix}.{b_suffix}")

    rank = q_A.shape[0]  # LoRA rank

    # Check if lora_A weights are actually shared (all identical)
    if torch.equal(q_A, k_A) and torch.equal(q_A, v_A):
        # Fast path: lora_A IS shared — simple concatenation
        dst[f"{out_prefix}.{a_suffix}"] = q_A
        dst[f"{out_prefix}.{b_suffix}"] = torch.cat([q_B, k_B, v_B], dim=0)
        return

    # Slow path: lora_A differs — merge + SVD re-decomposition
    # Use GPU when available for 10-50x faster SVD on large matrices
    orig_dtype = q_A.dtype
    svd_device = torch.device("cuda") if torch.cuda.is_available() else q_A.device

    # Compute full-rank deltas in float32 for numerical stability
    delta_Q = (q_B.float().to(svd_device) @ q_A.float().to(svd_device))
    delta_K = (k_B.float().to(svd_device) @ k_A.float().to(svd_device))
    delta_V = (v_B.float().to(svd_device) @ v_A.float().to(svd_device))

    # Concatenate: [out_q + out_k + out_v, in_dim]
    delta_fused = torch.cat([delta_Q, delta_K, delta_V], dim=0)

    # SVD to re-decompose into rank-r approximation.
    # Three independent rank-r LoRAs have effective rank up to 3r,
    # so rank-r truncation is lossy.  We keep rank=r so that
    # ComfyUI's alpha/rank scaling stays correct.
    U, S, Vt = torch.linalg.svd(delta_fused, full_matrices=False)
    U_r = U[:, :rank]      # [out_total, rank]
    S_r = S[:rank]          # [rank]
    Vt_r = Vt[:rank, :]    # [rank, in_dim]

    # Split S between A and B (sqrt allocation)
    S_sqrt = torch.sqrt(S_r)
    new_lora_A = (torch.diag(S_sqrt) @ Vt_r).to(dtype=orig_dtype, device="cpu")
    new_lora_B = (U_r @ torch.diag(S_sqrt)).to(dtype=orig_dtype, device="cpu")

    # Diagnostic: measure approximation quality (still on svd_device)
    reconstructed = new_lora_B.float().to(svd_device) @ new_lora_A.float().to(svd_device)
    frob_original = torch.norm(delta_fused).item()
    frob_error = torch.norm(delta_fused - reconstructed).item()
    retained = 1.0 - (frob_error / max(frob_original, 1e-8))

    # measure lora_A divergence
    a_qk_cos = torch.nn.functional.cosine_similarity(
        q_A.float().flatten(), k_A.float().flatten(), dim=0,
    ).item()
    a_qv_cos = torch.nn.functional.cosine_similarity(
        q_A.float().flatten(), v_A.float().flatten(), dim=0,
    ).item()

    logger.info(
        "qkv_svd_fusion",
        prefix=out_prefix,
        rank=rank,
        retained_pct=f"{retained * 100:.1f}%",
        frob_error=f"{frob_error:.4e}",
        cos_qk=f"{a_qk_cos:.4f}",
        cos_qv=f"{a_qv_cos:.4f}",
        device=str(svd_device),
    )

    dst[f"{out_prefix}.{a_suffix}"] = new_lora_A
    dst[f"{out_prefix}.{b_suffix}"] = new_lora_B


def _rename_pair(
    src: dict[str, torch.Tensor],
    dst: dict[str, torch.Tensor],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """Rename lora_A and lora_B weight keys from old_prefix to new_prefix."""
    for suffix in ("lora_A.weight", "lora_B.weight"):
        old_key = f"{old_prefix}.{suffix}"
        new_key = f"{new_prefix}.{suffix}"
        if old_key in src:
            dst[new_key] = src.pop(old_key)


class Flux2Saver(ModelSaver):
    """Save FLUX.2 LoRA weights in BFL-format for ComfyUI inference."""

    def save(
        self, components: dict[str, Any], path: Path, metadata: dict[str, Any] | None = None
    ) -> None:
        """Extract PEFT LoRA weights, convert to BFL format, and save.

        Pipeline:
            1. Extract PEFT state dict
            2. Strip PEFT prefixes → clean diffusers keys
            3. Convert diffusers → BFL keys (QKV fusion + rename)
            4. Add ``diffusion_model.`` prefix
            5. Cast to save dtype and write safetensors

        Args:
            components: Dict containing ``unet`` (the PEFT-wrapped model)
                and optionally ``config`` with save settings.
            path: Output file path.
            metadata: Additional metadata for the safetensors header.
        """
        model = components.get("unet")
        if model is None:
            logger.error("flux2_save_no_model")
            return

        if not hasattr(model, "peft_config"):
            logger.warning("transformer_not_peft_model")
            return

        # 1. Extract PEFT state dict
        peft_sd = get_peft_model_state_dict(model)

        # 2. Clean diffusers keys: strip PEFT prefix, keep only LoRA weights
        diffusers_sd: dict[str, torch.Tensor] = {}
        for key, value in peft_sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            if "lora_A" not in key and "lora_B" not in key:
                continue
            clean = key.replace("base_model.model.", "")
            diffusers_sd[clean] = value

        if not diffusers_sd:
            logger.warning("no_lora_weights_found_to_save")
            return

        # 3. Convert diffusers keys → BFL keys
        bfl_sd = _diffusers_to_bfl(diffusers_sd)

        # 4. Add diffusion_model. prefix
        final_dict: dict[str, torch.Tensor] = {}
        for key, value in bfl_sd.items():
            if not key.startswith("diffusion_model."):
                key = f"diffusion_model.{key}"
            final_dict[key] = value

        # 5. Extract rank/alpha from PEFT config
        config = components.get("config", {})
        rank = 16
        alpha = 16.0
        peft_cfg = next(iter(model.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        # 6. Metadata — enrich with Kohya-compatible ss_ keys
        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": "flux2",
        }

        # Map training config → Kohya ss_ metadata keys
        if config and isinstance(config, dict):
            _MAP = {
                "optimizer_type": "ss_optimizer",
                "lr_scheduler": "ss_lr_scheduler",
                "mixed_precision": "ss_mixed_precision",
                "lora_name": "ss_output_name",
                "definition_id": "ss_sd_model_name",
                "model_family": "ss_base_model_version",
                "learning_rate": "ss_learning_rate",
                "max_train_steps": "ss_steps",
                "train_batch_size": "ss_batch_size_per_device",
                "gradient_accumulation_steps": "ss_gradient_accumulation_steps",
                "noise_offset": "ss_noise_offset",
                "min_snr_gamma": "ss_min_snr_gamma",
                "lr_warmup_steps": "ss_warmup_steps",
                "weight_decay": "ss_weight_decay",
                "seed": "ss_seed",
            }
            for cfg_key, ss_key in _MAP.items():
                val = config.get(cfg_key)
                if val is not None and str(val).strip():
                    save_metadata[ss_key] = str(val)
            save_metadata.update(trigger_metadata(config))

            resolutions = config.get("resolutions")
            if resolutions and isinstance(resolutions, list):
                r = resolutions[0]
                save_metadata["ss_resolution"] = f"({r},{r})"

            if peft_cfg:
                tm = getattr(peft_cfg, "target_modules", None)
                if tm:
                    import json as _json
                    save_metadata["ss_network_args"] = _json.dumps({
                        "target_modules": sorted(tm) if not isinstance(tm, str) else [tm],
                    })

        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        # 7. Save precision (Flux default: bf16)
        save_prec = config.get("save_precision", "bf16").lower()
        save_dtype = (
            torch.float16 if save_prec == "fp16"
            else torch.bfloat16 if save_prec == "bf16"
            else torch.float32
        )
        for k in final_dict:
            final_dict[k] = final_dict[k].to(save_dtype)

        logger.info(
            "flux2_save_lora",
            path=str(path),
            num_tensors=len(final_dict),
            save_dtype=str(save_dtype),
        )

        # 8. Save
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_save_file(final_dict, str(path), metadata=save_metadata)
        logger.info("flux2_save_complete", path=str(path))
