"""
LoRA Tooling — inspect and resize LoRA safetensor files.

Provides standalone utilities (no model loading required):
- **inspect_lora**: Rich analysis matching lora-inspector feature set
- **resize_lora**: SVD-based rank change (up or down)

Inspired by https://github.com/rockerBOO/lora-inspector
"""

import json
import os
import struct
import structlog
import torch
from safetensors.torch import load_file, save_file
from typing import Any

logger = structlog.get_logger(__name__)


# ── Inspect ──────────────────────────────────────────────────────────────


def inspect_lora(path: str) -> dict[str, Any]:
    """
    Inspect a LoRA safetensors file without loading into a model.

    Provides comprehensive analysis including:
    - Format detection (Kohya/ai-toolkit/PEFT)
    - Rank and alpha extraction (from metadata or weight shapes)
    - Weight statistics: average magnitude and strength per component
    - Per-layer breakdown with norms and shapes
    - Training metadata parsing (all ``ss_`` prefixed keys)
    - Tag frequency analysis (from ``ss_tag_frequency``)
    - Dataset info (from ``ss_dataset_config``)
    - Block dim/alpha support (variable rank per block)

    Args:
        path: Absolute path to a ``.safetensors`` file.

    Returns:
        Dict with comprehensive analysis results.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If file cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA file not found: {path}")

    logger.info("inspecting_lora", path=path)

    try:
        state_dict = load_file(path)
    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Failed to load LoRA file: {e}") from e

    # File size
    file_size = os.path.getsize(path)
    file_size_mb = round(file_size / (1024 * 1024), 2)

    # Read safetensors metadata
    metadata = _read_safetensors_metadata(path)

    # Detect format
    fmt = _detect_format(state_dict, metadata)

    # Extract rank from metadata or weight shapes
    rank = _extract_rank(state_dict, metadata)

    # Extract alpha from metadata or alpha tensors
    alpha = _extract_alpha(state_dict, metadata)

    # Component breakdown
    components = _breakdown_components(state_dict)

    # Unique LoRA modules (pairs of A/B or down/up)
    modules = _extract_modules(state_dict)

    # Detect dtype from first weight tensor
    dtype = "unknown"
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 2:
            dtype = str(v.dtype)
            break

    # ── Weight Statistics (magnitude + strength per component) ──
    weight_stats = _compute_weight_stats(state_dict)

    # ── Per-layer breakdown ──
    layer_details = _compute_layer_details(state_dict)

    # ── Training parameters (parsed from ss_ metadata) ──
    training_params = _parse_training_params(metadata)

    # ── Tag frequency ──
    tag_frequency = _parse_tag_frequency(metadata)

    # ── Dataset info ──
    dataset_info = _parse_dataset_info(metadata)

    # ── Block dims/alphas (variable rank per block) ──
    block_config = _parse_block_config(metadata)

    # ── Norm distribution (summary stats across all layers) ──
    norm_summary = _compute_norm_summary(state_dict)

    result = {
        "path": path,
        "file_size_mb": file_size_mb,
        "format": fmt,
        "metadata": metadata,
        "rank": rank,
        "alpha": alpha,
        "total_keys": len(state_dict),
        "components": components,
        "lora_modules": len(modules),
        "module_list": sorted(modules),
        "dtype": dtype,
        # Enhanced fields
        "weight_stats": weight_stats,
        "layer_details": layer_details,
        "training_params": training_params,
        "tag_frequency": tag_frequency,
        "dataset_info": dataset_info,
        "block_config": block_config,
        "norm_summary": norm_summary,
        "layer_relevance": _compute_layer_relevance(layer_details),
    }

    logger.info(
        "lora_inspected",
        rank=rank,
        alpha=alpha,
        format=fmt,
        modules=len(modules),
        keys=len(state_dict),
        size_mb=file_size_mb,
    )

    return result


# ── Resize ───────────────────────────────────────────────────────────────


def resize_lora(
    input_path: str,
    output_path: str,
    new_rank: int,
    new_alpha: float | None = None,
    save_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """
    Resize a LoRA's rank using SVD decomposition.

    For each LoRA module pair (A, B), reconstructs the effective weight
    delta ``W = B @ A``, then re-decomposes it via truncated SVD to the
    target rank. Works for both rank increase (zero-padded) and decrease
    (truncated).

    Args:
        input_path: Path to input ``.safetensors`` file.
        output_path: Path for the resized output file.
        new_rank: Target rank (must be ≥ 1).
        new_alpha: New alpha value. If None, scales proportionally:
                   ``new_alpha = old_alpha * (new_rank / old_rank)``.
        save_dtype: Override save precision. If None, preserves original.

    Returns:
        Dict with ``input_path``, ``output_path``, ``old_rank``,
        ``new_rank``, ``old_alpha``, ``new_alpha``, ``modules_resized``.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If new_rank < 1 or file has no LoRA weight pairs.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"LoRA file not found: {input_path}")

    if new_rank < 1:
        raise ValueError(f"new_rank must be ≥ 1, got {new_rank}")

    logger.info("resizing_lora", input=input_path, output=output_path, new_rank=new_rank)

    state_dict = load_file(input_path)
    metadata = _read_safetensors_metadata(input_path)

    old_rank = _extract_rank(state_dict, metadata)
    old_alpha = _extract_alpha(state_dict, metadata)

    if new_alpha is None:
        if old_alpha is not None and old_rank is not None and old_rank > 0:
            new_alpha = old_alpha * (new_rank / old_rank)
        else:
            new_alpha = float(new_rank)

    # Detect format to know key patterns
    fmt = _detect_format(state_dict, metadata)

    # Build pairs: module_name → (A_key, B_key)
    pairs = _find_lora_pairs(state_dict, fmt)

    if not pairs:
        raise ValueError("No LoRA weight pairs found in file")

    # Process each pair via SVD
    new_dict: dict[str, torch.Tensor] = {}
    resized_count = 0

    for module_name, (a_key, b_key) in pairs.items():
        weight_a = state_dict[a_key].float()  # [rank, in_features] or higher-dim
        weight_b = state_dict[b_key].float()  # [out_features, rank] or higher-dim

        # Save original shapes for restoring after SVD (conv layers may be >2D)
        orig_a_shape = weight_a.shape
        orig_b_shape = weight_b.shape

        # Flatten to 2D for matmul: A → [rank, -1], B → [out, -1]
        a_2d = weight_a.view(weight_a.shape[0], -1)
        b_2d = weight_b.view(weight_b.shape[0], -1)

        # Reconstruct effective delta: W = B @ A
        delta = b_2d @ a_2d  # [out_features, in_features_flat]

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # Truncate or pad to new_rank
        effective_rank = min(new_rank, S.shape[0])

        # New B = U[:, :r] * sqrt(S[:r])  → [out_features, new_rank]
        # New A = sqrt(S[:r]) * Vh[:r, :] → [new_rank, in_features_flat]
        sqrt_s = torch.sqrt(S[:effective_rank])

        new_b = U[:, :effective_rank] * sqrt_s.unsqueeze(0)  # broadcast
        new_a = sqrt_s.unsqueeze(1) * Vh[:effective_rank, :]  # broadcast

        # If new_rank > effective_rank, zero-pad
        if new_rank > effective_rank:
            pad_cols = new_rank - effective_rank
            new_b = torch.cat([new_b, torch.zeros(new_b.shape[0], pad_cols)], dim=1)
            new_a = torch.cat([new_a, torch.zeros(pad_cols, new_a.shape[1])], dim=0)

        # Restore original spatial dims for conv layers (e.g., [rank, C, kH, kW])
        if len(orig_a_shape) > 2:
            new_a = new_a.view(new_rank, *orig_a_shape[1:])
        if len(orig_b_shape) > 2:
            new_b = new_b.view(orig_b_shape[0], new_rank, *orig_b_shape[2:])

        # Determine target dtype
        target_dtype = save_dtype or state_dict[a_key].dtype
        new_dict[a_key] = new_a.contiguous().to(target_dtype)
        new_dict[b_key] = new_b.contiguous().to(target_dtype)
        resized_count += 1

    # Copy non-LoRA-weight keys (alpha tensors, other keys) and update alphas
    alpha_keys = set()
    for key, value in state_dict.items():
        if key not in new_dict:
            if ".alpha" in key:
                # Update alpha tensors to new value
                new_dict[key] = torch.tensor(new_alpha, dtype=torch.float32)
                alpha_keys.add(key)
            else:
                new_dict[key] = value

    # Update metadata
    new_metadata = dict(metadata) if metadata else {}
    new_metadata["ss_network_dim"] = str(new_rank)
    new_metadata["ss_network_alpha"] = str(new_alpha)
    new_metadata["arcane_resized_from_rank"] = str(old_rank)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(new_dict, output_path, metadata=new_metadata)

    output_size = os.path.getsize(output_path)
    output_size_mb = round(output_size / (1024 * 1024), 2)

    result = {
        "input_path": input_path,
        "output_path": output_path,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_alpha": old_alpha,
        "new_alpha": new_alpha,
        "modules_resized": resized_count,
        "output_size_mb": output_size_mb,
    }

    logger.info(
        "lora_resized",
        old_rank=old_rank,
        new_rank=new_rank,
        old_alpha=old_alpha,
        new_alpha=new_alpha,
        modules=resized_count,
        size_mb=output_size_mb,
    )

    return result


# ── Private Helpers ──────────────────────────────────────────────────────


def _read_safetensors_metadata(path: str) -> dict[str, str]:
    """Read metadata header from a safetensors file."""
    try:
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)
            metadata = header.get("__metadata__", {})
            return {k: str(v) for k, v in metadata.items()}
    except (OSError, json.JSONDecodeError, struct.error):
        return {}


def _detect_format(state_dict: dict, metadata: dict) -> str:
    """Detect LoRA format from key patterns and metadata."""
    keys = list(state_dict.keys())

    has_kohya = any("lora_down" in k or "lora_up" in k for k in keys)
    has_peft = any("lora_A" in k or "lora_B" in k for k in keys)
    has_diffusion_model = any("diffusion_model" in k for k in keys)
    has_lora_unet = any(k.startswith("lora_unet") for k in keys)

    if has_kohya and has_lora_unet:
        return "kohya"
    elif has_peft and has_diffusion_model:
        return "ai-toolkit"
    elif has_peft:
        return "peft"
    elif has_kohya:
        return "kohya"
    else:
        return "unknown"


def _extract_rank(state_dict: dict, metadata: dict) -> int | None:
    """Extract rank from metadata or by inspecting weight dimensions."""
    # 1. From metadata (most authoritative)
    meta_rank = metadata.get("ss_network_dim")
    if meta_rank:
        try:
            return int(meta_rank)
        except ValueError:
            pass

    # 2. From weight shapes — collect all lora_A/lora_down shape[0] values
    #    and return the most common (handles conv layers with different shapes)
    ranks: list[int] = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if "lora_A" in key or "lora_down" in key:
            ranks.append(value.shape[0])

    if ranks:
        # Return the most common rank (mode)
        from collections import Counter
        return Counter(ranks).most_common(1)[0][0]

    return None


def _extract_alpha(state_dict: dict, metadata: dict) -> float | None:
    """Extract alpha from metadata or alpha tensors."""
    # 1. From metadata
    meta_alpha = metadata.get("ss_network_alpha")
    if meta_alpha:
        try:
            return float(meta_alpha)
        except ValueError:
            pass

    # 2. From explicit alpha tensors
    for key, value in state_dict.items():
        if ".alpha" in key and isinstance(value, torch.Tensor):
            return float(value.item())

    return None


def _breakdown_components(state_dict: dict) -> dict[str, int]:
    """Count keys by component prefix."""
    breakdown: dict[str, int] = {}

    for key in state_dict:
        prefix = _get_component_prefix(key)
        breakdown[prefix] = breakdown.get(prefix, 0) + 1

    return breakdown


def _get_component_prefix(key: str) -> str:
    """Get human-readable component prefix from a key."""
    if key.startswith("lora_unet"):
        return "unet"
    elif key.startswith("lora_te1"):
        return "text_encoder_1"
    elif key.startswith("lora_te2"):
        return "text_encoder_2"
    elif key.startswith("lora_te"):
        return "text_encoder"
    elif "diffusion_model" in key:
        return "transformer"
    else:
        return "other"


def _extract_modules(state_dict: dict) -> set[str]:
    """Extract unique LoRA module names from state dict."""
    modules = set()

    for key in state_dict:
        # Strip the weight suffix to get module name
        clean = key
        for suffix in (".lora_A.weight", ".lora_B.weight",
                        ".lora_down.weight", ".lora_up.weight", ".alpha"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                modules.add(clean)
                break

    return modules


def _find_lora_pairs(
    state_dict: dict,
    fmt: str,
) -> dict[str, tuple[str, str]]:
    """
    Find matched A/B (or down/up) weight pairs for SVD resize.

    Returns:
        Dict mapping module name → (A_key, B_key).
    """
    pairs: dict[str, tuple[str, str]] = {}

    # Collect A and B keys separately
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}

    for key in state_dict:
        if not isinstance(state_dict[key], torch.Tensor):
            continue

        if "lora_A.weight" in key:
            module = key.replace(".lora_A.weight", "")
            a_keys[module] = key
        elif "lora_down.weight" in key:
            module = key.replace(".lora_down.weight", "")
            a_keys[module] = key
        elif "lora_B.weight" in key:
            module = key.replace(".lora_B.weight", "")
            b_keys[module] = key
        elif "lora_up.weight" in key:
            module = key.replace(".lora_up.weight", "")
            b_keys[module] = key

    # Match pairs
    for module in a_keys:
        if module in b_keys:
            pairs[module] = (a_keys[module], b_keys[module])

    return pairs


# ── Enhanced Inspection Helpers ──────────────────────────────────────────


def _compute_weight_stats(state_dict: dict) -> dict[str, dict[str, float]]:
    """
    Compute per-component weight statistics.

    For each component (unet, text_encoder_1, etc.) calculates:
    - **avg_magnitude**: √(Σ wᵢ²) averaged across all weight tensors
      (Frobenius norm / √n). Measures overall "energy".
    - **avg_strength**: mean(|wᵢ|) averaged across all weight tensors.
      Measures typical weight value.

    Matches lora-inspector's definition:
    - magnitude = sqrt(sum of squares)
    - strength = mean of absolute values
    """
    component_tensors: dict[str, list[torch.Tensor]] = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue  # Skip alpha scalars
        component = _get_component_prefix(key)
        if component not in component_tensors:
            component_tensors[component] = []
        component_tensors[component].append(value.float())

    stats: dict[str, dict[str, float]] = {}
    for component, tensors in component_tensors.items():
        magnitudes = []
        strengths = []
        for t in tensors:
            # Magnitude: Frobenius norm (sqrt of sum of squares)
            magnitudes.append(torch.norm(t, p="fro").item())
            # Strength: mean of absolute values
            strengths.append(torch.mean(torch.abs(t)).item())

        stats[component] = {
            "avg_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
            "avg_strength": sum(strengths) / len(strengths) if strengths else 0.0,
            "num_tensors": len(tensors),
            "total_params": sum(t.numel() for t in tensors),
        }

    return stats


def _compute_layer_details(state_dict: dict) -> list[dict[str, Any]]:
    """
    Compute per-layer details for each LoRA module.

    For each matched A/B pair, computes:
    - Frobenius norm of the effective delta W = B @ A
    - Individual norms of A and B
    - Shapes and parameter counts
    - Component classification
    """
    pairs = _find_lora_pairs(state_dict, "")
    details = []

    for module_name, (a_key, b_key) in sorted(pairs.items()):
        try:
            weight_a = state_dict[a_key].float()
            weight_b = state_dict[b_key].float()

            # Flatten higher-dim tensors (conv layers) to 2D for matmul
            a_2d = weight_a.view(weight_a.shape[0], -1)
            b_2d = weight_b.view(weight_b.shape[0], -1)

            # Effective delta W = B @ A
            delta = b_2d @ a_2d

            detail = {
                "module": module_name,
                "component": _get_component_prefix(a_key),
                "rank": weight_a.shape[0],
                "in_features": int(a_2d.shape[1]),
                "out_features": weight_b.shape[0],
                "params": weight_a.numel() + weight_b.numel(),
                # Norms
                "norm_a": round(torch.norm(weight_a, p="fro").item(), 6),
                "norm_b": round(torch.norm(weight_b, p="fro").item(), 6),
                "norm_delta": round(torch.norm(delta, p="fro").item(), 6),
                # Statistics of effective delta
                "delta_mean": round(delta.mean().item(), 8),
                "delta_std": round(delta.std().item(), 8),
                "delta_max": round(delta.max().item(), 8),
                "delta_min": round(delta.min().item(), 8),
                # Magnitude and strength of the delta
                "magnitude": round(torch.norm(delta, p="fro").item(), 6),
                "strength": round(torch.mean(torch.abs(delta)).item(), 8),
            }

            details.append(detail)
        except RuntimeError:
            # Skip layers that can't be reconstructed (shape incompatibility)
            continue

    return details


def _parse_training_params(metadata: dict) -> dict[str, Any]:
    """
    Parse Kohya-ss style training parameters from metadata.

    Extracts all ``ss_`` prefixed keys and presents them in a
    cleaned, structured format.
    """
    if not metadata:
        return {}

    params: dict[str, Any] = {}

    # Direct string → float/int mappings
    _FLOAT_KEYS = {
        "ss_learning_rate": "learning_rate",
        "ss_unet_lr": "unet_lr",
        "ss_text_encoder_lr": "text_encoder_lr",
        "ss_network_alpha": "alpha",
        "ss_noise_offset": "noise_offset",
        "ss_adaptive_noise_scale": "adaptive_noise_scale",
        "ss_ip_noise_gamma": "ip_noise_gamma",
        "ss_multires_noise_discount": "multires_noise_discount",
        "ss_multires_noise_iterations": "multires_noise_iterations",
        "ss_min_snr_gamma": "min_snr_gamma",
        "ss_scale_weight_norms": "scale_weight_norms",
        "ss_max_grad_norm": "max_grad_norm",
        "ss_network_dropout": "network_dropout",
    }

    _INT_KEYS = {
        "ss_network_dim": "rank",
        "ss_epoch": "epochs",
        "ss_steps": "total_steps",
        "ss_batch_size_per_device": "batch_size",
        "ss_gradient_accumulation_steps": "gradient_accumulation_steps",
        "ss_num_train_images": "train_images",
        "ss_num_reg_images": "regularization_images",
        "ss_num_batches_per_epoch": "batches_per_epoch",
        "ss_mixed_precision": "mixed_precision",
        "ss_seed": "seed",
        "ss_clip_skip": "clip_skip",
        "ss_warmup_steps": "warmup_steps",
        "ss_max_token_length": "max_token_length",
    }

    _STR_KEYS = {
        "ss_optimizer": "optimizer",
        "ss_lr_scheduler": "scheduler",
        "ss_network_module": "network_module",
        "ss_base_model_version": "base_model_version",
        "ss_resolution": "resolution",
        "ss_v2": "sd_v2",
        "ss_debiased_estimation": "debiased_estimation",
        "ss_zero_terminal_snr": "zero_terminal_snr",
        "ss_training_comment": "comment",
        "ss_sd_model_name": "model_name",
        "ss_sd_model_hash": "model_hash",
        "ss_output_name": "output_name",
    }

    for meta_key, param_name in _FLOAT_KEYS.items():
        val = metadata.get(meta_key)
        if val is not None:
            try:
                params[param_name] = float(val)
            except ValueError:
                params[param_name] = val

    for meta_key, param_name in _INT_KEYS.items():
        val = metadata.get(meta_key)
        if val is not None:
            try:
                params[param_name] = int(val)
            except ValueError:
                params[param_name] = val

    for meta_key, param_name in _STR_KEYS.items():
        val = metadata.get(meta_key)
        if val is not None:
            params[param_name] = val

    # Network args (LoCon, etc.)
    net_args = metadata.get("ss_network_args")
    if net_args:
        try:
            params["network_args"] = json.loads(net_args)
        except json.JSONDecodeError:
            params["network_args"] = net_args

    return params


def _parse_tag_frequency(metadata: dict) -> dict[str, list[dict[str, Any]]]:
    """
    Parse tag frequency from ``ss_tag_frequency`` metadata.

    Kohya-ss stores tag frequencies as a JSON dict mapping
    directory names to tag count dicts.

    Returns:
        Dict mapping directory name → sorted list of
        ``{tag, count}`` dicts (descending by count).
    """
    raw = metadata.get("ss_tag_frequency")
    if not raw:
        return {}

    try:
        tag_data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(tag_data, dict):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for directory, tags in tag_data.items():
        if isinstance(tags, dict):
            sorted_tags = sorted(
                [{"tag": tag, "count": count} for tag, count in tags.items()],
                key=lambda x: x["count"],
                reverse=True,
            )
            result[directory] = sorted_tags

    return result


def _parse_dataset_info(metadata: dict) -> dict[str, Any]:
    """
    Parse dataset configuration from metadata.

    Extracts directory names and image counts from
    ``ss_dataset_config`` or related metadata keys.
    """
    dataset: dict[str, Any] = {}

    # Number of images
    train_imgs = metadata.get("ss_num_train_images")
    if train_imgs:
        try:
            dataset["train_images"] = int(train_imgs)
        except ValueError:
            pass

    reg_imgs = metadata.get("ss_num_reg_images")
    if reg_imgs:
        try:
            dataset["regularization_images"] = int(reg_imgs)
        except ValueError:
            pass

    # Dataset config (JSON)
    ds_config = metadata.get("ss_dataset_config")
    if ds_config:
        try:
            parsed = json.loads(ds_config)
            if isinstance(parsed, dict):
                datasets = parsed.get("datasets", [])
                dirs = []
                for ds in datasets:
                    subsets = ds.get("subsets", [])
                    for subset in subsets:
                        img_dir = subset.get("image_dir", "")
                        n_repeats = subset.get("num_repeats", 1)
                        if img_dir:
                            dirs.append({
                                "directory": img_dir,
                                "num_repeats": n_repeats,
                            })
                if dirs:
                    dataset["directories"] = dirs
        except json.JSONDecodeError:
            pass

    return dataset


def _parse_block_config(metadata: dict) -> dict[str, Any]:
    """
    Parse variable block dims/alphas from metadata.

    Some LoRAs use different rank per UNet block (DyLoRA-like).
    Kohya stores ``block_dims`` and ``block_alphas`` in network args.
    """
    config: dict[str, Any] = {}

    # Check network_args first
    net_args = metadata.get("ss_network_args")
    if net_args:
        try:
            args = json.loads(net_args) if isinstance(net_args, str) else net_args
        except json.JSONDecodeError:
            args = {}

        if isinstance(args, dict):
            block_dims = args.get("block_dims")
            if block_dims:
                if isinstance(block_dims, str):
                    config["block_dims"] = [int(x.strip()) for x in block_dims.split(",") if x.strip()]
                elif isinstance(block_dims, list):
                    config["block_dims"] = block_dims

            block_alphas = args.get("block_alphas")
            if block_alphas:
                if isinstance(block_alphas, str):
                    config["block_alphas"] = [float(x.strip()) for x in block_alphas.split(",") if x.strip()]
                elif isinstance(block_alphas, list):
                    config["block_alphas"] = block_alphas

            block_dropout = args.get("block_dropout")
            if block_dropout:
                if isinstance(block_dropout, str):
                    config["block_dropout"] = [float(x.strip()) for x in block_dropout.split(",") if x.strip()]
                elif isinstance(block_dropout, list):
                    config["block_dropout"] = block_dropout

    return config


def _compute_norm_summary(state_dict: dict) -> dict[str, Any]:
    """
    Compute summary statistics of Frobenius norms across all LoRA layers.

    Returns:
        Dict with ``mean_norm``, ``std_norm``, ``max_norm``, ``min_norm``,
        ``max_norm_layer``, ``min_norm_layer``. Useful for detecting
        overtrained (hot) or dead layers.
    """
    pairs = _find_lora_pairs(state_dict, "")
    if not pairs:
        return {}

    norms: list[float] = []
    layer_names: list[str] = []

    for module_name, (a_key, b_key) in pairs.items():
        try:
            weight_a = state_dict[a_key].float()
            weight_b = state_dict[b_key].float()

            # Flatten higher-dim tensors (conv layers) to 2D for matmul
            a_2d = weight_a.view(weight_a.shape[0], -1)
            b_2d = weight_b.view(weight_b.shape[0], -1)

            delta = b_2d @ a_2d
            norm = torch.norm(delta, p="fro").item()
            norms.append(norm)
            layer_names.append(module_name)
        except RuntimeError:
            continue

    if not norms:
        return {}

    norms_tensor = torch.tensor(norms)
    max_idx = int(torch.argmax(norms_tensor).item())
    min_idx = int(torch.argmin(norms_tensor).item())

    return {
        "mean_norm": round(float(norms_tensor.mean().item()), 6),
        "std_norm": round(float(norms_tensor.std().item()), 6),
        "max_norm": round(float(norms[max_idx]), 6),
        "min_norm": round(float(norms[min_idx]), 6),
        "max_norm_layer": layer_names[max_idx],
        "min_norm_layer": layer_names[min_idx],
        "total_layers": len(norms),
    }


def _compute_layer_relevance(layer_details: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Cumulative energy analysis for layer relevance.

    Sorts layers by energy (‖ΔW‖_F²), computes cumulative % of total
    energy, and classifies into tiers:
    - **essential**: Layers contributing to ≤90% of total energy
    - **contributing**: Layers in 90–97% range
    - **negligible**: Bottom 3% — can be skipped for faster training

    Also extracts unique target_module patterns from essential layers
    for direct use in LoRA training configs.

    Returns:
        Dict with tiers, suggested target_modules, speed estimate,
        and per-layer tier classification.
    """
    if not layer_details:
        return {}

    # Compute energy per layer (norm²)
    layers_with_energy = []
    for layer in layer_details:
        norm = layer.get("norm_delta", 0.0)
        layers_with_energy.append({
            "module": layer["module"],
            "component": layer["component"],
            "params": layer.get("params", 0),
            "norm": norm,
            "energy": norm ** 2,
        })

    # Sort by energy descending
    layers_with_energy.sort(key=lambda x: x["energy"], reverse=True)

    total_energy = sum(l["energy"] for l in layers_with_energy)
    total_params = sum(l["params"] for l in layers_with_energy)
    if total_energy <= 0 or total_params <= 0:
        return {}

    # Classify tiers via cumulative energy
    cumulative = 0.0
    essential_layers: list[str] = []
    contributing_layers: list[str] = []
    negligible_layers: list[str] = []
    essential_params = 0
    # Map module → tier for frontend badges
    tier_map: dict[str, str] = {}

    for layer in layers_with_energy:
        pct = cumulative / total_energy
        if pct < 0.90:
            essential_layers.append(layer["module"])
            essential_params += layer["params"]
            tier_map[layer["module"]] = "essential"
        elif pct < 0.97:
            contributing_layers.append(layer["module"])
            tier_map[layer["module"]] = "contributing"
        else:
            negligible_layers.append(layer["module"])
            tier_map[layer["module"]] = "negligible"
        cumulative += layer["energy"]

    total_layers = len(layers_with_energy)

    # Extract unique target_module patterns from essential layers.
    # BFL/ComfyUI keys use merged naming (e.g. img_attn.qkv) while
    # diffusers/PEFT uses split naming (to_q, to_k, to_v).
    # We expand merged patterns so they match model scan output.
    BFL_TO_DIFFUSERS: dict[str, list[str]] = {
        # Flux2/Klein BFL → diffusers (both fused and unfused variants)
        "qkv":     ["to_qkv", "to_q", "to_k", "to_v"],
        "proj":    ["to_out.0", "to_out"],
        "img_attn.qkv":  ["to_qkv", "to_q", "to_k", "to_v"],
        "img_attn.proj": ["to_out.0", "to_out"],
        "txt_attn.qkv":  ["to_added_qkv", "add_q_proj", "add_k_proj", "add_v_proj"],
        "txt_attn.proj": ["to_add_out"],
        "img_mlp.0": ["ff.net.0.proj", "ff.linear_in"],
        "img_mlp.2": ["ff.net.2", "ff.linear_out"],
        "txt_mlp.0": ["ff_context.net.0.proj", "ff_context.linear_in"],
        "txt_mlp.2": ["ff_context.net.2", "ff_context.linear_out"],
        "linear1":  ["linear1", "to_qkv_mlp_proj"],
        "linear2":  ["linear2", "to_out"],
        # Flux2 dev single blocks (diffusers naming)
        "to_qkv_mlp_proj": ["to_qkv_mlp_proj", "linear1"],
        # Diffusers fused → pre-fusion PEFT targets
        "to_qkv":       ["to_q", "to_k", "to_v"],
        "to_added_qkv": ["add_q_proj", "add_k_proj", "add_v_proj"],
        "odd_q_proj":   ["to_q"],
        "odd_k_proj":   ["to_k"],
        "odd_v_proj":   ["to_v"],
    }

    target_module_patterns: set[str] = set()
    for module in essential_layers:
        parts = module.replace(".lora_A", "").replace(".lora_B", "").split(".")
        # Try multi-segment match first (e.g. "img_attn.qkv")
        raw_suffix = ".".join(parts[-2:]) if len(parts) >= 2 else parts[0]
        single_suffix = parts[-1] if parts else ""

        expanded = False
        # Check 2-segment suffix first, then 1-segment
        for key in (raw_suffix, single_suffix):
            if key in BFL_TO_DIFFUSERS:
                target_module_patterns.update(BFL_TO_DIFFUSERS[key])
                expanded = True
                break

        if not expanded:
            # No BFL mapping — use the raw suffix (diffusers-style LoRA)
            target_module_patterns.add(single_suffix if single_suffix else raw_suffix)

    # Speed estimate: ratio of essential params to total params
    speed_ratio = essential_params / total_params if total_params > 0 else 1.0
    speed_gain = round((1.0 / speed_ratio - 1.0) * 100, 0) if speed_ratio > 0 else 0.0

    return {
        "total_layers": total_layers,
        "essential_count": len(essential_layers),
        "contributing_count": len(contributing_layers),
        "negligible_count": len(negligible_layers),
        "essential_modules": essential_layers,
        "contributing_modules": contributing_layers,
        "negligible_modules": negligible_layers,
        "target_module_patterns": sorted(target_module_patterns),
        "essential_params": essential_params,
        "total_params": total_params,
        "essential_params_pct": round(speed_ratio * 100, 1),
        "speed_gain_pct": speed_gain,
        "tier_map": tier_map,
    }
