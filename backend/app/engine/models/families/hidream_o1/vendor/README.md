# HiDream-O1 vendored code

This directory contains a curated copy of model and pipeline code from
[Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI)
(MIT license).

## Two upstream sources

| What | Source |
|---|---|
| **Python code** (this directory) | GitHub: [`Saganaki22/HiDream_O1-ComfyUI`](https://github.com/Saganaki22/HiDream_O1-ComfyUI) |
| **Model weights** | HuggingFace Hub: [`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image) |

### Why Saganaki22 and not HiDream-ai's GitHub?

`HiDream-ai/HiDream-O1-Image` on GitHub is inference-only. It does not provide
the custom model class required for training: the checkpoint contains additional
`x_embedder` (proj1/proj2) and `final_layer2.linear` modules that are not present
in stock `Qwen3VLForConditionalGeneration`. Loading via HuggingFace's
`AutoModelForImageTextToText` silently drops these weights.

Saganaki22's ComfyUI integration vendors the actual architecture in
`qwen3_vl_transformers.py` (the custom `Qwen3VLForConditionalGeneration`
subclass with image-patch I/O heads) and re-implements ai-toolkit's May 2026
LoRA training recipe. That is the code we need.

## Vendored files

| File | Lines | Purpose |
|---|---|---|
| `pipeline.py` | 460 | `generate_image()` inference pipeline |
| `qwen3_vl_transformers.py` | 2201 | Custom model class with `x_embedder` + `final_layer2` |
| `flash_scheduler.py` | 445 | Flash flow-match Euler scheduler |
| `fm_solvers_unipc.py` | 800 | UniPC multistep scheduler |
| `seam_smoothing.py` | 149 | Seam-smoothing for tiled generation |
| `utils.py` | 368 | Shared helpers (resize, dimension calc, RoPE index) |
| `compat.py` | 35 | Thin compatibility shim (`TransformersKwargs`, `Unpack`, etc.) |

## Refresh workflow

1. Run `python -m app.engine.models.families.hidream_o1.vendor._refresh --revision <new-sha>` from `backend/`.
2. Inspect the diff. Re-apply or forward-port every `# MRLN-PATCH:` marker (see Patches section below).
3. Update `REVISION` and append a row to `CHANGELOG.md`.
4. Open a PR with the diff. Refreshes are never automatic — they go through review.

The script clones `Saganaki22/HiDream_O1-ComfyUI` at the specified SHA and copies
the seven files listed above. Model weights are NOT touched by this script; they
are pinned separately in the definition YAML's `components.unet.revision` field.

## Patches applied

Two patches are currently applied. Both are marked with `# MRLN-PATCH(N):`
comments and must be forward-ported on every refresh.

1. **Relative import fix in `qwen3_vl_transformers.py`** (`# MRLN-PATCH:` at line 136):
   Saganaki22's package places `compat.py` one level above `models/`, so the
   original import is `from ..compat import ...`. In our `vendor/` package both
   files are at the same level, so the import was changed to `from .compat import ...`.

2. **Gradient-checkpointing kwargs wrap in `qwen3_vl_transformers.py`**
   (`# MRLN-PATCH(4):` around line 1011): torch 2.10's
   `torch.utils.checkpoint.checkpoint` no longer silently forwards arbitrary
   kwargs to the wrapped function; it raises
   `ValueError: Unexpected keyword arguments: ...` when the wrapped layer's
   per-layer kwargs (`attention_mask`, `position_ids`, `past_key_values`,
   `cache_position`, `position_embeddings`) are passed as kwargs through
   `_gradient_checkpointing_func`. The patch wraps the per-layer call in a
   closure (`_layer_call`) that captures those kwargs, so `checkpoint` only
   sees positional args (the closure + hidden_states). Saganaki22's upstream
   depended on the older permissive forwarding; this patch is the version-skew
   fix.

Previously applied patches from Task 2 (against HiDream-ai's pipeline) are **no
longer needed**:
- **flash-attn flag** — Saganaki22's `pipeline.py` already has `use_flash_attn: bool = True`
  as an explicit parameter to `generate_image()`.
- **torch_dtype threading** — Saganaki22's `pipeline.py` derives dtype from
  `model.hidream_dtype` (or falls back to `next(model.parameters()).dtype`);
  there is no hardcoded dtype to override.
- **Gradient checkpointing hook** — still N/A; `qwen3_vl_transformers.py` inherits
  from `transformers.PreTrainedModel` with `supports_gradient_checkpointing = True`.

## NOT vendored

- Prompt-Refine agent (not used in training-time sampling)
- Flask `app.py`, Gradio UI, evaluation scripts
- Gemma-4-31B-it dependency (used only by the optional prompt agent)
- ComfyUI node wrappers (`comfy_runtime.py`, ComfyUI node classes)
- Training helpers (`training/lora.py`, `training/dataset.py`) — these are
  re-implemented natively in our trainer rather than vendored wholesale
