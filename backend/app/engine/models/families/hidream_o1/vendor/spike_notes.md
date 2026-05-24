# HiDream-O1 Spike Notes

This document captures findings from PR A (the spike). Each section is filled
in by the corresponding spike task. PR B references these notes directly.

## Task 2 — Vendored upstream files

**Upstream repo URL:** `https://github.com/HiDream-ai/HiDream-O1-Image`
- The implementation plan assumed `HiDream-ai/HiDream-O1`, but that repo does not exist (GitHub 404). The correct repo is `HiDream-ai/HiDream-O1-Image`. Both the HuggingFace model card README and GitHub search confirm this. The `_refresh.py` script's `UPSTREAM_REPO` was updated accordingly.

**Pinned revision:** `4e56686b857587e6723eb542f73bcab48b19c9ee` (latest commit on `main` as of 2026-05-24, message: "Update model links to include ModelScope repository").

**Files copied:**
- `models/pipeline.py` → `vendor/pipeline.py` (437 lines). Pure inference pipeline: `generate_image()` function + helpers. No model class definition.

**Upstream `models/` directory also contained** (not vendored — inference helpers, would bloat the vendor package and are not needed for training):
- `__init__.py`, `flash_scheduler.py`, `fm_solvers_unipc.py`, `qwen3_vl_transformers.py`, `utils.py`

**Patches applied:**

- **Patch 1 (flash-attn flag):** Added `use_flash_attn: bool = True` parameter to `generate_image()` signature and plumbed it through to the `forward_once` inner function (replacing hardcoded `True` at what is now line ~345). Upstream HF card says line 341 — confirmed the line in the actual file.

- **Patch 2 (torch_dtype threading):** Added `dtype: torch.dtype | None = None` parameter to `generate_image()`. The upstream code hardcoded `dtype = torch.bfloat16` inside the function body; replaced with `dtype = dtype or torch.bfloat16` so callers can override (e.g., pass `torch.float32` for training without flash-attn).

- **Patch 3 (gradient checkpointing):** **SKIPPED.** `pipeline.py` contains no model class — it is a pure inference pipeline. The model class (`Qwen3VLPreTrainedModel` in `models/qwen3_vl_transformers.py`) already inherits from `transformers.PreTrainedModel` with `supports_gradient_checkpointing = True` and blocks inherit from `GradientCheckpointingLayer`. No stub needed.

**Surprises:**
- The repo name divergence (plan said `HiDream-O1`, actual is `HiDream-O1-Image`) was the only significant surprise.
- `pipeline.py` is a flat inference script, not an OO pipeline class — no `__init__` or model constructor to patch for dtype; the dtype param was added to the `generate_image` function instead.
- The upstream code uses `dtype = torch.bfloat16` (not `torch.float32` as the plan assumed). Default preserved as bfloat16.

## Task 3 — Model loading & training API

**Method:** Used `accelerate.init_empty_weights()` + `AutoModelForImageTextToText.from_config(cfg)` to inspect the model class without downloading ~35 GB of weights. Equivalent for our purpose (we only need the class shape, not behavior). The full-weight introspection script (`spike_introspect.py`) was attempted but stalled at the start of the HF download phase (exit 0 with only 70 KB cached — config files only); root cause not investigated since `init_empty_weights` answered every question.

**Resolved model class:** `Qwen3VLForConditionalGeneration` (architecture `qwen3_vl`).

**MRO:**
```
Qwen3VLForConditionalGeneration
  -> Qwen3VLPreTrainedModel
  -> PreTrainedModel
  -> Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin,
     PeftAdapterMixin, GenerationMixin, ContinuousMixin, object
```

Notable: inherits `PeftAdapterMixin` — peft LoRA support is first-class.

**Forward signature** (verbatim from `inspect.signature(model.forward)`):
```
forward(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]
```

The returned `Qwen3VLCausalLMOutputWithPast` is a **causal LM output** with `.logits` over the **token vocabulary** — NOT pixel-space tensors. This is the key training-time finding.

**Methods of interest:**
- `training_forward`: **MISSING** (no separate training entrypoint — use `forward()` with `labels`).
- `enable_gradient_checkpointing`: **MISSING** (note the name — the existing method is `gradient_checkpointing_enable`, see below).
- `gradient_checkpointing_enable`: **present** (callable). Use this in the trainer.
- `supports_gradient_checkpointing` attribute: **`True`**.

**LoRA module patterns** (from `model.named_modules()`):
- Vision side: `model.visual.blocks.N.attn.qkv`, `model.visual.blocks.N.attn.proj`, `model.visual.blocks.N.mlp.linear_fc1`, `model.visual.blocks.N.mlp.linear_fc2`
- Language side: not enumerated within the 40-module cap, but the standard Qwen3 layer naming applies (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

**Excluded-module presence check:**
- `lm_head`: **1 match** (the LM head; exclude from LoRA).
- `patch_embed`: **2 matches** (`model.visual.patch_embed`, `model.visual.patch_embed.proj`).
- `visual`: **297 matches** — the **entire vision encoder** lives under `model.visual.*`. The exclusion `"visual"` in `LoraConfig.exclude_modules` will therefore freeze the full vision tower, which is what we want for text-to-image LoRA on the language side.

**Critical implication for Task 4 / Task 11:**

The plan's pseudocode in spec §4.6 / plan Task 11 implies a pixel-space MSE loss against `noise * noise_scale - image`. **That does not compile against this model:** `forward()` returns token logits, not a pixel-shaped tensor. The ai-toolkit May 2026 recipe must operate at a different level — most likely:
- Using hidden states at the image-token positions, OR
- Through a separate generation head that maps tokens → pixels, OR
- Via the `pipeline.py` `generate_image()` machinery (which uses `forward_once` to predict next-step pixels through some image-token decoder path).

The vendored `pipeline.py` is **inference-only** and does not show the training math. Saganaki22's `HiDream_O1-ComfyUI` repo re-implements ai-toolkit's recipe — Task 4 should derive the exact loss formulation from that source before attempting a 100-step convergence run. A literal implementation of the plan's pseudocode against this model will fail at the loss-shape mismatch.

**Environment notes (informational, not blockers):**
- `transformers` 4.57.0; `torch` 2.10.0+cu130 (warning: cpp extensions skipped, advises ≥2.11.0 — not blocking).
- HF cache root is at `D:\AI\huggingface\hub\hub\` (note the doubled `hub\hub\` — looks like an environment-variable quirk, but downloads work).
- Symlinks not active on Windows without dev-mode; degraded cache (HF warning) — not blocking.
- GPU confirmed: RTX PRO 6000 Blackwell, 102.6 GB VRAM.

## Task 4 — Recipe convergence

(Filled in by Task 4.)

## Task 5 — VRAM measurements

(Filled in by Task 5.)
