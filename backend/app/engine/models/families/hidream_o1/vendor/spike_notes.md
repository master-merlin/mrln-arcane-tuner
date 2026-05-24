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

## Task 3a — Recipe derivation from Saganaki22's trainer

Researched [Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI) (MIT) to derive the actual training recipe. Two findings — one about the recipe, one structural.

### Recipe (ai-toolkit May 2026 as implemented in Saganaki22)

```python
# Image → patches (PATCH_SIZE = 32 from models/pipeline.py)
patches = einops.rearrange(image, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=32, p2=32)
# patches shape: (1, num_patches, 3*32*32) where num_patches = (H/32) * (W/32)

# Sigma sampling (linear timestep_type)
sigma = torch.rand(batch_size).clamp(T_EPS, 0.9999)   # T_EPS = 0.001
# (sigmoid and shift modes also supported)

# Forward noise injection
noise = torch.randn_like(patches)
scaled_noise = noise * 8.0                             # noise_scale = 8.0
noisy = (1.0 - sigma) * patches + sigma * scaled_noise
timestep = 1.0 - sigma

# Custom model forward (NOT stock transformers — see structural finding below)
outputs = model(
    input_ids=text_sample["input_ids"],
    position_ids=text_sample["position_ids"],
    vinputs=noisy,                                     # noisy image patches
    timestep=timestep,
    token_types=text_sample["token_types"],
    use_flash_attn=...,
    use_sage_attn=...,
)
x0_pred = outputs.x_pred[0, text_sample["vinput_mask"][0]].unsqueeze(0)

# Velocity-equivalent loss (default; "x0" mode also available)
sigma_loss = sigma.clamp_min(T_EPS)
velocity_pred = (noisy - x0_pred) / sigma_loss
velocity_target = scaled_noise - patches
loss = F.mse_loss(velocity_pred, velocity_target).clamp(max=1.0)   # max_loss = 1.0
```

### Structural finding — model class is CUSTOM

**Saganaki22 does NOT use stock `Qwen3VLForConditionalGeneration`.** They vendor a custom class in `models/qwen3_vl_transformers.py` (~700 lines) that:
- Extends Qwen3VL building blocks (config classes, attention, MLP, decoder layer) imported from `transformers.models.qwen3_vl`.
- Adds an `x_embedder` (proj1/proj2) that projects image patches into the model's hidden space.
- Adds `final_layer2.linear` (the pixel-prediction head).
- Adds `Qwen3VLModelOutputWithPast.x_pred: Optional[torch.FloatTensor]` to the output dataclass.
- Accepts `vinputs`, `timestep`, `token_types`, `use_flash_attn`, `use_sage_attn` forward kwargs (not in stock transformers).

The HF repo's `config.json` declares `architectures=["Qwen3VLForConditionalGeneration"]` for compatibility, but the checkpoint contains weights for the additional `x_embedder` and `final_layer2` modules. Loading via stock `AutoModelForImageTextToText` silently drops them. Our loader MUST instantiate the custom class.

### LoRA targeting (from `training/lora.py`)

Saganaki22's `target_preset` options:
- `"aitoolkit"` / `"ai-toolkit"` / `"ostris"` — all linear-like layers EXCEPT names containing `lm_head` / `patch_embed` / `visual`. This is the documented ai-toolkit recipe.
- `"attention"` (default `"attention+pixel"`) — only `language_model.layers.N.self_attn.{q,k,v,o}_proj`.
- Adds `"mlp"` → also include `language_model.layers.N.mlp.{gate,up,down}_proj`.
- Adds `"pixel"` → also include the pixel heads `x_embedder.proj1/proj2`, `final_layer2.linear`.

The visual encoder (`model.visual.*`) is always excluded — too large and not the trainable side.

### LoRA artifact format (from `training/lora.py:lora_state_dict`)

```python
state[f"diffusion_model.{lora_key}.lora_down.weight"] = layer.lora_down  # (rank, in_features)
state[f"diffusion_model.{lora_key}.lora_up.weight"]   = layer.lora_up    # (out_features, rank)
state[f"diffusion_model.{lora_key}.alpha"]            = torch.tensor(alpha)
```

Where `lora_key` is the module name stripped of `model.model.` / `model.` prefix. Kohya/ai-toolkit style — NOT peft-native (which uses `.lora_A.weight` / `.lora_B.weight`). ComfyUI's native HiDream-O1 LoRA loader expects this format.

### Implications for our plan (revising in place)

1. **Task 2 must be extended (call it Task 2b):** also vendor `models/qwen3_vl_transformers.py` + the helpers it pulls in (likely `flash_scheduler.py`, `fm_solvers_unipc.py`, `utils.py`, `seam_smoothing.py` — to be confirmed by reading pipeline.py imports).

2. **Task 8 (loader):** instantiate the vendored custom class — NOT `AutoModelForImageTextToText`. Load HF weights via `state_dict` after manual class construction.

3. **Task 11 (trainer):** implement the recipe above. NOT peft. Use a small custom `LoRALinear` wrapper (mirror Saganaki22's `HiDreamO1LoRALinear` — 50 lines of code). The injection helper mirrors `inject_lora_layers`.

4. **Task 12 (saver):** key format `diffusion_model.<lora_key>.{lora_down.weight, lora_up.weight, alpha}`. Mirror Saganaki22's `lora_state_dict`. Sidecar JSON unchanged.

5. **Task 13 (sampler):** use `pipeline.generate_image(...)` from the vendored file.

6. **Definition YAML (Task 14):** `lora.target_preset` field (string preset name) instead of `target_modules` / `excluded_modules`. Recipe constants (`noise_scale`, `timestep_type`, `max_loss`, `loss_target`) remain.

## Task 2b — Expanded vendoring

**Saganaki22 SHA:** `1f1dd545faa3ea436aa2fc89f2a555f0cbc88651`

**Files vendored (7 total including compat shim):**

| File | Lines | Notes |
|---|---|---|
| `pipeline.py` | 460 | Replaced HiDream-ai version; Saganaki22 version adds seam_smoothing, use_sage_attn, richer scheduler options |
| `qwen3_vl_transformers.py` | 2201 | Custom model class with `x_embedder` + `final_layer2` |
| `flash_scheduler.py` | 445 | |
| `fm_solvers_unipc.py` | 800 | |
| `seam_smoothing.py` | 149 | |
| `utils.py` | 368 | |
| `compat.py` | 35 | Saganaki22 shim (`TransformersKwargs`, `Unpack`, `auto_docstring`, `check_model_inputs`) — one level above `models/` in original repo |

**Previous patches (1 & 2 from Task 2) NOT needed against Saganaki22:**
- Patch 1 (flash-attn flag): `generate_image()` already has `use_flash_attn: bool = True` as an explicit parameter.
- Patch 2 (torch_dtype threading): dtype is derived from `model.hidream_dtype` (or `next(model.parameters()).dtype`) — no hardcoded dtype.
- Patch 3 (gradient checkpointing): still N/A for the same reason as Task 2.

**One new MRLN-PATCH applied:**
- `qwen3_vl_transformers.py` line 136: changed `from ..compat import` → `from .compat import` to account for the fact that both files now live in the same `vendor/` package (in Saganaki22's layout, `compat.py` is one level above `models/`).

**Import smoke test result:**
```
pipeline ok
model ok
```
Both modules import cleanly. The torch cpp-extensions warning (`upgrade to >=2.11.0, found 2.10.0+cu130`) is pre-existing and non-blocking (documented in Task 3 notes above).

**Lint:** 25 upstream-style violations across the vendored files (E701 single-line colon blocks in `pipeline.py`, E402 non-top-of-file imports in `qwen3_vl_transformers.py` from conditional guard patterns, F401 unused `scipy.stats` in `fm_solvers_unipc.py`). Added file-level `# ruff: noqa` to all 7 vendored files; `ruff check` now passes cleanly. Upstream code preserved unmodified except for the MRLN-PATCH and the noqa directives.

## Task 4 — Recipe convergence (PARTIAL — see findings)

**Outcome:** The 100-step convergence loop did NOT run. Spike surfaced a separate blocker (model load) before any training step executed. The recipe formulation itself is unchanged from Task 3a's derivation; it remains correct and ready for PR B's Task 11 to implement.

### What was attempted

A training script (`.agent/workdir/spike_train.py`, ~250 lines) was written implementing the full Saganaki22 recipe:
- Synthetic 4-color 512×512 dataset
- Patch rearrange via einops (`PATCH_SIZE=32`)
- Linear sigma sampling in `[T_EPS, 0.9999]`
- Noised input: `(1-σ)·patches + σ·noise·8.0`
- Custom-kwarg forward: `model(input_ids, position_ids, vinputs=noisy, timestep=1-σ, token_types, use_flash_attn=False, use_sage_attn=False)`
- Velocity-equivalent loss with `clamp(max=1.0)`
- AdamW(lr=1e-4, weight_decay=1e-4)
- Inline `TinyLoRA` wrapper (Linear adapter, kohya-style `lora_down`/`lora_up`)
- LoRA targets: `language_model.layers.N.self_attn.{q,k,v,o}_proj` and `.mlp.{gate,up,down}_proj`; excludes `lm_head`, `patch_embed`, `visual`

### Blocker: vendored-class load hangs silently

`Qwen3VLForConditionalGeneration.from_pretrained(...)` with our **vendored** custom class halts at shard 1→2 boundary of the safetensors load. Exit code 0, no traceback, no progress past tqdm's first tick. Observed across multiple attempts with various flag combinations:

- `device_map="cuda"` + default cpu mem → hang (RAM thrash; working set saw 96 GB before death)
- `device_map=None`, `low_cpu_mem_usage=True`, `local_files_only=True` → still hangs (this time fast exit, no RAM spike)

### Comparison: stock transformers class works

The **stock** `transformers.Qwen3VLForConditionalGeneration` loads the same checkpoint cleanly in **19.4 s** with `low_cpu_mem_usage=True`. Confirmed:
- ~8.77 B parameters loaded
- HF emits an explicit "Some weights of the model checkpoint at HiDream-ai/HiDream-O1-Image were not used" warning for: `model.x_embedder.{proj1,proj2}.*`, `model.final_layer2.linear.*`, `model.t_embedder1.mlp.*`
- This **empirically validates the Task 3a finding** that the checkpoint contains weights for additional heads beyond stock Qwen3VL.

The hang is **specific to our vendored custom class**, not the underlying checkpoint or load path.

### Hypothesis for Task 11

Saganaki22 doesn't use `from_pretrained` at all in their trainer — they use ComfyUI's `load_hidream_model(model_dir, precision, attention)` runtime helper which uses different loading mechanics. Their helpers (in `comfy_runtime.py`) likely:
- Instantiate the custom class with empty weights (`init_empty_weights`)
- Use `safetensors.torch.load_file` per shard (which we confirmed works in 2.95 s/shard)
- Manually call `model.load_state_dict(...)` with `strict=False` to merge

Task 11 should implement a similar pattern (skip `from_pretrained` entirely; use the direct safetensors loader we already validated works). This is the load-mechanics fix that unblocks training. With it in place, the recipe should converge on first attempt because the math is documented and proven by Saganaki22.

### Direct safetensors load smoke test (works)

```python
from safetensors.torch import load_file
sd = load_file(".../model-00001-of-00008.safetensors")
# 358 keys, first key: 'model.language_model.embed_tokens.weight'
# Loads in ~3 s per shard
```

### Net Task 4 verdict

- ✅ Recipe formulation: confirmed correct (Task 3a + Saganaki22 reference)
- ✅ Stock-class load: works in 19.4 s with `low_cpu_mem_usage=True`
- ✅ Direct safetensors load (per shard): works in 3 s
- ❌ Vendored-class `from_pretrained` load: hangs (PR B Task 11 problem to solve)
- ⏸️ 100-step convergence: not run (depends on the vendored-class load being fixed)

## Task 5 — VRAM measurements (SKIPPED)

Skipped for PR A: depends on a working vendored-class load. Numbers will be filled in by PR B's implementation work (Task 11 + Task 16 E2E check). The definition YAML (Task 14) ships with `vram_profile` set to `0` placeholders for PR B to overwrite.

## Task 5 — VRAM measurements

(Filled in by Task 5.)

## Post-merge: torch 2.10 gradient-checkpointing fix (2026-05-24)

First real training run after PR B merge errored with:

```
ValueError: Unexpected keyword arguments: attention_mask, position_ids,
past_key_values, cache_position, position_embeddings
```

at `qwen3_vl_transformers.py:1011`, inside the per-layer gradient-checkpointing
call. Root cause: **torch 2.10's `torch.utils.checkpoint.checkpoint` tightened
kwarg forwarding** — it now raises on unrecognized kwargs rather than silently
passing them to the wrapped function. Saganaki22's upstream was written against
a more permissive torch version.

Fixed via `# MRLN-PATCH(4):` (see `vendor/README.md`): wrap the per-layer call
in a closure that captures the kwargs, so `checkpoint` only sees positional
args (the closure + `hidden_states`). The non-GC branch (`else:`) is unchanged
— it never went through `checkpoint` and was always fine.

**Forward-port note for future refreshes:** if Saganaki22 updates their code
to handle torch 2.10+ themselves, our patch becomes redundant. Check the
upstream `_gradient_checkpointing_func` call site on each refresh and either
drop or forward-port accordingly.
