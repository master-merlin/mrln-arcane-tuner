# HiDream-O1 vendored code

This directory contains a curated copy of inference-pipeline code from
[HiDream-ai/HiDream-O1](https://github.com/HiDream-ai/HiDream-O1).

The HF model repo (`HiDream-ai/HiDream-O1-Image`) is a flat weights-only repo
with no custom Python. The pipeline code lives in HiDream-ai's GitHub repo
separately — that is what we vendor here.

## Refresh workflow

1. Run `python -m app.engine.models.families.hidream_o1.vendor._refresh --revision <new-sha>` from `backend/`.
2. Inspect the diff. Re-apply or forward-port every `# MRLN-PATCH:` marker.
3. Update `REVISION` and append a row to `CHANGELOG.md`.
4. Open a PR with the diff. Refreshes are never automatic — they go through review.

## Patches applied

Each patch is marked with a `# MRLN-PATCH:` comment in the vendored file.
Currently applied:

1. **flash-attn flag externalized** — upstream `pipeline.py` hardcodes
   `use_flash_attn: True`. Patched to read from a constructor arg.
2. **`torch_dtype` threaded through** — upstream defaults to F32; we want bf16.
3. **Gradient checkpointing hook** — exposed `enable_gradient_checkpointing()`
   if not inherited from `transformers.PreTrainedModel`.

## NOT vendored

- Prompt-Refine agent (not used in training-time sampling)
- Flask `app.py`, Gradio UI, evaluation scripts
- Gemma-4-31B-it dependency (used only by the optional prompt agent)
