# FLUX.3 vendored code

A copy of the FLUX.3 transformer from the open diffusers pull request
[huggingface/diffusers#15046](https://github.com/huggingface/diffusers/pull/15046)
(Apache-2.0), pinned at head `ec288418a5efa32e9fd49eb7b193babce920a106`.

## Upstream sources

| What | Source |
|---|---|
| **Python code** (`transformer_flux3.py`) | `src/diffusers/models/transformers/transformer_flux3.py` @ `ec288418` |
| **Model weights** | HuggingFace Hub: [`black-forest-labs/flux-3-image-klein-base`](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (gated) |

## Why only the transformer?

The VAE (`AutoencoderKLFlux2`) and the text encoder (`Qwen3VLForConditionalGeneration`) are stock
classes of the installed diffusers 0.41.0 and transformers 5.14.1 and are loaded directly. The only
local change to the file is that its package-relative imports are absolute `diffusers.` imports.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | lazy re-export of `Flux3Transformer2DModel` (never raises at import) |
| `transformer_flux3.py` | searched: no `transformer_flux3.py` and no `Flux3` symbol in the 0.41.0 wheel (grep of `site-packages/diffusers/`); PR #15046 is unmerged. Retire when a diffusers release ships `Flux3Transformer2DModel` |
