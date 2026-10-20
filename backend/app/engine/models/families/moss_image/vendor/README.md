# MOSS-Image vendored code

The MOSS-Image transformer and scheduler, copied from the checkpoint's own `pipeline_moss_image.py`
(Apache-2.0: "The model weights and bundled inference code are released under the Apache License,
Version 2.0"), pinned in `REVISION`. The upstream loads it with `trust_remote_code=True`; this
project never does -- the code is vendored and reviewed instead (ARCHITECTURE D10 invariant 1).

## Upstream sources

| What | Source |
|---|---|
| **Python code** (`transformer_moss_image.py`, upstream lines 56-261) | `OpenMOSS-Team/MOSS-Image-Base` @ `1ceede80badddd6a7812069619edaa5633587192`, `pipeline_moss_image.py` |
| **Python code** (`scheduler_moss_image.py`, upstream lines 264-287) | same file, same revision |
| **Model weights** | [`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), [`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (Apache-2.0, ungated) |

The SFT repository's `pipeline_moss_image.py` at `fc5de0d56fc910cbca6f6625a70f5f582b26deb1` is
byte-identical (`cmp`, `_harness/research/moss-image-family.md` §1.1).

## What changed

- `MOSSImagePipeline` (upstream lines 290-440) is dropped: the family's own sampler replaces it.
- The import blocks keep only the names the kept classes use; `AutoModel` / `AutoTokenizer` are gone,
  so nothing here loads remote code.
- Every class and function body is byte-for-byte upstream.

## Why only these two?

The VAE (`AutoencoderKLFlux2`) and the text encoder (`Qwen3_5Model`, `Qwen2Tokenizer`) are stock
classes of the installed diffusers 0.41.0 and transformers 5.14.1 and are loaded directly.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | lazy re-export of `MOSSImageTransformer`, `MOSSImageScheduler` (never raises at import) |
| `transformer_moss_image.py` | searched: no MOSS file or `MOSSImage` symbol in the 0.41.0 wheel (grep of `site-packages/diffusers/` hits only `cosmos` files); no upstream PR. Retire when a diffusers release ships it |
| `scheduler_moss_image.py` | same search; the shift-then-snap-to-the-training-grid schedule is MOSS's own subclass of `FlowMatchEulerDiscreteScheduler`. Retire with the transformer |
