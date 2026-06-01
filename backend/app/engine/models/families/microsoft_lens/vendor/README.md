# Microsoft Lens vendored code

Curated copy of the Lens DiT from [microsoft/Lens](https://github.com/microsoft/Lens) (MIT license).

## Upstream sources

| What | Source |
|---|---|
| **Python code** (`transformer.py`) | GitHub: [`microsoft/Lens`](https://github.com/microsoft/Lens) (`lens/transformer.py`) |
| **Model weights** | HuggingFace Hub: [`microsoft/Lens-Base`](https://huggingface.co/microsoft/Lens-Base) |

## Why only `transformer.py`?

Our trainer never uses Lens's own `LensPipeline`. We supply our own
loader/driver/trainer, so we vendor only the custom DiT class
(`LensTransformer2DModel`). The VAE (`AutoencoderKLFlux2`), scheduler
(`FlowMatchEulerDiscreteScheduler`), tokenizer (`PreTrainedTokenizerFast`),
and text encoder (`GptOssForCausalLM`) are stock and loaded directly.

`transformer.py` imports only from `diffusers` (our pinned 0.38.0) and `torch`
-- no `transformers` dependency, so it is immune to the upstream
`transformers==5.8.0` pin. **We stay on transformers 4.57.0.**

## NOT vendored
- `lens/text_encoder.py` (`LensGptOssEncoder`) -- we run stock `GptOssForCausalLM`
  and slice hidden layers in the driver (decoupled; no transformers-version coupling).
- `lens/pipeline.py`, `lens/reasoner.py`, `lens/resolution.py` -- inference-only.

## Patches applied
None. The file is used verbatim. Mark any future edit with `# MRLN-PATCH:`.

## Refresh
Re-copy `lens/transformer.py` from the pinned upstream SHA, diff, and re-apply any `# MRLN-PATCH:` markers.
