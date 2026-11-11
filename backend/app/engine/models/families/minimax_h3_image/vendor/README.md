# MiniMax-H3-Image vendored code

Copies of the MiniMax-H3-Image transformer and VAE from diffusers main, merged in
[huggingface/diffusers#15230](https://github.com/huggingface/diffusers/pull/15230)
(Apache-2.0), pinned at the merge commit `875373ac7f5856071e345a3e394c74efccc6b9e6`
(milestone 0.43.0; in no diffusers release yet). `REVISION` names the commit and the two
upstream paths.

## Upstream sources

| What | Source |
|---|---|
| **Python code** (`transformer_minimax_h3_image.py`) | `src/diffusers/models/transformers/transformer_minimax_h3_image.py` @ `875373ac` |
| **Python code** (`autoencoder_kl_minimax_h3_image.py`) | `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py` @ `875373ac` |
| **Model weights** | HuggingFace Hub: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) @ `fbde899f` (`transformer_diffusers/`, `vae_diffusers/`) |

## Why these two, and not the scheduler?

The installed diffusers 0.41.0 ships the MiniMax-H3 *video* classes (`MiniMaxH3Transformer3DModel`,
`AutoencoderKLMiniMaxH3`) but neither image class. The text encoder (`Qwen3VLForConditionalGeneration`)
is a stock transformers class and is loaded directly. The scheduler is not vendored: the sampler's
update is one line, written in the family's sampler.

The only local change to either file is that its package-relative imports are absolute `diffusers.`
imports, under a provenance comment; every class and function body is byte-for-byte upstream.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | lazy re-export of `MiniMaxH3ImageTransformer2DModel` and `AutoencoderKLMiniMaxH3Image` (never raises at import) |
| `transformer_minimax_h3_image.py` | searched: no `transformer_minimax_h3_image.py` and no `MiniMaxH3ImageTransformer2DModel` in the 0.41.0 wheel (grep of `site-packages/diffusers/`; the only `MiniMaxH3Image*` name there is the video pipeline's `MiniMaxH3ImageReference` dataclass). Retire when a diffusers release ships `MiniMaxH3ImageTransformer2DModel` |
| `autoencoder_kl_minimax_h3_image.py` | searched: no `autoencoder_kl_minimax_h3_image.py` and no `AutoencoderKLMiniMaxH3Image` in the 0.41.0 wheel (grep of `site-packages/diffusers/`). Retire when a diffusers release ships `AutoencoderKLMiniMaxH3Image` |
