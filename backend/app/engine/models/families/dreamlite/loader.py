"""DreamLite model loader — manifest-driven via GenericComponentLoader.

All four components are diffusers-0.39 / transformers-4.57-native and load
via ``from_pretrained`` (no vendoring, no config translation):

- ``DreamLiteUNetModel`` — the primary model lives under the checkpoint's
  ``unet/`` subfolder (it IS a U-Net; ``model_index.json`` names the
  component "unet", unlike the DiT families' ``transformer/``).
- ``transformers.Qwen3VLModel`` text encoder (base model, no LM head —
  its ``hidden_states[-1]`` is identical to the pipeline's
  ``Qwen3VLForConditionalGeneration`` tap for text-only inputs). The
  checkpoint TE config is saved by transformers 4.57.3 (verified), so the
  krea2-style rope-config translation is NOT needed.
- ``AutoTokenizer`` (resolves the checkpoint's fast Qwen2TokenizerFast).
- ``AutoencoderTiny`` (taesdxl — 4 latent channels, encode returns
  ``.latents``, no ``latent_dist``).

The checkpoint repos pin ``revision="diffusers"`` via the definition's
``huggingface:<repo>@diffusers`` URI (resolved by ``ModelPathResolver``).
The ``processor`` component (Qwen3VLProcessor) is edit-mode-only and is
deliberately NOT loaded — the diptych "[Edit]" mode is out of scope.
"""

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)


class DreamLiteLoader(GenericComponentLoader):
    """Load DreamLite components — tokenizer, TE, VAE, unet."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen3-VL, hidden 2048) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE (AutoencoderTiny) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderTiny",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Primary model: DreamLiteUNetModel under unet/ --
            # ``definition_key="unet"`` + ``separate_repo=True`` let a future
            # definition pin the unet to its own repo; with no ``unet``
            # component declared (dreamlite-base/mobile) this falls back to
            # discovering ``unet/`` inside the repo root.
            ComponentSpec(
                key="unet",
                hf_class="diffusers.DreamLiteUNetModel",
                subfolder="unet",
                candidates=["unet"],
                definition_key="unet",
                separate_repo=True,
            ),
        ]
