"""SDXL model loader — manifest-driven via GenericComponentLoader.

Loads Stable Diffusion XL components (UNet, dual CLIP text encoders,
tokenizers, VAE) via HuggingFace ``from_pretrained`` with subfolder
resolution from a single base repository or explicit per-component paths.
"""

import torch

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class SDXLLoader(GenericComponentLoader):
    """Load SDXL components from HuggingFace repos with subfolder resolution.

    Uses ``ComponentSpec.root_key="unet"`` to resolve the repo root from
    the definition's ``unet`` component path, and ``separate_repo=True``
    for the VAE to handle standalone VAE repositories declaratively.
    """

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizers --
            ComponentSpec(
                key="tokenizer_1",
                hf_class="transformers.CLIPTokenizer",
                subfolder="tokenizer",
                definition_key="tokenizer_1",
                is_torch_model=False,
                use_subfolder_kwarg=True,
                root_key="unet",
            ),
            ComponentSpec(
                key="tokenizer_2",
                hf_class="transformers.CLIPTokenizer",
                subfolder="tokenizer_2",
                definition_key="tokenizer_2",
                is_torch_model=False,
                use_subfolder_kwarg=True,
            ),
            # -- Text Encoders --
            ComponentSpec(
                key="text_encoder_1",
                hf_class="transformers.CLIPTextModel",
                subfolder="text_encoder",
                definition_key="text_encoder_1",
                use_subfolder_kwarg=True,
            ),
            ComponentSpec(
                key="text_encoder_2",
                hf_class="transformers.CLIPTextModelWithProjection",
                subfolder="text_encoder_2",
                definition_key="text_encoder_2",
                use_subfolder_kwarg=True,
            ),
            # -- VAE (may be in a separate repo) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                definition_key="vae",
                dtype_override=torch.float32,
                use_subfolder_kwarg=True,
                separate_repo=True,
            ),
            # -- UNet --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.UNet2DConditionModel",
                subfolder="unet",
                definition_key="unet",
                use_subfolder_kwarg=True,
            ),
        ]
