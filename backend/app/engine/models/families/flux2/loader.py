"""FLUX.2 model loader — manifest-driven via GenericComponentLoader.

Components:
- Transformer: ``Flux2Transformer2DModel``
- VAE: ``AutoencoderKLFlux2``
- Text encoder: ``Qwen3ForCausalLM`` (Klein) or ``Mistral3ForConditionalGeneration`` (Dev)
- Tokenizer: ``AutoTokenizer`` (Klein) or ``AutoProcessor`` (Dev)
"""


from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class Flux2Loader(GenericComponentLoader):
    """Load FLUX.2 Klein / Dev from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        # Determine text encoder variant
        arch = getattr(definition, "architecture_params", {}) or {}
        te_type = arch.get("te.model_type", "qwen3")

        if te_type == "mistral3":
            te_class = "transformers.Mistral3ForConditionalGeneration"
            tok_class = "transformers.AutoProcessor"
        else:
            # Pin the concrete class instead of ``AutoModelForCausalLM``. The
            # Klein ``text_encoder/config.json`` declares
            # ``architectures=["Qwen3ForCausalLM"]`` (verified: klein-base-4B,
            # klein-9B), which is exactly what ``AutoModelForCausalLM`` resolves
            # to today — so this is behaviour-preserving — but pinning stops a
            # silent upstream Auto-mapping change from swapping in a different
            # architecture and random-initialising the TE (the B1 loading-contract
            # bug class; see tests/test_te_loading_contracts.py).
            te_class = "transformers.Qwen3ForCausalLM"
            tok_class = "transformers.AutoTokenizer"

        return [
            # -- Tokenizer / Processor --
            ComponentSpec(
                key="tokenizer",
                hf_class=tok_class,
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder --
            ComponentSpec(
                key="text_encoder",
                hf_class=te_class,
                subfolder="text_encoder",
                fallback_to_root=True,
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLFlux2",
                subfolder="vae",
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.Flux2Transformer2DModel",
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]
