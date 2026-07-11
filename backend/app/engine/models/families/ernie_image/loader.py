"""ERNIE-Image model loader — manifest-driven via GenericComponentLoader.

Components (matching ``ErnieImagePipeline.__init__``):
- Tokenizer: ``transformers.PreTrainedTokenizerFast`` — loads directly
  from ``tokenizer.json``.  ``AutoTokenizer`` fails on this checkpoint
  because ``tokenizer_config.json`` declares
  ``"tokenizer_class": "TokenizersBackend"``, a Baidu-internal class
  name not registered in ``transformers``; the fast-tokenizer loader
  bypasses that class lookup and accepts any ``tokenizer.json``.
- Text encoder: ``transformers.Mistral3Model`` (pinned concrete class).
  The official ``text_encoder/config.json`` declares
  ``architectures=["Mistral3Model"]`` (verified from cache) — exactly what
  ``AutoModel`` resolved to before, so behaviour is unchanged — but pinning
  stops a silent upstream Auto-mapping change from swapping the architecture
  and random-initialising the TE (B1 loading-contract bug class; see
  tests/test_te_loading_contracts.py).
- VAE: ``diffusers.AutoencoderKLFlux2`` (reused from FLUX.2).
- Transformer: ``diffusers.ErnieImageTransformer2DModel`` → mapped to
  ``unet`` key for compatibility with the generic pipeline.

The optional ``pe`` / ``pe_tokenizer`` Prompt Enhancer components are
deliberately omitted — they are a prompt-rewriter LM, not a conditioning
input to the diffusion model.
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class ErnieImageLoader(GenericComponentLoader):
    """Load ERNIE-Image components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.PreTrainedTokenizerFast",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Mistral3Model",
                subfolder="text_encoder",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLFlux2",
                subfolder="vae",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="unet",
                hf_class="diffusers.ErnieImageTransformer2DModel",
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]
