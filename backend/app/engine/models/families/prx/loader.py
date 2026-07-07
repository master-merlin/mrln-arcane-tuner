"""PRX model loader — manifest-driven via GenericComponentLoader.

Uses diffusers ``from_pretrained`` for all components:
- ``PRXTransformer2DModel`` (16-block cross-attention DiT, native 0.39)
- ``T5GemmaEncoder`` (text encoder — see resolution quirk below)
- ``AutoTokenizer`` (resolves the checkpoint's GemmaTokenizerFast)
- ``AutoencoderKL`` (Flux-style 16-channel diffusers VAE)

T5GemmaEncoder resolution quirk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The checkpoint's ``model_index.json`` declares ``["prx", "T5GemmaEncoder"]``
(diffusers resolves that against its own ``pipelines.prx`` module) and the
class is NOT exported at the ``transformers`` top level — so the manifest
must carry the full module path for importlib. Additionally the
``text_encoder/config.json`` is a FLAT ``t5_gemma_module`` config that
``AutoConfig`` rejects; ``T5GemmaEncoder.from_pretrained`` resolves it via
its own ``config_class`` (``T5GemmaConfig``) with a benign model-type
warning — verified to produce the correct 2.61B structure, identical to
what ``PRXPipeline.from_pretrained`` builds.
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class PRXLoader(GenericComponentLoader):
    """Load PRX components — tokenizer, TE, VAE, transformer."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer --
            # AutoTokenizer resolves the fast GemmaTokenizerFast from
            # tokenizer.json (same policy as qwen_image / longcat_image).
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (T5GemmaEncoder — full path, see module doc) --
            ComponentSpec(
                key="text_encoder",
                hf_class=(
                    "transformers.models.t5gemma.modeling_t5gemma.T5GemmaEncoder"
                ),
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.PRXTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]
