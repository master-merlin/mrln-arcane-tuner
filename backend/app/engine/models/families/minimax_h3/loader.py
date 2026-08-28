"""MiniMax-H3 model loader — manifest-driven via GenericComponentLoader.

Component sourcing differs per component, and the split matters:

- Transformer / both VAEs / scheduler are VENDORED (vendor/, pinned diffusers
  SHA 245d78fb). H3 is NOT in any diffusers release — the installed 0.39.0
  contains zero MiniMax code, so a ``diffusers.MiniMaxH3*`` class path would
  ImportError. When upstream ships H3 natively these paths become the only
  thing that changes.
- Text encoder is Qwen3-VL via stock ``transformers`` — no vendoring, the same
  pattern ``nucleus_image`` already proves. Confirmed importable in THIS venv
  (``transformers`` 4.57.0): ``from transformers import
  Qwen3VLForConditionalGeneration, AutoProcessor`` succeeds. The definitions'
  ``architecture_params`` (Task 4) also record ``te.type: qwen3_vl``.

Repo layout (MiniMaxAI/MiniMax-H3, verified via the HF API 2026-08-05):
``transformer/``, ``transformer_ref/``, ``vae/``, ``audio_vae/``,
``text_encoder/``, ``tokenizer/``, ``processor/``, ``scheduler/``,
``audio_scheduler/``, plus ``FL2VA/`` and ``Ref2VA/`` task bundles.

``transformer_ref/`` is a SECOND 33B checkpoint used only by ref2va; the
subfolder comes from the definition's ``architecture_params["transformer.subfolder"]``
(Task 4) so t2va/fl2va never download it.
"""

from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)
from app.engine.core.definitions import ModelDefinition

_VENDOR = "app.engine.models.families.minimax_h3.vendor."


class MiniMaxH3Loader(GenericComponentLoader):
    """Load MiniMax-H3 components — tokenizer, TE, both VAEs, transformer."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        arch = definition.architecture_params or {}
        # ref2va reads transformer_ref/; t2va and fl2va read transformer/.
        transformer_subfolder = arch.get("transformer.subfolder", "transformer")

        return [
            # -- Processor (AutoProcessor is not moved to device — no
            #    .to(device).eval() on a tokenizer/processor object). --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoProcessor",
                subfolder="processor",
                candidates=["processor", "tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen3-VL-32B, cached then unloaded before the
            #    DiT loads — see definition comments). --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="text_encoder",
            ),
            # -- Visual VAE (vendored) --
            ComponentSpec(
                key="vae",
                hf_class=_VENDOR + "autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3",
                subfolder="vae",
            ),
            # -- Audio VAE (vendored, MONO — run once per channel) --
            ComponentSpec(
                key="audio_vae",
                hf_class=_VENDOR
                + "autoencoder_kl_minimax_h3_audio.AutoencoderKLMiniMaxH3Audio",
                subfolder="audio_vae",
            ),
            # -- Transformer (vendored). Subfolder comes from the
            #    definition so ref2va's second 33B checkpoint is never
            #    downloaded by t2va/fl2va. --
            ComponentSpec(
                key="transformer",
                hf_class=_VENDOR
                + "transformer_minimax_h3.MiniMaxH3Transformer3DModel",
                subfolder=transformer_subfolder,
            ),
        ]
