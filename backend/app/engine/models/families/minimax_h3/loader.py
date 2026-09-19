"""MiniMax-H3 model loader — manifest-driven via GenericComponentLoader.

Component sourcing differs per component, and the split matters:

- Transformer / both VAEs / scheduler are the INSTALLED diffusers classes
  (``MiniMaxH3Transformer3DModel``, ``AutoencoderKLMiniMaxH3``,
  ``AutoencoderKLMiniMaxH3Audio``, ``MiniMaxH3Scheduler``): diffusers 0.40.0,
  the ``requirements.txt`` floor, ships the whole H3 stack. The family's
  earlier ``vendor/`` fork (pre-release SHA 245d78fb) was retired in plan row
  1.8 once its forward proved bit-exact with upstream on the tiny arch
  (``tests/fixtures/h3_vendor_parity.npz``). Two upstream behaviours the data
  path must respect: the video VAE casts its inputs to the encoder/decoder
  parameter dtype (``_keep_in_fp32_modules``), and the audio VAE declares
  ``_supports_group_offloading = False`` — never group-offload it.
  ``diffusers_ships_native_h3()`` probes the installed package for the class.
- Text encoder is Qwen3-VL via stock ``transformers``, the same pattern
  ``nucleus_image`` already proves. Confirmed importable in THIS venv
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


def diffusers_ships_native_h3(class_name: str = "MiniMaxH3Transformer3DModel") -> bool:
    """``True`` when the installed ``diffusers`` exports ``class_name``.

    A real lookup against the installed package (imported here, not at module
    import — nothing imported at startup may raise, ARCHITECTURE D1), so a
    downgrade below the 0.40.0 floor answers ``False`` instead of failing at
    ``from_pretrained`` after the download.
    """
    try:
        import diffusers

        return getattr(diffusers, class_name, None) is not None
    except Exception:  # noqa: BLE001 — a broken install is "does not ship it"
        return False


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
            # -- Visual VAE (diffusers; casts inputs to its fp32 modules) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLMiniMaxH3",
                subfolder="vae",
            ),
            # -- Audio VAE (diffusers, MONO — run once per channel; no
            #    group offloading, see the module docstring) --
            ComponentSpec(
                key="audio_vae",
                hf_class="diffusers.AutoencoderKLMiniMaxH3Audio",
                subfolder="audio_vae",
            ),
            # -- Transformer (diffusers). Subfolder comes from the
            #    definition so ref2va's second 33B checkpoint is never
            #    downloaded by t2va/fl2va. --
            ComponentSpec(
                key="transformer",
                hf_class="diffusers.models.transformers.transformer_minimax_h3"
                ".MiniMaxH3Transformer3DModel",
                subfolder=transformer_subfolder,
            ),
        ]
