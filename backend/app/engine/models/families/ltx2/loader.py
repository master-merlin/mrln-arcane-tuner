"""LTX-2 model loader — manifest-driven via GenericComponentLoader.

Serves every definition in the family, so two of the component classes are read
from the definition rather than fixed here (see ``DEFAULT_TE_CLASS`` /
``DEFAULT_VOCODER_CLASS``); the defaults are LTX-2.3's values.

Components (diffusers / Lightricks ``LTX-2`` repo layout):
- Transformer: ``LTX2VideoTransformer3DModel``      (subfolder ``transformer``)
- VAE (video): ``AutoencoderKLLTX2Video``            (subfolder ``vae``)
- Text encoder: per definition ``te._class_name``    (subfolder ``text_encoder``)
- Tokenizer: ``AutoTokenizer``                       (subfolder ``tokenizer``)
- Connectors: ``LTX2TextConnectors``                 (subfolder ``connectors``)

Audio components are LAZY / CONDITIONAL — only declared when audio training is
requested AND the dataset supplies audio:
- Audio VAE: ``AutoencoderKLLTX2Audio``              (subfolder ``audio_vae``)
- Vocoder: per definition ``vocoder._class_name``    (subfolder ``vocoder``,
  SAMPLING-only — decodes audio latents back to a waveform).

``LTX2TextConnectors`` and ``LTX2Vocoder`` are NOT exported at the top level of
``diffusers`` 0.38.0 — they live in ``diffusers.pipelines.ltx2``.  The fully
qualified class paths below resolve them through ``GenericComponentLoader``.
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Ltx2Loader(GenericComponentLoader):
    """Load LTX 2.3 from a diffusers-format repository.

    Audio components are appended only when ``train_audio`` is requested.  The
    flag is read from the run config (passed through the loader at construction
    by the trainer); absent config defaults to video-only so unit tests and
    audio-free runs never reach for the audio VAE / vocoder.
    """

    #: Component classes that differ BETWEEN definitions of this family, with the
    #: LTX-2.3 value as the default so an existing definition that names neither
    #: keeps loading exactly what it loaded before.
    #:
    #: Read from the definition rather than hardcoded because 2.5 changes both:
    #: the text encoder moves Gemma3 -> Gemma4Unified, and the vocoder moves
    #: ``LTX2Vocoder`` -> ``LTX2VocoderWithBWE`` (16 kHz in, 48 kHz out). The
    #: vocoder half is easy to miss -- it is reached only on an audio run, so a
    #: hardcoded class would load a 2.3 vocoder against 2.5 weights and surface
    #: as a shape error deep in sampling rather than as a load failure.
    #:
    #: Concrete classes, never an ``Auto*`` mapping: an upstream mapping change
    #: would otherwise silently random-initialise the text encoder, which is the
    #: loading-contract bug class pinned by tests/test_te_loading_contracts.py.
    DEFAULT_TE_CLASS = "Gemma3ForConditionalGeneration"
    DEFAULT_VOCODER_CLASS = "LTX2Vocoder"

    #: The transformer class is resolved the same way, but from a FULLY QUALIFIED
    #: path rather than a bare name, because it is the one component whose class
    #: may not live in ``diffusers`` at all: if a definition ever needs a flag the
    #: installed diffusers ignores, the replacement is a vendored module here.
    #: Silently ignoring a declared flag is the failure mode that matters -- the
    #: weights load, the shapes agree, and the model is quietly not the one the
    #: checkpoint describes.
    #:
    #: The override key is ``transformer._module``, NOT ``transformer._class_name``:
    #: the latter already exists in both definitions, holding the BARE class name
    #: the config harvester read out of the checkpoint. That is a released schema
    #: key describing what the checkpoint says; redefining it as "where to import
    #: from" would change the meaning of a frozen key (ARCHITECTURE D2) and make
    #: the harvested value and the import path the same field.
    DEFAULT_TRANSFORMER_CLASS = "diffusers.LTX2VideoTransformer3DModel"

    def __init__(self, device, *, train_audio: bool = False) -> None:
        super().__init__(device)
        self.train_audio = bool(train_audio)

    @staticmethod
    def _arch(definition: ModelDefinition) -> dict:
        return getattr(definition, "architecture_params", {}) or {}

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        arch = self._arch(definition)
        te_class = str(arch.get("te._class_name") or self.DEFAULT_TE_CLASS)
        specs: list[ComponentSpec] = [
            # -- Tokenizer (Gemma3) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text encoder (Gemma3 for 2.3, Gemma4Unified for 2.5 → connectors) --
            ComponentSpec(
                key="text_encoder",
                hf_class=f"transformers.{te_class}",
                subfolder="text_encoder",
                fallback_to_root=True,
            ),
            # -- Text connectors (Gemma3 hidden states → video/audio text emb) --
            ComponentSpec(
                key="connectors",
                hf_class="diffusers.pipelines.ltx2.LTX2TextConnectors",
                subfolder="connectors",
                fallback_to_root=True,
            ),
            # -- Video VAE (5D [B, C, F, H, W]; spatial 32×, temporal 8×) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLLTX2Video",
                subfolder="vae",
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class=str(
                    arch.get("transformer._module") or self.DEFAULT_TRANSFORMER_CLASS
                ),
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]

        if self.train_audio:
            specs.extend(self._audio_specs(definition))

        return specs

    def _audio_specs(self, definition: ModelDefinition) -> list[ComponentSpec]:
        """Audio VAE + vocoder specs (only when training audio).

        The audio VAE encodes a waveform → audio latents (≈4× temporal
        compression).  The vocoder is SAMPLING-only — it decodes audio
        latents back to a waveform and is loaded lazily at sample time, but
        we declare it here so a single manifest covers the audio run.
        """
        vocoder_class = str(
            self._arch(definition).get("vocoder._class_name") or self.DEFAULT_VOCODER_CLASS
        )
        return [
            ComponentSpec(
                key="audio_vae",
                hf_class="diffusers.AutoencoderKLLTX2Audio",
                subfolder="audio_vae",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="vocoder",
                hf_class=f"diffusers.pipelines.ltx2.{vocoder_class}",
                subfolder="vocoder",
                fallback_to_root=True,
            ),
        ]
