"""LTX 2.3 model loader — manifest-driven via GenericComponentLoader.

Components (diffusers / Lightricks ``LTX-2`` repo layout):
- Transformer: ``LTX2VideoTransformer3DModel``      (subfolder ``transformer``)
- VAE (video): ``AutoencoderKLLTX2Video``            (subfolder ``vae``)
- Text encoder: ``Gemma3ForConditionalGeneration``  (subfolder ``text_encoder``)
- Tokenizer: ``AutoTokenizer``                       (subfolder ``tokenizer``)
- Connectors: ``LTX2TextConnectors``                 (subfolder ``connectors``)

Audio components are LAZY / CONDITIONAL — only declared when audio training is
requested AND the dataset supplies audio:
- Audio VAE: ``AutoencoderKLLTX2Audio``              (subfolder ``audio_vae``)
- Vocoder: ``LTX2Vocoder``                           (subfolder ``vocoder``,
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

    def __init__(self, device, *, train_audio: bool = False) -> None:
        super().__init__(device)
        self.train_audio = bool(train_audio)

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        specs: list[ComponentSpec] = [
            # -- Tokenizer (Gemma3) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text encoder (Gemma3 → connectors) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Gemma3ForConditionalGeneration",
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
                hf_class="diffusers.LTX2VideoTransformer3DModel",
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]

        if self.train_audio:
            specs.extend(self._audio_specs())

        return specs

    @staticmethod
    def _audio_specs() -> list[ComponentSpec]:
        """Audio VAE + vocoder specs (only when training audio).

        The audio VAE encodes a waveform → audio latents (≈4× temporal
        compression).  The vocoder is SAMPLING-only — it decodes audio
        latents back to a waveform and is loaded lazily at sample time, but
        we declare it here so a single manifest covers the audio run.
        """
        return [
            ComponentSpec(
                key="audio_vae",
                hf_class="diffusers.AutoencoderKLLTX2Audio",
                subfolder="audio_vae",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="vocoder",
                hf_class="diffusers.pipelines.ltx2.LTX2Vocoder",
                subfolder="vocoder",
                fallback_to_root=True,
            ),
        ]
