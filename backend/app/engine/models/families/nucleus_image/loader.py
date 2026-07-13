"""Nucleus-Image model loader — manifest-driven via GenericComponentLoader.

All four components are diffusers-0.39/transformers-native and load via
``from_pretrained`` (no vendoring, no config translation) — confirmed by
reading the installed package directly (not the HF blog, which is vague on
this point):

- ``NucleusMoEImageTransformer2DModel`` — 32-layer single-stream DiT with
  optional MoE FFN (first 3 blocks dense, last 29 MoE per the checkpoint's
  ``dense_moe_strategy: "leave_first_three_blocks_dense"``). Verified
  importable top-level: ``diffusers/__init__.py`` (``diffusers`` 0.39.0,
  this venv), source at
  ``diffusers/models/transformers/transformer_nucleusmoe_image.py``.
- ``transformers.Qwen3VLForConditionalGeneration`` text encoder
  (``model_index.json``: ``["transformers", "Qwen3VLForConditionalGeneration"]``
  — the full VLM class; only used in text-only mode here, same pattern as
  ``krea2``/``dreamlite``).
- ``transformers.Qwen3VLProcessor`` — NOT a bare tokenizer. The real
  pipeline's ``_format_prompt`` calls ``self.processor.apply_chat_template(...)``
  and ``encode_prompt`` calls ``self.processor(text=..., ...)`` directly, so
  this driver needs the full processor object (chat template + tokenizer),
  loaded from the repo's ``processor/`` subfolder (ships
  ``chat_template.json`` + ``tokenizer.json`` + ``merges.txt``/``vocab.json``
  — a Qwen3-VL BPE tokenizer, NOT SentencePiece). Mapped to the driver's
  ``tokenizer`` slot (this codebase's convention for "whatever object
  tokenizes captions" — see ``lumina2``/``qwen_image``).
- ``diffusers.AutoencoderKLQwenImage`` — the SAME VAE class already resident
  for the ``qwen_image`` family (``vae/config.json``:
  ``"_class_name": "AutoencoderKLQwenImage"``, ``z_dim=16``,
  ``temperal_downsample=[false,true,true]`` — a 3-D causal VAE used here in
  single-frame/image mode). No new VAE code needed.

Repo layout confirmed via the HF API (``NucleusAI/Nucleus-Image``,
2026-07-13, no download needed): ``transformer/``, ``text_encoder/``,
``vae/``, ``processor/``, ``scheduler/``, ``model_index.json`` — standard
diffusers multi-folder layout, ~46 GB total (bf16 shards + tokenizer/
processor assets).
"""

from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)
from app.engine.core.definitions import ModelDefinition


class NucleusImageLoader(GenericComponentLoader):
    """Load Nucleus-Image components — processor, TE, VAE, transformer."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Processor (chat template + Qwen3-VL BPE tokenizer) --
            # Mapped to the driver's "tokenizer" slot per this codebase's
            # convention; the object is a full Qwen3VLProcessor because
            # `_format_prompt`/`encode_prompt` both need it (see module
            # docstring).
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.Qwen3VLProcessor",
                subfolder="processor",
                candidates=["processor"],
                is_torch_model=False,
            ),
            # -- Text Encoder --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLQwenImage",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.NucleusMoEImageTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
                definition_key="transformer",
                separate_repo=True,
            ),
        ]
