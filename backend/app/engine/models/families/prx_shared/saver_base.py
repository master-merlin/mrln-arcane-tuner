"""Shared PRX LoRA saver base — canonical ai-toolkit keys.

No usable upstream LoRA mixin exists for PRX: ``PRXPipeline`` inherits the
legacy ``LoraLoaderMixin`` whose ``_lora_loadable_modules`` are
``["unet", "text_encoder"]`` — it cannot target the PRX ``transformer``.
OUR canonical ai-toolkit keys are therefore the format of record::

    diffusion_model.blocks.{i}.{module}.lora_A.weight
    diffusion_model.blocks.{i}.{module}.lora_B.weight

Family-agnostic: ``architecture_name`` is intentionally left EMPTY here and
must be set by each family's concrete subclass (e.g. the latent family and
the future pixel-space sibling each stamp their own
``modelspec.architecture``).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class PRXSharedLoRASaver(GenericLoRASaver):
    """Base saver for PRX-architecture families.

    Subclasses MUST set ``architecture_name`` — nothing PRX-family-specific
    is hardcoded here beyond the shared key format documented above (which
    ``GenericLoRASaver`` already implements).
    """

    architecture_name: str = ""
