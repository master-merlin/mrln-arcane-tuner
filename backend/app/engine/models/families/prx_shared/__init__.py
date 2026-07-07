"""Shared PRX-architecture code (no ``family.py`` — not a family itself).

Consumed by the latent ``prx`` family and the future pixel-space sibling
(``prx_pixel``). Everything here is family-agnostic: no family name,
``architecture_name``, or checkpoint id is hardcoded.
"""

from .forward import prx_transformer_forward as prx_transformer_forward
from .lora_targets import (
    PRX_BLOCK_LORA_TARGETS as PRX_BLOCK_LORA_TARGETS,
    PRX_TARGETS_PER_BLOCK as PRX_TARGETS_PER_BLOCK,
    get_prx_lora_targets as get_prx_lora_targets,
    matching_linear_modules as matching_linear_modules,
)
from .saver_base import PRXSharedLoRASaver as PRXSharedLoRASaver
from .text_encoding import encode_prx_text as encode_prx_text
