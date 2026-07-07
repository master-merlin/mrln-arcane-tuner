"""PRXPixel LoRA saver -- saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight`` — the shared
PRX-architecture ai-toolkit format (via ``prx_shared``). No usable upstream
LoRA loader mixin exists for PRX pipelines — these canonical keys are the
format of record (pinned by test_prx_pixel_lora_portability.py: 288 keys
for the 24-block prxpixel-t2i checkpoint).

``architecture_name`` is stamped ``"prx_pixel"`` — deliberately DIFFERENT
from the latent sibling's ``"prx"``: the two architectures are not
LoRA-portable (in_channels 16 vs 3, hidden 1792 vs 3584, depth 16 vs 24,
plain vs bottleneck img_in), and the metadata stamp is what lets a loader
reject a cross-family file up front.
"""

from app.engine.models.families.prx_shared import PRXSharedLoRASaver


class PRXPixelSaver(PRXSharedLoRASaver):
    """Saves pixel-space PRX model weights as LoRA safetensors."""

    architecture_name = "prx_pixel"
