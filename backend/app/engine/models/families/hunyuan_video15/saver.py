"""HunyuanVideo 1.5 LoRA saver — ai-toolkit-format PEFT keys.

diffusers 0.39 ships NO pipeline-level LoRA loader mixin for HunyuanVideo 1.5
(the old ``HunyuanVideoLoraLoaderMixin`` targets the ORIGINAL HunyuanVideo
transformer's key layout, which the 1.5 model does not share) — so our
ai-toolkit-style keys are the record::

    diffusion_model.transformer_blocks.{i}.attn.to_q.lora_A.weight
    diffusion_model.transformer_blocks.{i}.ff.net.0.proj.lora_B.weight
    ...

The t2v and i2v checkpoints share one transformer layout (verified hub
configs: identical except ``task_type``), so a LoRA trained on either mode
round-trips losslessly onto the other (pinned by the portability test).
"""

from __future__ import annotations

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class Hv15Saver(GenericLoRASaver):
    """Save HunyuanVideo 1.5 LoRA weights in ai-toolkit-compatible format."""

    architecture_name = "hunyuanvideo-1.5"

    def __init__(self, mode: str = "t2v") -> None:
        self.mode = str(mode).lower() if mode else "t2v"
        # Instance attr shadows the class attr — records the trained mode in
        # ``modelspec.architecture`` (portability across modes still holds).
        self.architecture_name = f"hunyuanvideo-1.5-{self.mode}"
