"""PRXPixelDriver — family-specific training behavior for pixel-space PRX.

Implements ``IModelDriver`` for Photoroom PRXPixel (Photoroom/prxpixel-t2i).

PRXPixel specifics (transformer-level logic shared with the latent ``prx``
family via ``families/prx_shared``):
- PIXEL SPACE: no VAE — "latents" are raw RGB tensors ``[B, 3, H, W]`` in
  ``[-1, 1]``; the transformer patchifies internally (patch_size 16).
- x0 OBJECTIVE: the model predicts the CLEAN image (x-prediction), not a
  velocity. ``compute_target`` therefore returns the clean pixels and the
  training loss is ``MSE(x0_pred, clean_pixels)``. This mirrors the
  pipeline's per-step conversion ``v = (x_t - x0_pred) / clamp(t, 0.05)``
  (``PRXPixelPipeline.__call__``), which is only consistent with a model
  trained to output x0.
- SCALED NOISE: training noise is ``randn × noise_scale`` (2.0) — the same
  scale ``prepare_latents`` applies to the initial sampling noise. Wired
  through ``prepare_noise`` so the generic loop's linear interpolation
  becomes ``x_t = (1-t)·x0 + t·(noise·2.0)``.
- Text encoder: ``Qwen3VLTextModel`` (hidden 2048). Prompt embeds replicate
  ``PRXPixelPipeline._tokenize_prompts``: LIGHT ``_basic_clean`` only (NO
  DeepFloyd lowercasing — differs from latent PRX), tokenize
  ``padding='max_length'`` to ``prompt_max_tokens`` (256, NOT the Qwen
  tokenizer's huge ``model_max_length``), ``last_hidden_state``, BOOLEAN
  attention mask.
- Timestep scale: PRX convention — the transformer receives NORMALIZED
  ``timesteps / 1000`` (via the shared adapter). Exactly once, never twice;
  the scheduler side stays raw ``[0, 1000]``.
- LoRA targets: FUSED projections from ``prx_shared``. The pixel variant's
  bottleneck ``img_in`` becomes ``img_in.0`` / ``img_in.1`` which still
  match no target suffix — the exclusion-free contract holds.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.prx_shared import (
    encode_prx_text,
    get_prx_lora_targets,
    prx_transformer_forward,
)


logger = structlog.get_logger(__name__)

# Pipeline registered config (model_index.json / PRXPixelPipeline defaults).
_DEFAULT_MAX_LENGTH = 256  # prompt_max_tokens
_DEFAULT_NOISE_SCALE = 2.0
_DEFAULT_T_FLOOR = 0.05


def x0_to_velocity(
    latents: torch.Tensor,
    x0_pred: torch.Tensor,
    timestep: torch.Tensor,
    num_train_timesteps: int = 1000,
    t_floor: float = _DEFAULT_T_FLOOR,
) -> torch.Tensor:
    """Convert an x0 prediction to a flow-matching velocity.

    Pipeline-verbatim (``PRXPixelPipeline.__call__``)::

        t_x = clamp(t / num_train_timesteps, min=0.05)
        velocity = (latents - x0_pred) / t_x

    The floor keeps the division stable on the final low-``t`` steps. Lives
    here (next to ``compute_target``) because it IS the model objective's
    inverse — the sampler consumes it per step.

    Args:
        latents: Current trajectory state ``[B, C, H, W]``.
        x0_pred: Model's clean-image prediction, same shape.
        timestep: RAW timestep on the ``[0, num_train_timesteps]`` scale
            (scalar or ``[B]``).
        num_train_timesteps: Scheduler scale (default 1000).
        t_floor: Minimum normalized t for the division (pipeline: 0.05).

    Returns:
        Velocity tensor, same shape as ``latents``.
    """
    t_x = torch.clamp(
        timestep.float() / float(num_train_timesteps),
        min=t_floor,
    ).to(latents.device)
    while t_x.ndim < latents.ndim:
        t_x = t_x.unsqueeze(-1)
    return (latents - x0_pred) / t_x


class PRXPixelDriver(IModelDriver):
    """PRX Pixel family driver.

    Handles:
    - Qwen3-VL prompt encoding (pipeline-identical, basic clean, bool mask)
    - Fused-projection LoRA targets (via prx_shared)
    - Flow-matching scheduler (None)
    - x0 target + ×2.0 training-noise scale
    - Unpacked pixel forward with normalized timesteps (via prx_shared)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components(). NO VAE slot — pixel space.
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None  # stays None; interface compat
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        arch = getattr(definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", _DEFAULT_MAX_LENGTH))
        self.num_train_timesteps = int(
            arch.get("scheduler.num_train_timesteps", 1000),
        )
        self.noise_scale = float(
            arch.get("pipeline.noise_scale", _DEFAULT_NOISE_SCALE),
        )
        self.velocity_t_floor = float(
            arch.get("pipeline.velocity_t_floor", _DEFAULT_T_FLOOR),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded PRXPixel components into driver state (no VAE)."""
        self._components = components
        self.model = components["unet"]
        self.text_encoder = components.get(
            "text_encoder",
            getattr(self, "text_encoder", None),
        )
        self.tokenizer = components.get(
            "tokenizer",
            getattr(self, "tokenizer", None),
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """PRX LoRA targets — fused per-block projections (prx_shared)."""
        definition_targets = getattr(
            self.definition,
            "lora_targetable_modules",
            None,
        )
        if definition_targets and len(definition_targets) > 0:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        self.logger.info("lora_targets_pattern_defaults")
        return get_prx_lora_targets()

    # get_lora_exclude_modules — inherited default (None): the pixel
    # variant's bottleneck img_in.0/img_in.1 Linears match no target suffix
    # — verified by test_shared_targets_match_pixel_variant_no_top_level_sweep.

    def init_scheduler(self) -> Any:
        """PRXPixel uses flow matching — no external scheduler needed."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """PRXPixel loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for PRXPixel."""
        return []

    def get_layer_manifest(self) -> Any:
        """Layer manifest for the single 24-block PRX stack."""
        from app.engine.core.layer_manifest import (  # noqa: PLC0415
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            stack = getattr(model, "blocks", None)
            if stack is not None:
                for i, block in enumerate(stack):
                    blocks.append(
                        BlockInfo(
                            name=f"blocks.{i}",
                            block_type="joint",
                            param_count=sum(p.numel() for p in block.parameters()),
                            depth_index=i,
                        )
                    )

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions replicating ``PRXPixelPipeline._tokenize_prompts``.

        Delegates to :func:`prx_shared.encode_prx_text` with
        ``clean_text=False`` — the pixel pipeline always uses the light
        ``_basic_clean`` (ftfy + HTML unescape; behavior-identical to
        ``TextPreprocessor.basic_clean``, pinned by test) and the training
        token budget ``prompt_max_tokens`` (256) instead of the tokenizer's
        own ``model_max_length``.

        Returns:
            ``TextEncoderOutput`` with embeddings ``[B, 256, 2048]`` and a
            BOOLEAN ``attention_mask`` ``[B, 256]`` (the transformer
            consumes it directly).
        """
        embeddings, attention_mask = encode_prx_text(
            self.tokenizer,
            self.text_encoder,
            captions,
            self.device,
            max_length=self.max_length,
            clean_text=False,
        )
        return TextEncoderOutput(
            embeddings=embeddings.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Scale training noise by the pipeline's ``noise_scale`` (2.0).

        PRXPixel trains with non-unit noise: sampling starts from
        ``randn × noise_scale`` (``prepare_latents``), so the training
        interpolation must blend against the SAME scaled noise —
        ``x_t = (1-t)·x0 + t·(noise·noise_scale)``. The generic loop calls
        this via ``prepare_noise_for_training`` before ``add_noise``.
        """
        return noise * self.noise_scale

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """PRXTransformer2DModel forward via the shared adapter.

        PRX timestep convention: the transformer receives NORMALIZED
        ``t / 1000`` — the division happens HERE (in prx_shared's adapter),
        exactly once. Pixels stay unpacked ``[B, 3, H, W]``; the model
        patchifies internally.

        Args:
            noisy_input: Noisy pixels ``[B, 3, H, W]``.
            timesteps: Flow-matching timesteps on the ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings, bool attention_mask)`` tuple or
                a plain embeddings tensor.
            batch: Full batch dict (unused; interface compat).

        Returns:
            CLEAN-IMAGE (x0) prediction ``[B, 3, H, W]`` — NOT a velocity.
        """
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings
            enc_mask = None

        return prx_transformer_forward(
            self.get_primary_model(),
            noisy_input,
            timesteps,
            enc_hs,
            attention_mask=enc_mask,
            num_train_timesteps=self.num_train_timesteps,
        )

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """x0 objective: the target IS the clean image.

        The pipeline converts the model output per step via
        ``v = (x_t - x0_pred) / t`` — only consistent with a model trained
        on ``MSE(x0_pred, clean_pixels)``. ``noise`` (already ×2.0 via
        ``prepare_noise``) deliberately does not appear here.
        """
        return latents

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return the PRXPixel ai-toolkit-format LoRA saver."""
        from app.engine.models.families.prx_pixel.saver import (  # noqa: PLC0415
            PRXPixelSaver,
        )

        return PRXPixelSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """PRXPixel block topology: one uniform 24-block stack."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            stack = getattr(model, "blocks", None)
            if stack is not None:
                topology.append(
                    {
                        "name": "blocks",
                        "attr_path": "blocks",
                        "count": len(stack),
                        # hidden 3584 × 24 blocks ≈ 292M params/block ≈
                        # 560 MB bf16.
                        "approx_vram_mb": 560,
                    }
                )
        return topology
