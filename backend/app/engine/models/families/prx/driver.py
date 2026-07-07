"""PRXDriver — family-specific training behavior for latent PRX.

Implements ``IModelDriver`` for Photoroom PRX (Photoroom/prx-512-t2i-sft).

PRX specifics (all shared transformer-level logic lives in
``families/prx_shared`` so the future pixel-space sibling reuses it):
- Text encoder: ``T5GemmaEncoder`` (hidden 2304). Prompt embeds replicate
  ``PRXPipeline.encode_prompt`` exactly: DeepFloyd-style text cleaning,
  tokenize ``padding='max_length'`` to ``tokenizer.model_max_length``
  (256), ``last_hidden_state``, BOOLEAN attention mask. No slicing, no
  zero-masking.
- Transformer: ``PRXTransformer2DModel`` — patchify/unpatchify happen
  INSIDE the model, so the driver passes unpacked ``[B, 16, H, W]``
  latents straight through (no Flux-style packing).
- Timestep scale: PRX convention — the driver passes NORMALIZED
  ``timesteps / 1000`` (the model's ``time_factor=1000`` re-scales
  internally for the embedding). Exactly once, never twice; the
  scheduler side stays raw ``[0, 1000]``.
- LoRA targets: FUSED projections from ``prx_shared`` (no to_q/to_k/to_v
  exist in this architecture).
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

# tokenizer.model_max_length of the sft checkpoint's GemmaTokenizerFast.
_DEFAULT_MAX_LENGTH = 256


class PRXDriver(IModelDriver):
    """PRX family driver.

    Handles:
    - T5Gemma prompt encoding (pipeline-identical, bool mask)
    - Fused-projection LoRA targets (via prx_shared)
    - Flow-matching scheduler (None)
    - Unpacked-latent forward with normalized timesteps (via prx_shared)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        arch = getattr(definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", _DEFAULT_MAX_LENGTH))
        self.num_train_timesteps = int(
            arch.get("scheduler.num_train_timesteps", 1000),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded PRX components into driver state."""
        self._components = components
        self.model = components["unet"]
        self.vae = components["vae"]
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

    # get_lora_exclude_modules — inherited default (None): none of the
    # fused-projection suffixes collide with the transformer's top-level
    # Linears (img_in / txt_in / time_in.* / final_layer.*) — verified by
    # test_shared_targets_match_tiny_model_and_fused_shapes.

    def init_scheduler(self) -> Any:
        """PRX uses flow matching — no external scheduler needed."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """PRX loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for PRX."""
        return []

    def get_layer_manifest(self) -> Any:
        """Layer manifest for the single 16-block PRX stack."""
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
        """Encode captions replicating ``PRXPipeline.encode_prompt``.

        Delegates to :func:`prx_shared.encode_prx_text` (DeepFloyd cleaning,
        max_length padding, ``last_hidden_state``, bool mask).

        Returns:
            ``TextEncoderOutput`` with embeddings ``[B, 256, 2304]`` and a
            BOOLEAN ``attention_mask`` ``[B, 256]`` (the transformer
            consumes it directly).
        """
        embeddings, attention_mask = encode_prx_text(
            self.tokenizer,
            self.text_encoder,
            captions,
            self.device,
            max_length=self.max_length,
        )
        return TextEncoderOutput(
            embeddings=embeddings.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

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
        exactly once. Latents stay unpacked ``[B, C, H, W]``; the model
        patchifies internally.

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]``.
            timesteps: Flow-matching timesteps on the ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings, bool attention_mask)`` tuple or
                a plain embeddings tensor.
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 16, H, W]``.
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
        """Standard flow-match velocity target: ``noise - latents``."""
        return noise - latents

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return the PRX ai-toolkit-format LoRA saver."""
        from app.engine.models.families.prx.saver import PRXSaver  # noqa: PLC0415

        return PRXSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """PRX block topology: one uniform 16-block stack."""
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
                        "approx_vram_mb": 150,
                    }
                )
        return topology
