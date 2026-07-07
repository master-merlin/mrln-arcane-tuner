"""OvisImageDriver — family-specific training behavior for Ovis-Image.

Implements ``IModelDriver`` for Ovis-Image (AIDC-AI/Ovis-Image-7B).

Ovis-Image specifics:
- Text encoder: plain ``transformers.Qwen3Model`` (hidden 2048). Prompt
  embeds replicate ``OvisImagePipeline._get_ovis_prompt_embeds`` exactly:
  system-prompt prefix + chat template (``enable_thinking=False``),
  tokenize with ``max_length = max_sequence_length + user_prompt_begin_id``
  and ``add_special_tokens=False``, take ``last_hidden_state``, zero the
  padding positions, then slice off the first ``user_prompt_begin_id``
  (28) template tokens.
- Transformer: ``OvisImageTransformer2DModel`` — Flux-style packed tokens.
  The driver packs ``[B, 16, H, W]`` latents 2×2 → ``[B, L, 64]`` (reusing
  the flux1 packing utils, byte-identical to the pipeline's
  ``_pack_latents``), builds Flux img_ids and Ovis txt_ids (columns 1 AND
  2 carry ``arange(text_len)``), and unpacks the prediction back to
  ``[B, 16, H, W]``.
- Timestep scale: the driver passes ``timesteps / 1000.0`` — the
  transformer multiplies by 1000 internally. Exactly once, never twice.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# Fixed system prompt prepended to every user prompt — copied verbatim from
# ``OvisImagePipeline.__init__`` (diffusers 0.39).
OVIS_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, quantity, text, shape, "
    "size, texture, spatial relationships of the objects and background: "
)

# Number of chat-template tokens preceding the user prompt; the pipeline
# slices these off the encoder output (``user_prompt_begin_id``).
_DEFAULT_USER_PROMPT_BEGIN_ID = 28

# ``OvisImagePipeline.encode_prompt`` default / hard maximum.
_DEFAULT_MAX_SEQUENCE_LENGTH = 256


class OvisImageDriver(IModelDriver):
    """Ovis-Image family driver.

    Handles:
    - Qwen3 masked/sliced prompt encoding (pipeline-identical)
    - Flux-style packed flow-matching forward pass
    - Double + single stream LoRA targets (final proj_out excluded)
    - bf16 loading dtype; no external scheduler
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
        self.max_sequence_length = int(
            arch.get("te.max_sequence_length", _DEFAULT_MAX_SEQUENCE_LENGTH),
        )
        self.user_prompt_begin_id = int(
            arch.get("te.user_prompt_begin_id", _DEFAULT_USER_PROMPT_BEGIN_ID),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Ovis-Image components into driver state."""
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
        """Ovis-Image LoRA targets — double + single stream projections."""
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
        return [
            # Double stream (OvisImageTransformerBlock, num_layers=6)
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            # Single stream (OvisImageSingleTransformerBlock, num_single_layers=27)
            "proj_mlp",
            "proj_out",
        ]

    def get_lora_exclude_modules(self) -> str | list[str] | None:
        """Exclude the model's FINAL projection from LoRA.

        ``proj_out`` as a suffix target also matches the top-level output
        projection. Returned as a regex STRING so PEFT applies
        ``re.fullmatch`` — it hits only the module literally named
        ``proj_out`` while ``single_transformer_blocks.N.proj_out`` stays
        targeted.
        """
        return "proj_out"

    def init_scheduler(self) -> Any:
        """Ovis-Image uses flow matching — no external scheduler needed."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Ovis-Image loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Ovis-Image."""
        return []

    def get_layer_manifest(self) -> Any:
        """Ovis-Image layer manifest with double + single stream blocks."""
        from app.engine.core.layer_manifest import (  # noqa: PLC0415
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                for i, block in enumerate(joint):
                    blocks.append(
                        BlockInfo(
                            name=f"transformer_blocks.{i}",
                            block_type="joint",
                            param_count=sum(p.numel() for p in block.parameters()),
                            depth_index=i,
                        )
                    )
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                offset = len(blocks)
                for i, block in enumerate(single):
                    blocks.append(
                        BlockInfo(
                            name=f"single_transformer_blocks.{i}",
                            block_type="single",
                            param_count=sum(p.numel() for p in block.parameters()),
                            depth_index=offset + i,
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
        """Encode captions replicating ``OvisImagePipeline`` prompt embeds.

        Mirrors ``_get_messages`` + ``_get_ovis_prompt_embeds``:
        1. Prefix each caption with the fixed system prompt and apply the
           Qwen chat template (``add_generation_prompt=True``,
           ``enable_thinking=False``).
        2. Tokenize with ``padding='max_length'``, ``truncation=True``,
           ``max_length = max_sequence_length + user_prompt_begin_id``,
           ``add_special_tokens=False``.
        3. Forward through Qwen3Model (no ``output_hidden_states``) and
           take ``last_hidden_state``.
        4. Zero the padding positions (× attention_mask) and slice off the
           first ``user_prompt_begin_id`` template tokens.

        Returns:
            ``TextEncoderOutput`` with fixed-length embeddings
            ``[B, max_sequence_length, 2048]`` and the identically sliced
            ``attention_mask`` ``[B, max_sequence_length]``.
        """
        # 1. System prompt + chat template
        messages: list[str] = []
        for cap in captions:
            message = [{"role": "user", "content": OVIS_SYSTEM_PROMPT + cap}]
            templated = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            messages.append(templated)

        # 2. Tokenize (max_length includes the 28-token template prefix)
        tokens = self.tokenizer(
            messages,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length + self.user_prompt_begin_id,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        # 3. Forward — last_hidden_state (NOT a tapped hidden layer)
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        prompt_embeds = outputs.last_hidden_state

        # 4. Zero-mask padding, slice off the template prefix
        prompt_embeds = prompt_embeds * attention_mask[..., None]
        prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id :, :]
        sliced_mask = attention_mask[:, self.user_prompt_begin_id :]

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=sliced_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """OvisImageTransformer2DModel forward with pack / unpack.

        Tensor flow:
        1. Unpack ``text_embeddings`` → ``(enc_hs [B,L,2048], mask)``. The
           transformer takes no encoder attention mask — padding is already
           zeroed in the embeddings (pipeline semantics).
        2. Pack ``noisy_input [B,C,H,W]`` 2×2 → ``[B, (H/2)(W/2), C*4]``
           (identical to ``OvisImagePipeline._pack_latents``) + img_ids.
        3. Build txt_ids ``[L, 3]``: columns 1 AND 2 carry ``arange(L)``
           (Ovis convention — NOT Flux's all-zeros).
        4. Scale timesteps ``/ 1000.0`` exactly once (transformer multiplies
           by 1000 internally).
        5. Forward, then unpack the prediction back to ``[B, C, H, W]``.

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]``.
            timesteps: Flow-matching timesteps in ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings, attention_mask)`` tuple or a
                plain embeddings tensor.
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 16, H, W]``.
        """
        from app.engine.models.families.flux1.utils import (  # noqa: PLC0415
            pack_latents,
            unpack_latents,
        )

        # 1. Unpack text embeddings (mask is not consumed by the transformer).
        if isinstance(text_embeddings, tuple):
            enc_hs, _enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings

        B, C, H, W = noisy_input.shape

        # 2. Pack latents 2×2 → tokens + Flux-grid img_ids.
        packed, img_ids = pack_latents(noisy_input)
        img_ids = img_ids.to(device=noisy_input.device, dtype=enc_hs.dtype)

        # 3. Ovis txt_ids: cols 1 and 2 = arange(text_seq_len), col 0 = 0.
        text_seq_len = enc_hs.shape[1]
        txt_ids = torch.zeros(
            text_seq_len,
            3,
            device=noisy_input.device,
            dtype=enc_hs.dtype,
        )
        positions = torch.arange(
            text_seq_len,
            device=noisy_input.device,
            dtype=enc_hs.dtype,
        )
        txt_ids[:, 1] = positions
        txt_ids[:, 2] = positions

        # 4. Scale timesteps exactly once: [0, 1000] → [0, 1].
        model_timesteps = timesteps / 1000.0

        # 5. Transformer forward.
        model = self.get_primary_model()
        output = model(
            hidden_states=packed,
            encoder_hidden_states=enc_hs,
            timestep=model_timesteps,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # Unpack tokens back to spatial latents.
        return unpack_latents(pred, H, W)

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
        """Return the Ovis-Image ai-toolkit-format LoRA saver."""
        from app.engine.models.families.ovis_image.saver import (  # noqa: PLC0415
            OvisImageSaver,
        )

        return OvisImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Ovis-Image block topology: double (joint) + single stream blocks."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                topology.append(
                    {
                        "name": "double_blocks",
                        "attr_path": "transformer_blocks",
                        "count": len(joint),
                        "approx_vram_mb": 800,
                    }
                )
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                topology.append(
                    {
                        "name": "single_blocks",
                        "attr_path": "single_transformer_blocks",
                        "count": len(single),
                        "approx_vram_mb": 350,
                    }
                )
        return topology
