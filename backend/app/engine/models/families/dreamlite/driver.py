"""DreamLiteDriver — family-specific training behavior for DreamLite.

Implements ``IModelDriver`` for DreamLite (Base / Mobile).

DreamLite specifics:
- Primary model: ``DreamLiteUNetModel`` — a **U-Net** (extends
  ``UNet2DConditionModel``), NOT a DiT. No patchify/packing; the forward
  is the inherited UNet call with ``encoder_attention_mask`` and
  ``added_cond_kwargs={"time_ids": [[w_px, h_px]]}``.
- Generate-mode input: the pipeline concatenates ZERO image-conditioning
  latents along the WIDTH axis (``cat([latents, zeros], dim=3)``) and
  slices the prediction back to the latent width (``[..., :W]``). The
  driver mirrors this exactly.
- Timesteps: RAW ``[0, 1000]`` — the UNet's sinusoidal ``time_proj``
  consumes raw timesteps (``t.expand(B).to(latents.dtype)`` in the
  pipeline). NEVER divide by 1000 (unlike the DiT families).
- Text encoder: Qwen3-VL; ``encode_text`` replicates
  ``DreamLitePipeline.encode_prompt`` (generate mode): pinned chat
  template, tokenize with ``max_length = max_sequence_length + drop_idx``,
  ``hidden_states[-1]``, per-sequence mask-select, drop the 34-token
  template prefix, zero re-pad + fresh 0/1 mask. The ``"[Generate]: "``
  prompt prefix is applied by the CALLER (trainer/sampler) — exactly as
  the pipeline's ``__call__`` (not ``encode_prompt``) applies it, so the
  CFG negative prompt stays un-prefixed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# Pinned chat template — copied verbatim from ``DreamLitePipeline.__init__``
# (``prompt_template_encode_generate``, diffusers 0.39).
DREAMLITE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects "
    "and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

# ``prompt_template_encode_generate_start_idx`` — number of chat-template
# prefix tokens dropped from the encoder hidden states.
_DEFAULT_DROP_IDX = 34

# ``DreamLitePipeline.__call__`` default ``max_sequence_length``.
_DEFAULT_MAX_SEQUENCE_LENGTH = 200

# The pipeline's generate-mode prompt prefix (applied by __call__, NOT by
# encode_prompt — negatives stay un-prefixed). Exposed for trainer/sampler.
GENERATE_PREFIX = "[Generate]: "


def te_template_fingerprint() -> str:
    """Fingerprint of every constant that transforms a caption before encoding.

    Covers the pinned chat template (``DREAMLITE_PROMPT_TEMPLATE``) and the
    default template-prefix drop index / max sequence length (the values
    ``self.drop_idx`` / ``self.max_sequence_length`` fall back to when a
    definition does not override ``te.drop_idx`` / ``te.max_sequence_length``).
    Computed FRESH on every call (reads the current module globals rather
    than freezing a value at import time) so a future edit — or a test
    monkeypatching one of these constants — is picked up immediately.

    Used by ``DreamLiteTrainer`` to version its disk-cache key
    (``_disk_cache_key`` in trainer.py — the qwen_image/boogu_image
    precedent) so a template or drop-idx change can never silently reuse
    embeddings encoded under the OLD template (the poisoned-cache incident
    class this closes: only the ``"[Generate]: "`` prefix previously reached
    the hashed key, not the chat template or drop-idx it wraps).
    """
    src = "|".join(
        [
            DREAMLITE_PROMPT_TEMPLATE,
            str(_DEFAULT_DROP_IDX),
            str(_DEFAULT_MAX_SEQUENCE_LENGTH),
        ]
    )
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


class DreamLiteDriver(IModelDriver):
    """DreamLite family driver.

    Handles:
    - Qwen3-VL template-drop prompt encoding (pipeline-identical)
    - U-Net width-concat flow-matching forward pass (raw timesteps)
    - LoRA targets for the DreamLite attention/ff projections
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
        self.drop_idx = int(arch.get("te.drop_idx", _DEFAULT_DROP_IDX))
        self.vae_scale_factor = int(arch.get("vae.vae_scale_factor", 8))

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded DreamLite components into driver state."""
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

    def release_text_encoders(self) -> None:
        """Null the attr get_text_encoders() reads."""
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """DreamLite LoRA targets — attention + feed-forward projections.

        The UNet's cross-attention (attn2) exists in every attention-bearing
        block; self-attention (attn1) only in down_blocks.2, mid_block, and
        up_blocks.0 (the non-"NoSelfAttn" blocks). On the real checkpoint
        config these patterns match exactly 312 Linear modules → 624 LoRA
        keys (pinned by test_dreamlite_lora_portability.py). With MQA
        (num_kv_heads=1, head_dim 64) every to_k/to_v is 64-wide — the
        LoRA B matrices there are narrow; PEFT handles this automatically.
        """
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
            # Self-attention (down_blocks.2 / mid_block / up_blocks.0 only)
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            # Cross-attention (every attention-bearing block)
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            # Feed-forward (GEGLU proj + output proj)
            "ff.net.0.proj",
            "ff.net.2",
        ]

    def init_scheduler(self) -> Any:
        """DreamLite uses flow matching — no external scheduler needed."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """DreamLite loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for DreamLite."""
        return []

    def get_layer_manifest(self) -> Any:
        """DreamLite layer manifest with down/mid/up U-Net blocks (SDXL style)."""
        from app.engine.core.layer_manifest import (  # noqa: PLC0415
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            depth = 0
            down = getattr(model, "down_blocks", None)
            if down is not None:
                for i, block in enumerate(down):
                    blocks.append(BlockInfo(
                        name=f"down_blocks.{i}",
                        block_type="down",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=depth,
                    ))
                    depth += 1
            mid = getattr(model, "mid_block", None)
            if mid is not None:
                blocks.append(BlockInfo(
                    name="mid_block",
                    block_type="mid",
                    param_count=sum(p.numel() for p in mid.parameters()),
                    depth_index=depth,
                ))
                depth += 1
            up = getattr(model, "up_blocks", None)
            if up is not None:
                for i, block in enumerate(up):
                    blocks.append(BlockInfo(
                        name=f"up_blocks.{i}",
                        block_type="up",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=depth,
                    ))
                    depth += 1

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
        """Encode captions replicating ``DreamLitePipeline.encode_prompt``.

        Generate mode, step by step (pipeline_dreamlite.py:272-312):
        1. insert each caption into the pinned chat template;
        2. tokenize with ``max_length = max_sequence_length + drop_idx``,
           padding + truncation;
        3. TE forward with ``output_hidden_states=True`` →
           ``hidden_states[-1]``;
        4. ``_extract_masked_hidden`` per-sequence mask-select, drop the
           first ``drop_idx`` (template prefix) tokens, re-pad with zeros,
           build a fresh 0/1 mask.

        Cacheability deviation (mask-equivalent): sequences are re-padded to
        the FIXED ``max_sequence_length`` instead of the batch max — padded
        positions carry zero embeddings and mask 0, the pipeline's exact
        padding convention, so cross-attention sees identical conditioning.

        NOTE: the ``"[Generate]: "`` prefix is NOT applied here — the
        caller adds it (mirrors ``__call__`` vs ``encode_prompt``), so CFG
        negative prompts stay un-prefixed.

        Returns:
            ``TextEncoderOutput`` with embeddings
            ``[B, max_sequence_length, hidden]`` and attention mask
            ``[B, max_sequence_length]``.
        """
        txts = [DREAMLITE_PROMPT_TEMPLATE.format(p) for p in captions]
        tokens = self.tokenizer(
            text=txts,
            max_length=self.max_sequence_length + self.drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        # Per-sequence mask-select + template-prefix drop (pipeline's
        # _extract_masked_hidden + ``[drop_idx:]`` slice).
        bool_mask = tokens.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1).tolist()
        split_hidden = torch.split(
            hidden_states[bool_mask], valid_lengths, dim=0,
        )
        split_hidden = [e[self.drop_idx:] for e in split_hidden]

        # Zero re-pad to the fixed max_sequence_length + fresh 0/1 mask.
        B = len(split_hidden)
        L = self.max_sequence_length
        D = hidden_states.shape[-1]
        prompt_embeds = torch.zeros(
            (B, L, D), dtype=hidden_states.dtype, device=self.device,
        )
        prompt_mask = torch.zeros((B, L), dtype=torch.long, device=self.device)
        for i, seq in enumerate(split_hidden):
            n = min(seq.shape[0], L)
            prompt_embeds[i, :n] = seq[:n]
            prompt_mask[i, :n] = 1

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=prompt_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """DreamLiteUNetModel forward — pipeline-identical UNet invocation.

        Tensor flow (mirrors ``DreamLitePipeline.__call__`` generate mode):
        1. Unpack ``text_embeddings`` → ``(enc_hs [B,L,D], mask [B,L])``.
        2. Width-concat ZERO image-conditioning latents:
           ``model_input = cat([noisy, zeros_like(noisy)], dim=3)``.
        3. ``added_cond_kwargs = {"time_ids": [[w_px, h_px]] * B}`` with
           pixel dims = latent dims × vae_scale_factor.
        4. Timesteps RAW ``[0, 1000]`` cast to the latent dtype
           (``t.expand(B).to(latents.dtype)``) — NO division; the UNet's
           sinusoidal ``time_proj`` consumes raw timesteps.
        5. Slice the prediction back to the latent width (``[..., :W]``).

        Args:
            noisy_input: Noisy latents ``[B, 4, H, W]``.
            timesteps: Flow-matching timesteps on the ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings, attention_mask)`` tuple or a
                plain embeddings tensor.
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 4, H, W]``.
        """
        # 1. Unpack text embeddings.
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings
            enc_mask = None

        B, _C, _H, W = noisy_input.shape
        model = self.get_primary_model()

        # 2. Generate-mode width concat (zero image-conditioning half).
        model_input = torch.cat(
            [noisy_input, torch.zeros_like(noisy_input)], dim=3,
        )

        # 3. time_ids carry the ORIGINAL pixel (width, height).
        width_px = W * self.vae_scale_factor
        height_px = noisy_input.shape[2] * self.vae_scale_factor
        time_ids = torch.tensor(
            [[float(width_px), float(height_px)]],
            device=noisy_input.device,
            dtype=noisy_input.dtype,
        ).expand(B, -1)

        # 4. RAW timesteps in the latent dtype (pipeline: t.expand(B).to(dtype)).
        model_timesteps = timesteps.to(dtype=noisy_input.dtype)

        output = model(
            model_input,
            timestep=model_timesteps,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # 5. Drop the image-conditioning half of the spatial concat.
        return pred[..., :W]

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
        """Return the DreamLite ai-toolkit-format LoRA saver."""
        from app.engine.models.families.dreamlite.saver import (  # noqa: PLC0415
            DreamLiteSaver,
        )

        return DreamLiteSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """DreamLite U-Net block topology: down + mid + up (SDXL style)."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            down = getattr(model, "down_blocks", None)
            if down is not None:
                topology.append({
                    "name": "down_blocks",
                    "attr_path": "down_blocks",
                    "count": len(down),
                    "approx_vram_mb": 100,
                })
            mid = getattr(model, "mid_block", None)
            if mid is not None:
                topology.append({
                    "name": "mid_block",
                    "attr_path": "mid_block",
                    "count": 1,
                    "approx_vram_mb": 100,
                })
            up = getattr(model, "up_blocks", None)
            if up is not None:
                topology.append({
                    "name": "up_blocks",
                    "attr_path": "up_blocks",
                    "count": len(up),
                    "approx_vram_mb": 100,
                })
        return topology
