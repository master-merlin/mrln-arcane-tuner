"""Shared WAN driver base — flow-match + WAN transformer forward.

:class:`WanDriverBase` implements the family-agnostic WAN training behavior that
both WAN 2.1 and WAN 2.2 reuse:

- **Flow-match ``add_noise``** in the ``[0, 1000]`` timestep space::

      noisy = (t / 1000) * noise + (1 - t / 1000) * latents

  with the implied velocity target ``noise - latents`` (handled by the generic
  ``compute_target`` in the base interface, same space). The ``/1000`` belongs
  to the LERP ONLY (it needs ``sigma ∈ [0,1]``); the timestep handed to the
  transformer stays RAW ``[0, 1000]`` (see below). This is THE pure-noise gotcha:
  the diffusers WAN time embedder is a sinusoidal ``Timesteps`` over the raw
  FlowMatchEuler value — feeding it ``[0,1]`` makes the frozen (non-LoRA'd)
  embedder read every step as ``t≈0`` (clean) → ≈0 velocity → noise samples.

- **``forward_pass``** calling the WAN transformer with
  ``hidden_states=noisy[B,C,F,H,W]``, ``timestep=t`` (RAW ``[0, 1000]`` — the
  scale the diffusers ``WanPipeline`` feeds, NOT ``t/1000``),
  ``encoder_hidden_states=text``, and (I2V only) ``encoder_hidden_states_image``
  + a 36-channel concatenated input built from the batch's first-frame latent.

- **``get_lora_targets``** fallback (definition enrichment overrides at runtime).

Weight loading stays lazy: nothing here imports model weights. The transformer
is whatever ``assign_components`` is handed (a fake in tests).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.wan_shared.i2v_conditioning import (
    build_i2v_conditioning,
    build_still_t2v_input,
)
from app.engine.models.families.wan_shared.text_encoding import (
    WAN_TE_MAX_LENGTH,
    encode_umt5,
)

logger = structlog.get_logger(__name__)


# Self-attention (attn1) + cross-attention (attn2) + feed-forward LoRA targets
# for ``WanTransformerBlock``. PEFT matches these as ``key.endswith(target)``.
WAN_T2V_LORA_TARGETS: list[str] = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]

# I2V adds the image cross-attention key/value projections.
WAN_I2V_EXTRA_LORA_TARGETS: list[str] = [
    "attn2.add_k_proj",
    "attn2.add_v_proj",
]


class WanDriverBase(IModelDriver):
    """Family-agnostic WAN driver (shared by WAN 2.1 + WAN 2.2).

    Concrete families subclass this and set ``self.is_i2v`` (via the
    definition's ``mode``) so the LoRA targets and forward path pick up the
    image-conditioning channels.
    """

    # Key in ``batch`` (a build_batch_extra entry) holding the I2V first-frame
    # latent tensor ``[B, 16, 1, H, W]`` and the CLIP image embedding.
    BATCH_FIRST_FRAME_LATENT = "wan_first_frame_latent"
    BATCH_IMAGE_EMBED = "wan_image_embed"

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.image_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self.image_processor: Any = None
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.te_max_length: int = int(arch.get("te.max_length", WAN_TE_MAX_LENGTH))
        self.mode: str = str(arch.get("mode", "t2v")).lower()
        self.is_i2v: bool = self.mode == "i2v"

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded WAN components."""
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.image_encoder = components.get("image_encoder")
        self.image_processor = components.get("image_processor")

        self.logger.info(
            "wan_config",
            mode=self.mode,
            is_i2v=self.is_i2v,
            te_max_length=self.te_max_length,
            has_image_encoder=self.image_encoder is not None,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """WAN LoRA targets — definition enrichment overrides at runtime."""
        definition_targets = getattr(self.definition, "lora_targetable_modules", None)
        if definition_targets:
            self.logger.info(
                "lora_targets_from_definition", count=len(definition_targets)
            )
            return list(definition_targets)

        targets = list(WAN_T2V_LORA_TARGETS)
        if self.is_i2v:
            targets += list(WAN_I2V_EXTRA_LORA_TARGETS)
        self.logger.info("lora_targets_pattern_defaults", count=len(targets))
        return targets

    def init_scheduler(self) -> Any:
        """WAN uses flow matching — no external training scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """WAN loads the transformer + TE in bf16 (VAE stays fp32 via loader)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for WAN."""
        return []

    # --- Phase 2: Text Encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> TextEncoderOutput:
        """Encode captions through UMT5-XXL (shared WAN logic)."""
        return encode_umt5(
            self.text_encoder,
            self.tokenizer,
            captions,
            self.device,
            dtype,
            max_length=self.te_max_length,
        )

    # --- Phase 5: Training Loop Hooks ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Lift a 4D still latent to 5D so the WAN transformer (which unpacks
        ``b,c,f,h,w = hidden_states.shape``) accepts a single still as a 1-frame
        clip. A still in a mixed stills+video run is cached 4D ``[B,C,H,W]`` via
        the image path; without this lift the RoPE unpack raised
        'not enough values to unpack (expected 5, got 4)'. Mirrors LTX-2's
        ``_pack_latents`` 4D handling. 5D input is returned unchanged.
        """
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
        return latents

    def attach_conditioning(self, batch: dict[str, Any], latents: torch.Tensor) -> None:
        """I2V: stash the clean first-frame latent (the clip's own frame 0).

        T2V is a no-op. The first-frame latent is ``[B, 16, 1, H, W]`` — the
        forward's ``build_i2v_conditioning`` zero-pads it to F and concatenates
        ``[noisy(16), mask(4), cond(16)]``. (CLIP image embed stays None — the
        diffusers WAN transformer guards None; full-fidelity CLIP conditioning is
        a documented follow-up.)

        F=1 STILL GUARD: a single still on an i2v run trains as t2v — there is
        no frame to predict beyond a conditioning frame, so stashing (and later
        leaking, via the ``cond`` channels) the still's own clean latent as the
        answer would be a degenerate, zero-information step. Skip the stash;
        :meth:`forward_pass` takes the zeroed-conditioning t2v path when F=1.
        (ltx2/k5 parity — their ``_i2v_conditioning_engaged`` gates on F>1.)
        """
        if not self.is_i2v:
            return
        if self.BATCH_FIRST_FRAME_LATENT in batch:
            return
        lat = latents if latents.ndim == 5 else latents.unsqueeze(2)
        if lat.shape[2] <= 1:
            return
        batch[self.BATCH_FIRST_FRAME_LATENT] = lat[:, :, :1, :, :].detach().clone()

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-match lerp in the ``[0, 1000]`` space.

        ``noisy = (t/1000) * noise + (1 - t/1000) * latents``.

        The implied flow-matching velocity target is ``noise - latents`` (the
        base ``compute_target`` default), in the SAME 16-channel space. This is
        the contract :func:`assert_flowmatch_timestep_contract` pins.
        """
        t = timesteps / 1000.0
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * noise + (1.0 - t) * latents

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """WAN transformer forward — predicts velocity over the 16 noise channels.

        For T2V the transformer consumes the 16-channel noised latent directly.
        For I2V the 36-channel ``[noisy(16), mask(4), cond(16)]`` input is built
        HERE from the batch's first-frame latent (``BATCH_FIRST_FRAME_LATENT``),
        and the CLIP image embedding (``BATCH_IMAGE_EMBED``) is passed as
        ``encoder_hidden_states_image``. The model output is 16 channels, so the
        velocity target (computed by the trainer over the 16-channel
        ``noisy_input``) and the prediction live in the same space.

        Args:
            noisy_input: 16-channel noised latent ``[B, 16, F, H, W]``.
            timesteps: Scaled timesteps in ``[0, 1000]``.
            text_embeddings: ``TextEncoderOutput`` or raw ``[B, L, D]`` tensor.
            batch: Full batch dict (I2V extras live here).

        Returns:
            Velocity prediction ``[B, 16, F, H, W]``.
        """
        enc_hs = self._as_text_tensor(text_embeddings)

        image_embed = None
        hidden_states = noisy_input
        if self.is_i2v:
            if noisy_input.shape[2] > 1:
                first_frame = batch.get(self.BATCH_FIRST_FRAME_LATENT)
                if first_frame is None:
                    raise ValueError(
                        "I2V forward_pass requires batch["
                        f"'{self.BATCH_FIRST_FRAME_LATENT}'] (first-frame latent)."
                    )
                hidden_states = build_i2v_conditioning(noisy_input, first_frame)
                image_embed = batch.get(self.BATCH_IMAGE_EMBED)
            else:
                # F=1 STILL on an i2v run → train as t2v. The 36-in-channel
                # patch_embedding still needs 36 channels, so pad mask+cond with
                # ZEROS (no conditioning frame, no answer leak). image_embed
                # stays None. See attach_conditioning / build_still_t2v_input;
                # ltx2/k5 parity (their F>1 _i2v_conditioning_engaged gate).
                hidden_states = build_still_t2v_input(noisy_input)

        # RAW [0, 1000] timestep — the diffusers WAN time embedder consumes the
        # FlowMatchEuler value directly (sinusoidal, no internal /1000). The
        # /1000 lives in add_noise's LERP only; dividing here too made the frozen
        # time embedder read every step as t≈0 → pure-noise samples.
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_image=image_embed,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _as_text_tensor(text_embeddings: Any) -> torch.Tensor:
        """Unwrap a ``TextEncoderOutput`` / tuple to a raw ``[B, L, D]`` tensor."""
        if isinstance(text_embeddings, TextEncoderOutput):
            return text_embeddings.embeddings
        if isinstance(text_embeddings, tuple):
            return text_embeddings[0]
        return text_embeddings

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:  # pragma: no cover - overridden by families
        raise NotImplementedError("WAN families must provide get_saver().")
