"""Ideogram 4 model driver -- family-specific training behavior.

Implements :class:`IModelDriver` for the Ideogram 4 diffusion family
(vendored DiT + Qwen3-VL text encoder + FLUX-style VAE).

Key family characteristics:

*  **Multi-layer Qwen3-VL conditioning**: the stock Qwen3-VL language model is
   run with ``output_hidden_states=True`` and the hidden states at
   :data:`~app.engine.models.families.ideogram4.utils.QWEN3VL_SELECTED_LAYERS`
   (13 multi-scale slices: post-layer indices ``(0,3,...,35)`` -- the real
   upstream ``QWEN3_VL_ACTIVATION_LAYERS``) are concatenated on the feature
   dimension.  These are POST-LAYER indices (the output of decoder layer ``k``);
   HF ``output_hidden_states`` prepends the embedding output at ``[0]``, so each
   tap reads HF ``hidden_states[k+1]`` (mirrors ``microsoft_lens
   lens_layers_to_hf_indices``).  The result is ``(B, L, 4096 * 13)`` raw hidden
   states; the DiT projects them internally, so the driver supplies them
   unprojected.

*  **Fused-QKV / SwiGLU LoRA targets** ``["qkv", "o", "w1", "w2", "w3"]``
   (confirmed against the vendored model in Task 1).

*  Sequence packing (text+image, position_ids/segment_ids/indicator) and the
   denoising ``forward_pass`` land in Task 5; they are stubbed here so the
   driver can instantiate.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ideogram4 import utils
from app.engine.models.families.ideogram4.vendor.modeling_ideogram4 import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)

logger = structlog.get_logger(__name__)

# Driver-level cap on tokenised sequence length; governs the driver's padding
# budget, not a shared utils constant.
DEFAULT_TE_MAX_LENGTH = 2048
DEFAULT_SELECTED_LAYERS = utils.QWEN3VL_SELECTED_LAYERS

# Image-grid position offset (upstream ideogram4.constants.IMAGE_POSITION_OFFSET):
# image (t,h,w) positions are offset by this so they never collide with text
# positions (which start at 0 and stay well below it). Inlined here as it is a
# packing-layout constant, not part of the vendored nn.Module graph.
IMAGE_POSITION_OFFSET = 65536


class IdeogramV4Driver(IModelDriver):
    """Ideogram 4 family driver (multi-layer Qwen3-VL text encoder)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.te_max_length = int(arch.get("te.max_length", DEFAULT_TE_MAX_LENGTH))
        sel = arch.get("te.selected_layers", DEFAULT_SELECTED_LAYERS)
        self.selected_layers = tuple(int(i) for i in sel)
        # QWEN3VL_SELECTED_LAYERS are upstream POST-LAYER indices (the output of
        # decoder layer k). HF output_hidden_states prepends the embedding output
        # at [0], so the post-layer-k activation is HF hidden_states[k+1] -- the
        # same +1 shift microsoft_lens.lens_layers_to_hf_indices applies.
        self.hf_layer_indices = tuple(k + 1 for k in self.selected_layers)
        # Stashed post-patchify latent grid (set by prepare_latents); consumed by
        # forward_pass when the batch doesn't carry latent_h/latent_w.
        self._latent_h: int = 0
        self._latent_w: int = 0

    # --- Phase 1: component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.logger.info(
            "ideogram4_config",
            te_max_length=self.te_max_length,
            selected_layers=self.selected_layers,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {"text_encoder": self.text_encoder} if self.text_encoder else {}

    def get_lora_targets(self) -> list[str]:
        defn_targets = getattr(self.definition, "lora_targetable_modules", None)
        if defn_targets:
            return defn_targets
        # Confirmed against the vendored DiT in Task 1: fused qkv + attn out
        # projection + SwiGLU feed-forward (w1/w2/w3).
        return ["qkv", "o", "w1", "w2", "w3"]

    def get_te_lora_targets(self) -> list[str]:
        return []  # Qwen3-VL text encoder is frozen

    def init_scheduler(self) -> Any:
        return None  # flow-matching; no train-time scheduler

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    # --- Phase 2: text encoding ---

    @torch.no_grad()
    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Chat-template -> Qwen3-VL -> concat 13 selected hidden states.

        Mirrors upstream ``Ideogram4Pipeline._encode_text``: the selected
        hidden states are stacked and reshaped so each text token's feature
        vector is the per-hidden-unit interleaving of all selected layers
        (``stack -> permute -> reshape``), yielding a final feature width of
        ``4096 * len(QWEN3VL_SELECTED_LAYERS)``.  The DiT projects this raw
        concatenation internally, so no projection/norm is applied here.

        Returns ``TextEncoderOutput`` with ``embeddings`` ``(B, L, H*n_layers)``
        and a bool ``attention_mask`` ``(B, L)``.
        """
        rendered = [utils.render_chat_prompt(c, self.tokenizer) for c in captions]
        encoded = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.te_max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        # Device-safety: follow the text encoder's parameters, not self.device.
        # Falls back to self.device for parameter-less encoders.
        te_device = self.device
        try:
            te_device = next(self.text_encoder.parameters()).device
        except StopIteration:
            pass
        input_ids = encoded["input_ids"].to(te_device)
        attn = encoded["attention_mask"].to(te_device)

        out = self.text_encoder(
            input_ids=input_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "text_encoder returned no hidden_states; ensure the model "
                "supports output_hidden_states=True"
            )

        # Each selected upstream post-layer index k maps to HF hidden_states[k+1]
        # (precomputed in self.hf_layer_indices); hidden_states[0] is the raw
        # embedding output, which upstream never taps. Guard against a TE that
        # didn't return enough layers.
        max_hf_index = max(self.hf_layer_indices)
        if len(hidden_states) <= max_hf_index:
            raise RuntimeError(
                f"text_encoder returned {len(hidden_states)} hidden states, but "
                f"the highest selected layer needs HF index {max_hf_index}; "
                "ensure the encoder exposes all decoder layers."
            )

        # Select the activation layers and concat on the feature dim, matching
        # upstream's (num_taps, B, L, H) -> (B, L, H, num_taps) -> (B, L, H*n)
        # interleaved layout.
        layers = [hidden_states[i].to(dtype=dtype) for i in self.hf_layer_indices]
        stacked = torch.stack(layers, dim=0)        # (n, B, L, H)
        stacked = stacked.permute(1, 2, 3, 0)       # (B, L, H, n)
        b, length = input_ids.shape
        embeddings = stacked.reshape(b, length, -1)  # (B, L, H*n)

        return TextEncoderOutput(embeddings=embeddings, attention_mask=attn.bool())

    # --- Phase 5: latent prep + sequence packing + forward pass ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latent ``[B,32,h,w]`` -> patchify -> latent-norm -> ``[B, S, 128]``.

        Stashes the post-patchify grid (``h//PATCH_FACTOR``, ``w//PATCH_FACTOR``)
        for use by :meth:`forward_pass` when the batch omits ``latent_h``/``latent_w``.
        """
        self._latent_h = latents.shape[-2] // utils.PATCH_FACTOR
        self._latent_w = latents.shape[-1] // utils.PATCH_FACTOR
        seq = utils.patchify_to_seq(latents)
        return utils.normalize_latents(seq).to(self.device)

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Raw noise ``[B,32,h,w]`` -> patchify -> ``[B, S, 128]`` (NO latent-norm).

        Mirrors microsoft_lens: noise stays un-normalized; only the image latents
        carry the per-channel latent-norm shift/scale.
        """
        return utils.patchify_to_seq(noise).to(self.device)

    # Trainer/scheduler timestep convention. ``sample_timesteps`` and the
    # noise-interpolation lerp use the ``[0, num_train_timesteps]`` (i.e.
    # ``[0, 1000]``) convention. The Ideogram4 DiT's ``Ideogram4EmbedScalar`` is
    # built with ``input_range=(0.0, 1.0)`` and self-scales by 1e4 internally
    # (Task 1 contract, modeling_ideogram4.py §3), so it wants the flow-matching
    # value in ``[0, 1]``. We therefore DIVIDE by NUM_TRAIN_TIMESTEPS here.
    # NEVER multiply by 1000 — passing [0,1000] feeds the embedder up to 1e7 and
    # corrupts the conditioning (this is the prior-failure ×1000 bug guard).
    NUM_TRAIN_TIMESTEPS = 1000.0

    def _build_packed_inputs(
        self,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        image_seq: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> dict[str, torch.Tensor]:
        """Pack (text, image-latent) into the single-stream DiT layout.

        Replicates upstream ``Ideogram4Pipeline._build_inputs`` +
        ``__call__``'s packed-x assembly (pipeline_ideogram4.py lines 344-412 and
        579-600). Per-sample layout is ``[text tokens][image tokens]`` over a
        unified length ``L = S_text + S_img``:

        * ``position_ids (B, L, 3)``: each text token ``p`` gets ``(p, p, p)``;
          each image token gets ``(0, h, w)`` from the latent grid, offset by
          ``IMAGE_POSITION_OFFSET`` so image positions never collide with text
          (upstream lines 366-371, 390-394).
        * ``indicator (B, L)``: text positions = ``LLM_TOKEN_INDICATOR`` (3),
          image positions = ``OUTPUT_IMAGE_INDICATOR`` (2) (upstream 396-397).
          Padded text positions (``text_mask`` False) are left at 0 so the DiT
          zeroes them, matching upstream left-padding (indicator stays 0 there).
        * ``segment_ids (B, L)``: a single sample id (1) over its real
          (text+image) span; non-real text (padding) stays at
          ``SEQUENCE_PADDING_INDICATOR`` (-1) so the block-diagonal attention
          mask excludes it (upstream 376-378, 399-400). All tokens here share one
          sample id (per-sample packing), giving full attention across the
          sample's text+image span.
        * ``x (B, L, 128)``: image latents at image positions; text positions are
          zero-filled (upstream ``text_z_padding`` / ``pos_z = cat([pad, z])``,
          lines 579-585, 592). The DiT also re-zeroes non-image x via indicator.
        * ``llm_features (B, L, D)``: text features at text positions; image
          positions are zero-filled. The DiT re-zeroes non-text via indicator.

        Unlike upstream inference (which LEFT-pads each sample to a shared
        ``max_text_tokens``), training packs a single right-padded batch: text is
        placed text-first and ``text_mask`` marks the real tokens. The
        text-then-image ordering, position scheme, indicator/segment values and
        x/llm_features zero-fill all match upstream by construction.
        """
        b, s_text, feat_dim = text_feats.shape
        s_img = image_seq.shape[1]
        device = image_seq.device
        seq_len = s_text + s_img

        # Image (t=0, h, w) grid positions, offset to stay disjoint from text.
        h_idx = (
            torch.arange(latent_h, device=device)
            .view(-1, 1).expand(latent_h, latent_w).reshape(-1)
        )
        w_idx = (
            torch.arange(latent_w, device=device)
            .view(1, -1).expand(latent_h, latent_w).reshape(-1)
        )
        t_idx = torch.zeros_like(h_idx)
        image_pos = (
            torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
        )  # (s_img, 3)
        if image_pos.shape[0] != s_img:
            raise ValueError(
                f"latent grid {latent_h}x{latent_w}={image_pos.shape[0]} does not "
                f"match image sequence length {s_img}."
            )

        text_mask = text_mask.to(device=device, dtype=torch.bool)

        position_ids = torch.zeros(b, seq_len, 3, dtype=torch.long, device=device)
        indicator = torch.zeros(b, seq_len, dtype=torch.long, device=device)
        segment_ids = torch.full(
            (b, seq_len), -1, dtype=torch.long, device=device,
        )  # SEQUENCE_PADDING_INDICATOR

        # Text positions (p, p, p) for p in range(s_text).
        text_pos = torch.arange(s_text, device=device)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)  # (s_text, 3)
        position_ids[:, :s_text] = text_pos_3d.unsqueeze(0)
        position_ids[:, s_text:] = image_pos.unsqueeze(0)

        # Indicator: real text -> 3, image -> 2, padded text -> 0.
        indicator[:, :s_text] = torch.where(
            text_mask,
            torch.full_like(indicator[:, :s_text], LLM_TOKEN_INDICATOR),
            torch.zeros_like(indicator[:, :s_text]),
        )
        indicator[:, s_text:] = OUTPUT_IMAGE_INDICATOR

        # Segment ids: one sample id (1) over the real (text+image) span; padded
        # text positions stay at -1 so they are masked out of attention.
        segment_ids[:, s_text:] = 1
        segment_ids[:, :s_text] = torch.where(
            text_mask,
            torch.ones_like(segment_ids[:, :s_text]),
            torch.full_like(segment_ids[:, :s_text], -1),
        )

        # x: zeros at text positions, image latents at image positions.
        x = torch.zeros(b, seq_len, image_seq.shape[-1], device=device, dtype=image_seq.dtype)
        x[:, s_text:] = image_seq

        # llm_features: text feats at text positions, zeros at image positions.
        llm_features = torch.zeros(
            b, seq_len, feat_dim, device=device, dtype=text_feats.dtype,
        )
        llm_features[:, :s_text] = text_feats * text_mask.unsqueeze(-1).to(text_feats.dtype)

        return {
            "llm_features": llm_features,
            "x": x,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
            "s_text": s_text,  # type: ignore[dict-item]
            "s_img": s_img,  # type: ignore[dict-item]
        }

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Pack text+image, run the single-stream DiT, return image-position velocity.

        ``noisy_input`` is the image latent sequence ``[B, S_img, 128]``;
        ``timesteps`` arrive in the shared trainer ``[0, 1000]`` convention and
        are divided by :attr:`NUM_TRAIN_TIMESTEPS` to the ``[0, 1]`` the DiT wants
        (the embedder self-scales by 1e4 — see the class-attr comment). Returns
        ONLY the image-position outputs reshaped to ``[B, S_img, 128]`` so the
        generic trainer's velocity loss lines up with the noised image latents
        (upstream slices image outputs the same way: ``out[:, max_text_tokens:]``,
        pipeline_ideogram4.py line 601).
        """
        # Unpack text features + mask (mirror microsoft_lens unpacking).
        if isinstance(text_embeddings, tuple):
            text_feats, text_mask = text_embeddings
        else:
            text_feats = text_embeddings
            text_mask = torch.ones(
                text_feats.shape[0], text_feats.shape[1],
                dtype=torch.bool, device=text_feats.device,
            )
        text_feats = text_feats.to(noisy_input.device)
        text_mask = text_mask.to(noisy_input.device)

        latent_h = int(batch.get("latent_h", 0)) or self._latent_h
        latent_w = int(batch.get("latent_w", 0)) or self._latent_w
        if latent_h <= 0 or latent_w <= 0:
            raise ValueError(
                "latent_h/latent_w unavailable; call prepare_latents first or set "
                f"them on the batch (got latent_h={latent_h}, latent_w={latent_w})."
            )

        packed = self._build_packed_inputs(
            text_feats, text_mask, noisy_input, latent_h, latent_w,
        )
        s_text = int(packed["s_text"])

        model_t = timesteps.to(noisy_input.device) / self.NUM_TRAIN_TIMESTEPS

        out = self.transformer(
            llm_features=packed["llm_features"],
            x=packed["x"],
            t=model_t,
            position_ids=packed["position_ids"],
            segment_ids=packed["segment_ids"],
            indicator=packed["indicator"],
        )
        # Slice the image-position outputs back to [B, S_img, 128] (text tokens
        # are first, image tokens follow — mirror upstream out[:, max_text:]).
        return out[:, s_text:]

    def get_saver(self):
        from .saver import IdeogramV4Saver
        return IdeogramV4Saver()
