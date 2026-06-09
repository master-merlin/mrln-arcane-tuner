"""Ideogram 4 model driver -- family-specific training behavior.

Implements :class:`IModelDriver` for the Ideogram 4 diffusion family
(vendored DiT + Qwen3-VL text encoder + FLUX-style VAE).

Key family characteristics:

*  **Multi-layer Qwen3-VL conditioning (manual layer capture)**: the stock
   Qwen3-VL language model is run with an EXPLICIT decoder-layer loop that
   reproduces ai-toolkit's ``get_qwen3_vl_features`` verbatim (embed_tokens +
   hand-built mRoPE ``position_embeddings`` + causal mask + per-layer forward),
   capturing the OUTPUT of each decoder layer in
   :data:`~app.engine.models.families.ideogram4.utils.QWEN3VL_SELECTED_LAYERS`
   (13 multi-scale slices: layer-output indices ``(0,3,...,35)`` -- the upstream
   ``QWEN3_VL_ACTIVATION_LAYERS``) and concatenating them on the feature
   dimension.  Index ``k`` maps DIRECTLY to the output of decoder layer ``k`` --
   there is NO ``+1`` HF-offset (that offset belonged to the old top-level
   ``AutoModel(output_hidden_states=True)`` path, now replaced).  The result is
   ``(B, L, 4096 * 13)`` raw hidden states; the DiT projects them internally, so
   the driver supplies them unprojected.

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
from transformers.masking_utils import create_causal_mask

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
        # QWEN3VL_SELECTED_LAYERS are decoder-layer-OUTPUT indices: index ``k``
        # == the output of decoder layer ``k``. The manual forward captures layer
        # outputs DIRECTLY at these indices (ai-toolkit parity), so there is no
        # HF ``+1`` offset.
        self.selected_layers = tuple(int(i) for i in sel)
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

    def _resolve_language_model(self) -> nn.Module:
        """Reach the ``Qwen3VLTextModel`` (embed_tokens / layers / rotary_emb).

        ``AutoModel.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")`` returns a
        ``Qwen3VLModel`` whose text tower is ``.language_model`` -- the SAME
        attribute ai-toolkit uses. We also accept ``.model.language_model``
        (the ``...ForConditionalGeneration`` wrapper) and a TE that already IS
        the text model (exposes ``layers``/``embed_tokens``), so the manual
        forward works regardless of which wrapper the loader produced.
        """
        te = self.text_encoder
        if hasattr(te, "language_model"):
            return te.language_model
        inner = getattr(te, "model", None)
        if inner is not None and hasattr(inner, "language_model"):
            return inner.language_model
        if hasattr(te, "layers") and hasattr(te, "embed_tokens"):
            return te
        raise RuntimeError(
            "Could not locate the Qwen3-VL text model (expected "
            "text_encoder.language_model with .embed_tokens/.layers/.rotary_emb); "
            f"got {type(te).__name__}."
        )

    @torch.no_grad()
    def _run_qwen3vl_layers(
        self,
        language_model: nn.Module,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Manual Qwen3-VL layer forward -> concat selected layer OUTPUTS.

        Faithful port of ai-toolkit ``src/pipeline.py::get_qwen3_vl_features``,
        adapted to this transformers' ``create_causal_mask`` signature (which
        wants ``input_embeds=`` + ``cache_position=``) exactly the way
        ``Qwen3VLTextModel.forward`` calls it. Steps:

        1. ``inputs_embeds = embed_tokens(token_ids)``.
        2. Build the 4-row mRoPE position ids from ``pos_2d`` (text-only: all
           rows equal). Row 0 is the text position ids fed to the layers/mask;
           rows 1-3 are the (t, h, w) mRoPE sections fed to ``rotary_emb``.
        3. ``create_causal_mask`` + ``rotary_emb`` -> shared causal mask +
           ``position_embeddings``.
        4. Explicit decoder-layer loop; capture ``hidden_states`` (the OUTPUT of
           the layer) at each ``QWEN3VL_SELECTED_LAYERS`` index DIRECTLY.
        5. ``stack -> permute(1,2,3,0) -> reshape(B, L, H*n)`` interleaved
           layout, then zero pad positions via ``attention_mask``.

        Returns ``(B, L, H * n_layers)`` in ``dtype``.
        """
        inputs_embeds = language_model.embed_tokens(token_ids)

        # 4-row mRoPE position ids (text-only -> all rows == pos_2d).
        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        # cache_position: contiguous query indices (this transformers version's
        # create_causal_mask requires it; mirrors Qwen3VLTextModel.forward).
        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device,
        )
        causal_mask = create_causal_mask(
            config=language_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(
            inputs_embeds, mrope_position_ids,
        )

        tap_set = set(self.selected_layers)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        missing = [k for k in self.selected_layers if k not in captured]
        if missing:
            raise RuntimeError(
                f"Qwen3-VL exposed {len(language_model.layers)} decoder layers, "
                f"but selected layer index/indices {missing} were not reached; "
                "ensure the text model has all decoder layers."
            )

        selected = [captured[k] for k in self.selected_layers]
        batch_size, seq_len = token_ids.shape
        stacked = torch.stack(selected, dim=0)        # (n, B, L, H)
        stacked = stacked.permute(1, 2, 3, 0)         # (B, L, H, n)
        stacked = stacked.reshape(batch_size, seq_len, -1).to(dtype)

        # Zero non-text (padding) positions, matching ai-toolkit.
        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        return stacked * text_mask

    @torch.no_grad()
    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Chat-template -> manual Qwen3-VL layer forward -> concat 13 outputs.

        Faithful port of ai-toolkit ``Ideogram4Model.get_prompt_embeds`` +
        ``get_qwen3_vl_features``:

        * The user message ``content`` is the typed list
          ``[{"type": "text", "text": caption}]`` and tokenization uses
          ``add_special_tokens=False`` (the chat template already emits the
          ``<|im_start|>`` specials; Qwen3-VL has no BOS).
        * The Qwen3-VL language model is run with an EXPLICIT decoder-layer loop
          (``_run_qwen3vl_layers``) capturing the OUTPUT of each selected layer
          DIRECTLY (no HF ``+1`` offset), then stacked/permuted/reshaped into the
          interleaved ``(B, L, H * n_layers)`` layout the frozen DiT expects.

        Returns ``TextEncoderOutput`` with ``embeddings`` ``(B, L, H*n_layers)``
        and a bool ``attention_mask`` ``(B, L)``.
        """
        rendered = [utils.render_chat_prompt(c, self.tokenizer) for c in captions]
        encoded = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.te_max_length, return_tensors="pt",
            add_special_tokens=False,
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

        # Text-only mRoPE position ids: cumulative real-token index (pad -> 0),
        # mirroring ai-toolkit's (attention_mask.cumsum - 1).clamp(min=0).
        pos_2d = (attn.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

        language_model = self._resolve_language_model()
        embeddings = self._run_qwen3vl_layers(
            language_model, input_ids, attn, pos_2d, dtype,
        )

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
