"""Kandinsky 5.0 model driver — family-specific training behavior.

Implements ``IModelDriver`` for the ``Kandinsky5Transformer3DModel`` video DiT
(diffusers 0.39, ``pipelines/kandinsky5``). The load-bearing quirks, all
replicated from the upstream pipelines:

Channels-LAST latents
~~~~~~~~~~~~~~~~~~~~~
The transformer consumes ``(B, F, H, W, C)`` — unlike every other family.
Our latent cache / VAE side stays channels-first ``[B, C, F, H, W]``;
:meth:`Kandinsky5Driver.prepare_latents` transposes at the boundary (and the
model's prediction comes back channels-last, so pred/target live in the same
space with no inverse transpose needed for the loss).

Dual text encoder + cu_seqlens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Qwen2.5-VL supplies the sequence embedding: chat-template prompt, LAST hidden
layer, sliced from ``crop_start=129`` — and instead of a padding mask the
transformer consumes ``cu_seqlens`` (int32 cumulative true lengths, left-padded
with 0) which only feeds ``text_rope_pos``. CLIP ViT-L supplies a 77-token
``pooler_output`` into ``pooled_projections`` (added to the time embedding).

Flow-match contract
~~~~~~~~~~~~~~~~~~~
Training runs on the raw ``[0, 1000]`` FlowMatchEuler scale::

    add_noise:  noisy = (t/1000) * noise + (1 - t/1000) * latents
    target:     v = noise - latents                (t-independent)
    forward:    transformer sees timestep = t      (RAW [0, 1000])

The ``/1000`` lives in ``add_noise``'s lerp ONLY — the Kandinsky time embedder
(``Kandinsky5TimeEmbeddings``) is sinusoidal over the raw value, exactly like
WAN (the pure-noise-LoRA gotcha).

visual_cond / I2V
~~~~~~~~~~~~~~~~~
BOTH shipped checkpoints have ``visual_cond=True``: the transformer input is
``cat([latents(C), visual_cond(C), mask(1)], dim=-1)``. Pure T2V feeds a ZERO
cond + mask. I2V puts the (clean) first-frame latent into frame 0 of BOTH the
latent stream and the cond stream with ``mask[:, 0:1] = 1``; frame 0 is never
noised and the loss excludes it (``Kandinsky5Trainer._compute_step_loss``).
"""

from __future__ import annotations

import html
import re
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput

logger = structlog.get_logger(__name__)

# Flow-match timestep scale (FlowMatchEulerDiscreteScheduler lives in [0, 1000]).
FLOWMATCH_SCALE = 1000.0

# Qwen2.5-VL prompt template — byte-identical to Kandinsky5T2VPipeline
# (pipeline_kandinsky.py:189-201). The first 129 tokens are the chat-template
# prefix, dropped from the encoder output (crop_start).
KANDINSKY5_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
KANDINSKY5_CROP_START = 129
KANDINSKY5_MAX_SEQUENCE_LENGTH = 512
KANDINSKY5_CLIP_MAX_LENGTH = 77

# Default negative prompt injected by BOTH upstream pipelines when
# guidance_scale > 1 and no negative prompt is provided.
KANDINSKY5_DEFAULT_NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst "
    "quality, low quality, ugly, deformed, walking backwards"
)

# LoRA target SUFFIXES for one Kandinsky5TransformerDecoderBlock (custom
# attention naming — NOT to_q/to_k/to_v). The driver expands these to
# fully-indexed ``visual_transformer_blocks.{i}.{suffix}`` paths: plain PEFT
# suffix matching would ALSO hit ``text_transformer_blocks`` (same
# self_attention/feed_forward sub-module names), which must stay frozen.
K5_LORA_TARGET_SUFFIXES: list[str] = [
    "self_attention.to_query",
    "self_attention.to_key",
    "self_attention.to_value",
    "self_attention.out_layer",
    "cross_attention.to_query",
    "cross_attention.to_key",
    "cross_attention.to_value",
    "cross_attention.out_layer",
    "feed_forward.in_layer",
    "feed_forward.out_layer",
]


# ── Module-level helpers (unit-tested; shared with the sampler) ─────────────


def prompt_clean(text: str) -> str:
    """Whitespace + HTML-entity normalization (pipeline ``prompt_clean``).

    ftfy fixing is applied when available (mirroring the upstream optional
    dependency); the double-unescape matches the pipeline exactly.
    """
    try:  # pragma: no cover - optional dependency parity with diffusers
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def build_cu_seqlens(lengths: torch.Tensor | list[int]) -> torch.Tensor:
    """Cumulative sequence lengths, int32, left-padded with 0.

    Replicates ``pipeline_kandinsky._encode_prompt_qwen``::

        cu_seqlens = F.pad(cumsum(mask[:, 129:].sum(1)), (1, 0)).to(int32)

    Args:
        lengths: Per-sample TRUE (unpadded) sequence lengths ``[B]``.

    Returns:
        ``[B + 1]`` int32 tensor ``[0, l0, l0+l1, ...]``.
    """
    t = torch.as_tensor(lengths)
    cu = torch.cumsum(t, dim=0)
    return F.pad(cu, (1, 0), value=0).to(dtype=torch.int32)


def get_scale_factor(height: int, width: int) -> tuple[float, float, float]:
    """RoPE scale factor by PIXEL resolution (pipeline ``_get_scale_factor``).

    Both dims within [480, 854] px → ``(1, 2, 2)``; else ``(1, 3.16, 3.16)``.
    """

    def between_480p(x: int) -> bool:
        return 480 <= x <= 854

    if between_480p(height) and between_480p(width):
        return (1, 2, 2)
    return (1, 3.16, 3.16)


def to_channels_last(latents: torch.Tensor) -> torch.Tensor:
    """Our 5D channels-first ``[B, C, F, H, W]`` → transformer ``[B, F, H, W, C]``."""
    if latents.ndim != 5:
        raise ValueError(
            f"to_channels_last expects a 5D [B, C, F, H, W] tensor, got "
            f"{tuple(latents.shape)}"
        )
    return latents.permute(0, 2, 3, 4, 1)


def to_channels_first(latents: torch.Tensor) -> torch.Tensor:
    """Transformer ``[B, F, H, W, C]`` → our 5D channels-first ``[B, C, F, H, W]``."""
    if latents.ndim != 5:
        raise ValueError(
            f"to_channels_first expects a 5D [B, F, H, W, C] tensor, got "
            f"{tuple(latents.shape)}"
        )
    return latents.permute(0, 4, 1, 2, 3)


class Kandinsky5Driver(IModelDriver):
    """Kandinsky 5.0 driver (T2V Lite / I2V Pro)."""

    # Key in ``batch`` holding the I2V first-frame latent (channels-FIRST
    # ``[B, C, 1, H, W]`` — the clip's own clean frame 0).
    BATCH_FIRST_FRAME_LATENT = "k5_first_frame_latent"

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None  # Qwen2.5-VL
        self.tokenizer: Any = None  # Qwen2VLProcessor (processor-as-tokenizer)
        self.text_encoder_2: nn.Module | None = None  # CLIP ViT-L text
        self.tokenizer_2: Any = None  # CLIPTokenizer
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.mode: str = str(arch.get("mode", "t2v")).lower()
        self.is_i2v: bool = self.mode == "i2v"
        # Both shipped checkpoints are visual_cond=True; the transformer config
        # (assign_components) re-confirms once loaded.
        self.visual_cond: bool = bool(arch.get("transformer.visual_cond", True))
        self.in_visual_dim: int = int(arch.get("transformer.in_visual_dim", 16))
        self.num_visual_blocks: int = int(
            arch.get("transformer.num_visual_blocks", 32)
        )
        self.max_sequence_length: int = int(
            arch.get("te.max_sequence_length", KANDINSKY5_MAX_SEQUENCE_LENGTH)
        )
        self.crop_start: int = int(arch.get("te.crop_start", KANDINSKY5_CROP_START))
        self.vae_spatial: int = int(arch.get("video.vae_spatial", 8))

        # Post-patch latent grid (F, H_lat, W_lat) recorded by prepare_latents.
        self._latent_shape: tuple[int, int, int] | None = None
        # Per-step i2v flag; set by Kandinsky5Trainer._attach_conditioning.
        self._i2v_active: bool = False

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Kandinsky 5.0 components."""
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.text_encoder_2 = components.get("text_encoder_2")
        self.tokenizer_2 = components.get("tokenizer_2")

        # The loaded transformer config is authoritative for the cond concat.
        cfg = getattr(self.transformer, "config", None)
        if cfg is not None:
            self.visual_cond = bool(getattr(cfg, "visual_cond", self.visual_cond))
            self.in_visual_dim = int(
                getattr(cfg, "in_visual_dim", self.in_visual_dim)
            )

        self.logger.info(
            "kandinsky5_config",
            mode=self.mode,
            is_i2v=self.is_i2v,
            visual_cond=self.visual_cond,
            in_visual_dim=self.in_visual_dim,
            num_visual_blocks=self.num_visual_blocks,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        if self.text_encoder_2 is not None:
            result["text_encoder_2"] = self.text_encoder_2
        return result

    def get_lora_targets(self) -> list[str]:
        """Fully-indexed visual-block LoRA targets.

        The Kandinsky5 attention uses custom projection names
        (``to_query``/``to_key``/``to_value``/``out_layer``) shared VERBATIM by
        the frozen ``text_transformer_blocks`` — plain suffix targets would
        bleed LoRA into the text stack. Expanding to full
        ``visual_transformer_blocks.{i}.{suffix}`` paths keys PEFT on exact
        module names, so the text blocks / modulations / embedders (incl. the
        time embedder's own ``in_layer``/``out_layer``) stay untouched.

        A definition's ``lora_targetable_modules`` overrides the SUFFIX set
        (still expanded per block), never the container.
        """
        suffixes = (
            list(getattr(self.definition, "lora_targetable_modules", None) or [])
            or list(K5_LORA_TARGET_SUFFIXES)
        )
        targets = [
            f"visual_transformer_blocks.{i}.{suffix}"
            for i in range(self.num_visual_blocks)
            for suffix in suffixes
        ]
        self.logger.info(
            "lora_targets_expanded",
            blocks=self.num_visual_blocks,
            suffixes=len(suffixes),
            total=len(targets),
        )
        return targets

    def init_scheduler(self) -> Any:
        """Flow matching — no external training scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Transformer + TEs load in bf16 (VAE stays fp32 via the loader)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Kandinsky 5.0."""
        return []

    # --- Phase 2: Text Encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> TextEncoderOutput:
        """Dual-encode captions → (Qwen sequence, CLIP pooled, cu_seqlens).

        Returns a :class:`TextEncoderOutput` where:

        - ``embeddings``: Qwen2.5-VL LAST hidden layer sliced from
          ``crop_start`` — ``[B, L, 3584]``, trimmed to the batch max TRUE
          length (``padding="longest"`` semantics, matching the pipeline).
        - ``pooled``: CLIP 77-token ``pooler_output`` — ``[B, 768]`` (the
          transformer's ``pooled_projections``).
        - ``attention_mask``: **cu_seqlens** int32 ``[B + 1]`` — the Kandinsky
          replacement for a padding mask (feeds ``text_rope_pos`` only).
        """
        prompts = [prompt_clean(p) for p in captions]
        with torch.no_grad():
            embeds, cu_seqlens = self._encode_qwen(prompts, dtype)
            pooled = self._encode_clip(prompts, dtype)
        return TextEncoderOutput(
            embeddings=embeds, attention_mask=cu_seqlens, pooled=pooled
        )

    def _encode_qwen(
        self, prompts: list[str], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Qwen2.5-VL encode (pipeline ``_encode_prompt_qwen``).

        Chat-template prefix + user prompt, tokenized to at most
        ``crop_start + max_sequence_length``; LAST hidden layer sliced from
        ``crop_start``; ``cu_seqlens`` built from the post-slice attention
        mask. (The pipeline's warn-and-retokenize truncation dance is replaced
        by plain ``truncation=True`` — identical for prompts within the limit.)
        """
        full_texts = [KANDINSKY5_PROMPT_TEMPLATE.format(p) for p in prompts]
        max_allowed_len = self.crop_start + self.max_sequence_length

        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_allowed_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        embeds = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )["hidden_states"][-1][:, self.crop_start :]

        attention_mask = inputs["attention_mask"][:, self.crop_start :]
        lengths = attention_mask.sum(1)
        cu_seqlens = build_cu_seqlens(lengths)
        return embeds.to(dtype), cu_seqlens

    def _encode_clip(self, prompts: list[str], dtype: torch.dtype) -> torch.Tensor:
        """CLIP pooled encode (pipeline ``_encode_prompt_clip``): 77 tokens,
        ``pooler_output`` → ``[B, 768]``."""
        inputs = self.tokenizer_2(
            prompts,
            max_length=KANDINSKY5_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        pooled = self.text_encoder_2(**inputs)["pooler_output"]
        return pooled.to(dtype)

    # --- Phase 5: Training Loop Hooks ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Channels-first cache/VAE latents → channels-LAST transformer layout.

        ``[B, C, F, H, W]`` → ``[B, F, H, W, C]`` (a still's 4D ``[B, C, H, W]``
        is lifted to F=1 first). Records the latent grid ``(F, H, W)`` for the
        RoPE position args.
        """
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
        out = to_channels_last(latents)
        _, f, h, w, _ = out.shape
        self._latent_shape = (f, h, w)
        return out

    def attach_conditioning(self, batch: dict[str, Any], latents: torch.Tensor) -> None:
        """I2V: stash the clean first-frame latent (the clip's own frame 0).

        Called on the RAW channels-first latents BEFORE noising/transposing.
        T2V is a no-op. Gating (Bernoulli, single-frame guard) lives in
        :meth:`Kandinsky5Trainer._attach_conditioning`.
        """
        if not self.is_i2v:
            return
        if self.BATCH_FIRST_FRAME_LATENT in batch:
            return
        lat = latents if latents.ndim == 5 else latents.unsqueeze(2)
        batch[self.BATCH_FIRST_FRAME_LATENT] = lat[:, :, :1, :, :].detach().clone()

    def _i2v_conditioning_engaged(self) -> bool:
        """True only when i2v first-frame conditioning applies THIS step.

        Requires the per-step ``_i2v_active`` flag AND a multi-frame latent —
        a single still has no frames left to predict once frame 0 is the
        conditioning frame (the loss mask would drop every token)."""
        if not getattr(self, "_i2v_active", False):
            return False
        if self._latent_shape is None:
            return False
        return self._latent_shape[0] > 1

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-match lerp in the ``[0, 1000]`` space (channels-last 5D).

        ``noisy = (t/1000) * noise + (1 - t/1000) * latents`` — the single
        place the scale division happens (pinned by
        ``assert_flowmatch_timestep_contract``).

        I2V (engaged): frame 0 stays CLEAN (``noisy[:, 0:1] = latents[:, 0:1]``)
        — mirroring the upstream I2V pipeline where frame 0 holds the image
        latent and the scheduler never updates it.
        """
        t = timesteps / FLOWMATCH_SCALE
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        noisy = t * noise + (1.0 - t) * latents
        if self._i2v_conditioning_engaged():
            noisy = torch.cat([latents[:, :1], noisy[:, 1:]], dim=1)
        return noisy

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``Kandinsky5Transformer3DModel`` forward → velocity (channels-last).

        - ``hidden_states``: channels-last ``[B, F, H, W, C(+C+1)]`` — when the
          checkpoint is ``visual_cond``, the input concatenates
          ``[latents, visual_cond, mask]`` on the LAST dim (zeros for T2V; the
          first-frame latent + mask=1 in frame 0 for engaged I2V).
        - ``timestep``: RAW ``[0, 1000]`` — the sinusoidal time embedder reads
          the FlowMatchEuler value directly (no internal /1000).
        - RoPE: ``visual_rope_pos = [arange(F), arange(H/2), arange(W/2)]``
          (patch (1,2,2)), ``text_rope_pos = arange(max true text length)``
          from cu_seqlens, ``scale_factor`` from the pixel resolution.
        - ``sparse_params=None``: dense attention (exact superset of the Pro
          checkpoint's inference-time "nabla" sparse approximation).

        Returns the velocity prediction ``[B, F, H, W, C]`` — same
        channels-last space as the prepared target.
        """
        if not isinstance(text_embeddings, TextEncoderOutput):
            raise TypeError(
                "Kandinsky5 forward_pass requires a TextEncoderOutput "
                f"(got {type(text_embeddings).__name__})"
            )
        embeds = text_embeddings.embeddings
        pooled = text_embeddings.require_pooled()
        cu_seqlens = text_embeddings.require_attention_mask()

        hidden_states = noisy_input
        if self.visual_cond:
            visual_cond = torch.zeros_like(noisy_input)
            cond_mask = noisy_input.new_zeros(*noisy_input.shape[:-1], 1)
            if self._i2v_conditioning_engaged():
                first_frame = batch.get(self.BATCH_FIRST_FRAME_LATENT)
                if first_frame is None:
                    raise ValueError(
                        "I2V forward_pass requires batch["
                        f"'{self.BATCH_FIRST_FRAME_LATENT}'] (first-frame latent)."
                    )
                # [B, C, 1, H, W] → channels-last [B, 1, H, W, C]
                ff = to_channels_last(first_frame).to(
                    device=noisy_input.device, dtype=noisy_input.dtype
                )
                visual_cond[:, :1] = ff
                cond_mask[:, :1] = 1.0
            hidden_states = torch.cat(
                [noisy_input, visual_cond, cond_mask], dim=-1
            )

        f, h, w = self._latent_grid()
        device = noisy_input.device
        visual_rope_pos = self.build_visual_rope_pos(f, h, w, device)
        text_rope_pos = self.build_text_rope_pos(cu_seqlens, device)
        scale_factor = get_scale_factor(h * self.vae_spatial, w * self.vae_spatial)

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=embeds,
            timestep=timesteps,  # RAW [0, 1000]
            pooled_projections=pooled,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=scale_factor,
            sparse_params=None,
            return_dict=False,
        )
        return output[0] if isinstance(output, (tuple, list)) else output

    @staticmethod
    def build_visual_rope_pos(
        f: int, h: int, w: int, device: torch.device | str
    ) -> list[torch.Tensor]:
        """RoPE positions for the latent grid (patch (1,2,2) → H/2, W/2).

        Replicates the pipeline's ``visual_rope_pos`` construction:
        ``[arange(F_lat), arange(H_lat // 2), arange(W_lat // 2)]``.
        """
        return [
            torch.arange(f, device=device),
            torch.arange(h // 2, device=device),
            torch.arange(w // 2, device=device),
        ]

    @staticmethod
    def build_text_rope_pos(
        cu_seqlens: torch.Tensor, device: torch.device | str
    ) -> torch.Tensor:
        """``arange(max true text length)`` from cu_seqlens (pipeline math)."""
        return torch.arange(int(cu_seqlens.diff().max().item()), device=device)

    def _latent_grid(self) -> tuple[int, int, int]:
        """Latent ``(F, H, W)`` recorded by :meth:`prepare_latents`."""
        if self._latent_shape is None:
            raise RuntimeError(
                "prepare_latents() must run before forward_pass() — it records "
                "the latent (F, H, W) the RoPE position args need."
            )
        return self._latent_shape

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        from .saver import Kandinsky5Saver

        return Kandinsky5Saver()

    def get_save_metadata(self) -> dict[str, str]:
        return {"modelspec.architecture": f"kandinsky5.0-{self.mode}"}

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Single stack of ``visual_transformer_blocks``."""
        topology: list[dict[str, Any]] = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "visual_transformer_blocks", None)
            if blocks is not None:
                per_block_mb = 120 if len(blocks) <= 32 else 614
                topology.append(
                    {
                        "name": "visual_transformer_blocks",
                        "attr_path": "visual_transformer_blocks",
                        "count": len(blocks),
                        "approx_vram_mb": per_block_mb,
                    }
                )
        return topology


def tiny_transformer_config(**overrides: Any) -> dict[str, Any]:
    """Config for a tiny CPU-instantiable ``Kandinsky5Transformer3DModel``.

    Shared by the unit tests (driver / sampler / saver / portability) so every
    suite exercises the SAME miniature architecture: 1 visual + 1 text block,
    model_dim 32, head_dim 16 (axes 8+4+4).
    """
    cfg: dict[str, Any] = dict(
        in_visual_dim=4,
        out_visual_dim=4,
        in_text_dim=16,
        in_text_dim2=8,
        time_dim=16,
        patch_size=(1, 2, 2),
        model_dim=32,
        ff_dim=64,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(8, 4, 4),
        visual_cond=False,
        attention_type="regular",
    )
    cfg.update(overrides)
    return cfg
