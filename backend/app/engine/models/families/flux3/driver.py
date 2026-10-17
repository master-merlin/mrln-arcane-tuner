"""FLUX.3 Image Klein driver -- family-specific training behaviour (LANE-144 spec D3).

The text path mirrors the reference trainer (diffusers PR #15046 ``ec288418``
``examples/dreambooth/train_dreambooth_lora_flux3_klein.py`` ``get_qwen3vl_prompt_embeds``):

- each prompt goes through the Qwen3-VL chat template (``add_generation_prompt=True``);
- the WINDOW is ``W = min(ceil(n_real / 80) * 80, cap)``, ``n_real`` the longest templated prompt's
  token count truncated at the cap (``te.max_length``, default 512);
- the prompts are right-padded to ``W`` and the text encoder is called with the attention mask AND
  explicit position ids ``arange(W).masked_fill(attn == 0, 1)``: the DiT has no text mask and attends
  to the padded embeddings, so their values are part of the recipe;
- hidden states 4, 8, ..., 32 are stacked and flattened to ``(B, W, 8 * 2560 = 20480)``.

Ids and time: image ids ``(0, row * 0.5, col * 0.5, 0)`` (float), text ids ``(0, 0, 0, pos)``; the
transformer receives ``t`` in ``[0, 1]`` and multiplies by 1000 itself, so the driver divides the
pipeline's ``[0, 1000]`` timesteps by 1000 exactly once. Latents: 2x2 patchify + the VAE's BatchNorm
running statistics through the ``flux2`` helpers (the VAE is ``AutoencoderKLFlux2``).
"""

from __future__ import annotations

import math
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput

logger = structlog.get_logger(__name__)

# Spec D3: Qwen3-VL hidden states 4, 8, ..., 32 (of embeddings + 32 layers).
DEFAULT_SELECT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
# Spec D3: the text window is a multiple of this many tokens (reference TEXT_PAD_MULTIPLE).
TEXT_PAD_MULTIPLE = 80
# Spec D3: the window cap (reference max_sequence_length default).
DEFAULT_MAX_LENGTH = 512
# Spec D3: the image RoPE scale on the H/W axes of the position ids (reference IMAGE_ROPE_SCALE).
IMAGE_ROPE_SCALE = 0.5

# BFL-native -> diffusers module-path renames: the semantics of diffusers' FLUX.3 LoRA converter
# (PR #15046 ec288418 ``lora_conversion_utils.py`` ``_convert_non_diffusers_flux3_lora_to_diffusers``,
# :2554-2571). The YAML lists LoRA targets in native names; PEFT wraps the vendored (diffusers-named)
# model, so the driver hands PEFT the diffusers paths.
_NATIVE_PREFIX_RENAMES = (
    ("content_mode_blocks.", "mode_blocks."),
    ("single_blocks.", "single_transformer_blocks."),
)
_NATIVE_LEAF_RENAMES = (
    (".q_proj", ".attn.to_q"),
    (".k_proj", ".attn.to_k"),
    (".v_proj", ".attn.to_v"),
    (".attn_out", ".attn.to_out.0"),
    (".mlp_in", ".ff.linear_in"),
    (".mlp_out", ".ff.linear_out"),
)


def native_to_diffusers_module(native: str) -> str:
    """One BFL-native module path (``content_mode_blocks.image.0.q_proj``) -> its diffusers path."""
    name = native.removeprefix("diffusion_model.")
    for old, new in _NATIVE_PREFIX_RENAMES:
        name = name.replace(old, new)
    for old, new in _NATIVE_LEAF_RENAMES:
        if name.endswith(old):
            name = name[: -len(old)] + new
            break
    return name


def text_window(n_real: int, cap: int = DEFAULT_MAX_LENGTH) -> int:
    """Spec D3: ``min(ceil(n_real / 80) * 80, cap)`` with ``n_real`` truncated at the cap (>= 1 token)."""
    n = max(1, min(int(n_real), int(cap)))
    return min(math.ceil(n / TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE, int(cap))


def make_img_ids(h: int, w: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``(h * w, 4)`` float ids ``(0, row * 0.5, col * 0.5, 0)`` over the patchified grid."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1),
    ).to(dtype=torch.float32)
    ids[:, 1:3] = ids[:, 1:3] * IMAGE_ROPE_SCALE
    return ids.to(device) if device is not None else ids


def make_txt_ids(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``(seq_len, 4)`` float ids ``(0, 0, 0, pos)``: text advances along the L axis."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len),
    ).to(dtype=torch.float32)
    return ids.to(device) if device is not None else ids


class Flux3Driver(IModelDriver):
    """FLUX.3 Image Klein (Base) driver: Qwen3-VL 8-layer text context, 4-axis ids, unit-range t."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None  # the Qwen3-VL processor (chat template + tokenizer)
        self._components: dict[str, Any] = {}
        self._current_img_ids: torch.Tensor | None = None

        arch = getattr(definition, "architecture_params", {}) or {}
        self.select_layers: tuple[int, ...] = tuple(arch.get("te.select_layers", DEFAULT_SELECT_LAYERS))
        self.max_length: int = int(arch.get("te.max_length", DEFAULT_MAX_LENGTH))

    # --- Loading & component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        # The loader registers the AutoProcessor under ``tokenizer`` (it lives in text_encoder/).
        self.tokenizer = components.get("tokenizer", components.get("processor"))
        self.logger.info(
            "flux3_config", select_layers=list(self.select_layers), te_max_length=self.max_length,
            context_width=len(self.select_layers) * 2560,
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

    def release_text_encoders(self) -> None:
        """Null the attr get_text_encoders() reads."""
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """The YAML's native-named targets (spec D4) as the vendored model's diffusers module paths."""
        native = list(getattr(self.definition, "lora_targetable_modules", None) or [])
        return [native_to_diffusers_module(name) for name in native]

    def init_scheduler(self) -> Any:
        """Flow matching with the fixed 6.93 shift; no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text-encoder LoRA is out of scope for FLUX.3 (TE frozen)."""
        return []

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Swappable block groups: the joint single-stream stack (the per-mode ModuleDict is
        not an attribute path). Empty until the transformer is assigned."""
        model = self.get_primary_model()
        single = getattr(model, "single_transformer_blocks", None) if model is not None else None
        if single is None:
            return []
        return [{
            "name": "single_blocks",
            "attr_path": "single_transformer_blocks",
            "count": len(single),
            "approx_vram_mb": 320,
        }]

    def get_layer_manifest(self) -> Any:
        from app.engine.core.layer_manifest import BlockInfo, ModelLayerManifest

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            groups: list[tuple[str, str, Any]] = []
            mode_blocks = getattr(model, "mode_blocks", None)
            if mode_blocks is not None and "image" in mode_blocks:
                groups.append(("mode_blocks.image", "joint", mode_blocks["image"]))
            groups.append(("txt_mode_blocks", "joint", getattr(model, "txt_mode_blocks", None)))
            groups.append(("single_transformer_blocks", "single",
                           getattr(model, "single_transformer_blocks", None)))
            for prefix, block_type, container in groups:
                if container is None:
                    continue
                for i, block in enumerate(container):
                    blocks.append(BlockInfo(
                        name=f"{prefix}.{i}", block_type=block_type,
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=len(blocks),
                    ))
        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Text encoding ---

    def _processor_tokenizer(self) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("flux3: the Qwen3-VL processor is not loaded (released or never assigned)")
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)

    def _template(self, captions: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True,
            )
            for caption in captions
        ]

    def real_token_counts(self, captions: list[str]) -> list[int]:
        """Each caption's chat-templated token count, truncated at the cap."""
        tok = self._processor_tokenizer()
        counts = []
        for text in self._template(captions):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
            counts.append(int(enc["input_ids"].shape[1]))
        return counts

    def window_for(self, captions: list[str]) -> int:
        """The window the longest of ``captions`` needs (spec D3 formula)."""
        counts = self.real_token_counts(list(captions)) or [1]
        return text_window(max(counts), self.max_length)

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, window: int | None = None,
    ) -> TextEncoderOutput:
        """Encode ``captions`` to ``(B, W, 20480)``.

        ``window`` fixes W (the trainer's job-wide ``text_window``; prompts are truncated at it);
        without it W is the batch's own window by the spec D3 formula.
        """
        if self.text_encoder is None:
            raise RuntimeError("flux3: the Qwen3-VL text encoder is not loaded (released or never assigned)")
        captions = list(captions)
        if window is None:
            window = self.window_for(captions)
        window = int(min(window, self.max_length))
        texts = self._template(captions)
        enc = self._processor_tokenizer()(
            texts, return_tensors="pt", padding="max_length", padding_side="right",
            truncation=True, max_length=window,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        position_ids = torch.arange(window, device=self.device)[None].masked_fill(attn == 0, 1)
        with torch.no_grad():
            out = self.text_encoder(
                input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                output_hidden_states=True, use_cache=False,
            )
            hidden = out.hidden_states
            stacked = torch.stack([hidden[k] for k in self.select_layers], dim=1)
            b, n, s, d = stacked.shape
            ctx = stacked.permute(0, 2, 1, 3).reshape(b, s, n * d)
        return TextEncoderOutput(embeddings=ctx.to(device=self.device, dtype=dtype))

    # --- Training-loop hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``Flux3Transformer2DModel`` forward: velocity for packed ``(B, L_img, 128)`` latents.

        ``timesteps`` arrive in the pipeline's ``[0, 1000]`` scale; the transformer multiplies by 1000
        itself, so they are divided here exactly once. No guidance, no reference image, no target t.
        """
        context = text_embeddings.embeddings if hasattr(text_embeddings, "embeddings") else text_embeddings
        img_ids = self._current_img_ids
        if img_ids is None or img_ids.shape[0] != noisy_input.shape[1]:
            raise RuntimeError(
                "flux3: no image ids for this latent grid -- prepare_latents must run before forward_pass "
                f"(ids {None if img_ids is None else tuple(img_ids.shape)}, tokens {noisy_input.shape[1]})"
            )
        output = self.transformer(
            hidden_states=noisy_input,
            encoder_hidden_states=context,
            timestep=timesteps / 1000.0,
            img_ids=img_ids.to(noisy_input.device),
            txt_ids=make_txt_ids(context.shape[1], noisy_input.device),
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify + BN-normalise + pack ``(B, 32, H, W)`` -> ``(B, H/2 * W/2, 128)``; stores the ids."""
        from app.engine.models.families.flux2.utils import _pack_spatial, bn_normalize, patchify_latents

        normalized = bn_normalize(patchify_latents(latents), self.vae)
        self._current_img_ids = make_img_ids(normalized.shape[2], normalized.shape[3], self.device)
        return _pack_spatial(normalized).to(self.device)

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Patchify + pack noise WITHOUT the BN normalisation (noise stays N(0, 1))."""
        from app.engine.models.families.flux2.utils import _pack_spatial, patchify_latents

        return _pack_spatial(patchify_latents(noise)).to(self.device)

    # --- LoRA output ---

    def get_saver(self):
        """The native-key FLUX.3 saver (spec D5)."""
        from app.engine.models.families.flux3.saver import Flux3Saver  # noqa: PLC0415

        return Flux3Saver()
