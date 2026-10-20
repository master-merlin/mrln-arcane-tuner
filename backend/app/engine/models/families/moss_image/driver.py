"""MOSS-Image driver -- family-specific training behaviour (LANE-165).

Text recipe -- MOSS's own, NOT flux3's (Evidence: the official pipeline ``encode_prompt``,
``.agent/workdir/moss/base_pipeline_moss_image.py:314-336``; research §1.5, §3.1):

- each caption, the empty one included, goes ALONE through the Qwen chat template
  (``add_generation_prompt=True``, ``enable_thinking=True``) and is tokenized UNPADDED, truncated at
  ``te.max_length`` (4096) tokens -- "unpadded individual forwards also avoid padding effects in
  Qwen GDN" (the pipeline's own comment);
- the embedding is the final-layer hidden state (``last_hidden_state``, 4096 wide), one layer, never
  a stack;
- a batch is right-padded to its longest row with a bool valid-token mask, which the transformer
  uses to drop the padding (``encoder_attention_mask``).

Time: the transformer receives the RAW ``[0, 1000]`` timestep (the pipeline passes
``scheduler.timesteps`` = ``sigma * 1000`` straight in, ``base_pipeline_moss_image.py:413``), so the
driver hands the pipeline's timesteps through UNCHANGED -- unlike flux3, which divides by 1000.
Latents: 2x2 patchify + the VAE's BatchNorm running statistics through the ``flux2`` helpers (the
VAE is ``AutoencoderKLFlux2``); the transformer takes spatial ``[B, 128, H/2, W/2]``, no packing.
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

# Research §1.5 / the pipeline's ``encode_prompt(max_sequence_length=4096)``.
DEFAULT_MAX_LENGTH = 4096
# Research §1.5 / the pipeline's ``apply_chat_template(..., enable_thinking=True)``.
DEFAULT_ENABLE_THINKING = True


def pad_rows(rows: list[torch.Tensor], dtype: torch.dtype, device: torch.device | str) -> TextEncoderOutput:
    """Right-pad ``[S_i, D]`` rows to ``(B, max S, D)`` with a bool mask of each row's true length.

    The pipeline's own batching (``encode_prompt``): zeros after each row, mask ``True`` on its tokens.
    """
    if not rows:
        raise ValueError("moss_image: no text rows to batch")
    length = max(int(row.shape[0]) for row in rows)
    emb = torch.zeros((len(rows), length, rows[0].shape[-1]), dtype=dtype, device=device)
    mask = torch.zeros((len(rows), length), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        n = int(row.shape[0])
        emb[i, :n] = row.to(device=device, dtype=dtype)
        mask[i, :n] = True
    return TextEncoderOutput(embeddings=emb, attention_mask=mask)


class MossImageDriver(IModelDriver):
    """MOSS-Image (Base / SFT) driver: Qwen3.5 final-layer context, raw 0-1000 timestep."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None  # the Qwen tokenizer carrying the chat template
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.max_length: int = int(arch.get("te.max_length", DEFAULT_MAX_LENGTH))
        self.enable_thinking: bool = bool(arch.get("te.enable_thinking", DEFAULT_ENABLE_THINKING))

    # --- Loading & component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer", components.get("processor"))
        self.logger.info("moss_image_config", te_max_length=self.max_length,
                         enable_thinking=self.enable_thinking)

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
        """The YAML's 252 block projections (``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``).

        The vendored model keeps the checkpoint's own module names, so the YAML names are the
        PEFT paths as they stand (no rename table, unlike flux3).
        """
        return list(getattr(self.definition, "lora_targetable_modules", None) or [])

    def init_scheduler(self) -> Any:
        """Flow matching; the trainer draws its own sigmas (``scheduler.training_shift``)."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text-encoder LoRA is out of scope for MOSS-Image (TE frozen)."""
        return []

    def get_block_topology(self) -> list[dict[str, Any]]:
        """The 42 uniform DiT blocks (``layers``). Empty until the transformer is assigned."""
        model = self.get_primary_model()
        layers = getattr(model, "layers", None) if model is not None else None
        if layers is None:
            return []
        return [{"name": "layers", "attr_path": "layers", "count": len(layers), "approx_vram_mb": 240}]

    def get_layer_manifest(self) -> Any:
        from app.engine.core.layer_manifest import BlockInfo, ModelLayerManifest

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        layers = getattr(model, "layers", None) if model is not None else None
        for i, block in enumerate(layers or []):
            blocks.append(BlockInfo(
                name=f"layers.{i}", block_type="single",
                param_count=sum(p.numel() for p in block.parameters()), depth_index=i,
            ))
        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Text encoding ---

    def te_cache_tag(self) -> str:
        """The recipe part of the TE-cache key: the template flags and the token cap.

        The caption is the other part (the cache filename hashes it); a change to any of the three
        misses every affected entry (spec scenario 8).
        """
        return f"moss_gen1_think{int(self.enable_thinking)}_cap{self.max_length}"

    def _te_device(self) -> torch.device:
        try:
            return next(self.text_encoder.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device(self.device)

    def encode_caption(self, caption: str) -> torch.Tensor:
        """ONE caption -> its ``[S, 4096]`` final-layer hidden state in the encoder's own dtype.

        Unpadded, batch size 1, truncated at ``max_length`` tokens; every row token is valid, so
        the mask is the row's length (``pad_rows``).
        """
        if self.text_encoder is None:
            raise RuntimeError("moss_image: the Qwen3.5 text encoder is not loaded (released or never assigned)")
        if self.tokenizer is None:
            raise RuntimeError("moss_image: the Qwen tokenizer is not loaded (released or never assigned)")
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        enc = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=self.max_length)
        device = self._te_device()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device) if "attention_mask" in enc else None
        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False, return_dict=True)
        return out.last_hidden_state[0]

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> TextEncoderOutput:
        """Encode ``captions`` one forward each -> right-padded ``(B, S, 4096)`` + bool mask."""
        rows = [self.encode_caption(caption) for caption in captions]
        return pad_rows(rows, dtype, self.device)

    # --- Training-loop hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``MOSSImageTransformer`` forward on spatial ``[B, 128, h, w]`` latents.

        ``timesteps`` arrive in the pipeline's ``[0, 1000]`` scale and are handed to the transformer
        UNCHANGED: it takes the raw value (spec scenario 2, ``test_moss_image_timestep_contract``).
        """
        if hasattr(text_embeddings, "embeddings"):
            context, mask = text_embeddings.embeddings, text_embeddings.attention_mask
        else:
            context, mask = text_embeddings, None
        output = self.transformer(
            hidden_states=noisy_input,
            timestep=timesteps,
            encoder_hidden_states=context,
            encoder_attention_mask=mask,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify + BN-normalise ``(B, 32, H, W)`` -> ``(B, 128, H/2, W/2)`` (spatial, unpacked)."""
        from app.engine.models.families.flux2.utils import bn_normalize, patchify_latents

        return bn_normalize(patchify_latents(latents), self.vae).to(self.device)

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Patchify noise WITHOUT the BN normalisation (noise stays N(0, 1))."""
        from app.engine.models.families.flux2.utils import patchify_latents

        return patchify_latents(noise).to(self.device)

    # --- LoRA output ---

    def get_saver(self):
        """The kohya-layout MOSS-Image saver (spec scenario 7)."""
        from app.engine.models.families.moss_image.saver import MossImageSaver  # noqa: PLC0415

        return MossImageSaver()
