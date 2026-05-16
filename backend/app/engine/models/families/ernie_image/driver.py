"""ERNIE-Image model driver -- family-specific training behavior.

Implements :class:`IModelDriver` for Baidu ERNIE-Image.

Key family characteristics (verified against the diffusers 0.38 source
at ``diffusers/models/transformers/transformer_ernie_image.py`` and
``diffusers/pipelines/ernie_image/pipeline_ernie_image.py``):

*  Single-stream DiT with shared AdaLN modulation (``model.layers``).
*  Latents live in **patchified** space ``[B, 128, H/2, W/2]`` — 2x2
   spatial patches of the FLUX.2 VAE's 32-ch latent packed into the
   channel dim.
*  Latent BN-normalization via the VAE's ``running_mean`` /
   ``running_var`` (same trick as FLUX.2); raw N(0,1) noise stays
   un-normalized (interpolated at t).
*  Text encoder: Mistral3-derived, per-prompt encoding (no chat
   template), uses ``hidden_states[-2]`` (second-to-last layer) padded
   to the longest in-batch caption.  ``text_lens`` carries the per-sample
   valid-token count -- the transformer builds its attention mask
   internally, so no ``attention_mask`` is passed to forward.
*  Flow-matching, velocity prediction (``target = noise - latents``).
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


# Number of tokens generated per encoded prompt is variable; we cap the
# padded sequence length at this many tokens to keep VRAM bounded.
DEFAULT_TE_MAX_LENGTH = 512

# Layer index extracted from the text encoder.  The official pipeline
# uses ``hidden_states[-2]`` (second-to-last); we follow suit.
DEFAULT_TE_HIDDEN_LAYER_INDEX = -2


class ErnieImageDriver(IModelDriver):
    """ERNIE-Image family driver."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params (populated in assign_components)
        self.te_max_length: int = DEFAULT_TE_MAX_LENGTH
        self.te_hidden_layer_index: int = DEFAULT_TE_HIDDEN_LAYER_INDEX
        self.text_in_dim: int = 3072

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded ERNIE-Image components and cache architecture params."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.te_max_length = int(arch.get("te.max_length", DEFAULT_TE_MAX_LENGTH))
        self.te_hidden_layer_index = int(
            arch.get("te.hidden_layer_index", DEFAULT_TE_HIDDEN_LAYER_INDEX),
        )

        # text_in_dim is what the transformer's text_proj expects.  Pull
        # it from the loaded checkpoint config — never trust the YAML
        # default, since the upstream class default (2560) does not match
        # the Mistral3 hidden_size used by the official 8B checkpoint.
        if self.transformer is not None:
            self.text_in_dim = int(
                getattr(self.transformer.config, "text_in_dim", 3072),
            )
        else:
            self.text_in_dim = int(arch.get("transformer.text_in_dim", 3072))

        self.logger.info(
            "ernie_image_config",
            te_max_length=self.te_max_length,
            te_hidden_layer_index=self.te_hidden_layer_index,
            text_in_dim=self.text_in_dim,
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
        """ERNIE-Image LoRA targets -- attn + MLP (comprehensive).

        Module suffixes inside each ``ErnieImageSharedAdaLNBlock``:
            self_attention.to_q/k/v             -> attention projections
            self_attention.to_out.0             -> attention output proj
            mlp.gate_proj / mlp.up_proj         -> SwiGLU gate / up
            mlp.linear_fc2                      -> SwiGLU down (note: NOT
                                                   ``down_proj``; the upstream
                                                   class uses the asymmetric
                                                   ``linear_fc2`` name)

        Definitions YAML may override via ``lora_targetable_modules``.
        """
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if definition_targets:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        return [
            "self_attention.to_q",
            "self_attention.to_k",
            "self_attention.to_v",
            "self_attention.to_out.0",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.linear_fc2",
        ]

    def init_scheduler(self) -> Any:
        """Flow matching -- no external scheduler needed at train time."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """ERNIE-Image loads in bf16 (YaRN RoPE in the TE is unstable in fp16)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for ERNIE-Image."""
        return []

    def get_layer_manifest(self) -> Any:
        """ERNIE-Image layer manifest with one block list (``model.layers``)."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            layers = getattr(model, "layers", None)
            if layers is not None:
                for i, block in enumerate(layers):
                    blocks.append(BlockInfo(
                        name=f"layers.{i}",
                        block_type="single",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=i,
                    ))

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Per-prompt encode, then pad to the longest in-batch caption.

        Mirrors ``ErnieImagePipeline.encode_prompt`` + ``_pad_text``.
        Returns variable-length valid tokens (via attention mask carrying
        the per-sample lengths) padded with zeros.

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype for the returned embeddings.

        Returns:
            ``TextEncoderOutput`` with:
            - ``embeddings`` ``[B, Tmax, text_in_dim]``
            - ``attention_mask`` ``[B, Tmax]`` (1 = valid token, 0 = pad)
              Lengths reconstructible via ``attention_mask.sum(dim=1)``.
        """
        per_prompt_hidden: list[torch.Tensor] = []

        with torch.no_grad():
            for caption in captions:
                ids = self.tokenizer(
                    caption,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.te_max_length,
                    padding=False,
                )["input_ids"]

                if len(ids) == 0:
                    bos = getattr(self.tokenizer, "bos_token_id", None)
                    ids = [bos if bos is not None else 0]

                input_ids = torch.tensor([ids], device=self.device)
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = outputs.hidden_states[self.te_hidden_layer_index][0]
                per_prompt_hidden.append(hidden)

        # Pad each prompt's hidden states up to the longest in the batch.
        lens = torch.tensor(
            [t.shape[0] for t in per_prompt_hidden],
            device=self.device, dtype=torch.long,
        )
        t_max = int(lens.max().item()) if len(lens) > 0 else 0
        text_in_dim = per_prompt_hidden[0].shape[-1] if per_prompt_hidden else self.text_in_dim
        batch_size = len(per_prompt_hidden)

        text_bth = torch.zeros(
            (batch_size, t_max, text_in_dim),
            device=self.device, dtype=dtype,
        )
        attention_mask = torch.zeros(
            (batch_size, t_max), device=self.device, dtype=torch.long,
        )
        for i, hidden in enumerate(per_prompt_hidden):
            n = hidden.shape[0]
            text_bth[i, :n, :] = hidden.to(dtype=dtype)
            attention_mask[i, :n] = 1

        return TextEncoderOutput(
            embeddings=text_bth,
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
        """``ErnieImageTransformer2DModel`` forward -- predict velocity.

        Args:
            noisy_input: Patchified noisy latents ``[B, 128, H/2, W/2]``.
            timesteps: Scaled timesteps ``[0, 1000]`` (transformer divides
                by 1000 internally; we keep them in ``[0, 1000]`` for
                compatibility with the flow-matching pipeline base).
            text_embeddings: Either a ``(embeddings, attention_mask)``
                tuple from the cached path, or a raw ``[B, T, D]`` tensor
                from the direct-encode path (in which case lengths are
                assumed to be the full T).
            batch: Full batch dict (unused).

        Returns:
            Velocity prediction ``[B, out_channels, H/2, W/2]`` matching
            ``noisy_input``.
        """
        if isinstance(text_embeddings, tuple):
            text_bth, attention_mask = text_embeddings
        else:
            text_bth = text_embeddings
            attention_mask = None

        if attention_mask is not None:
            text_lens = attention_mask.sum(dim=1).to(
                dtype=torch.long, device=text_bth.device,
            )
        else:
            text_lens = torch.full(
                (text_bth.shape[0],),
                text_bth.shape[1],
                dtype=torch.long,
                device=text_bth.device,
            )

        # Transformer expects t in [0, 1] (its internal Timesteps embedding
        # multiplies by 1000).  Our pipeline keeps timesteps in [0, 1000].
        model_timesteps = timesteps / 1000.0

        output = self.transformer(
            hidden_states=noisy_input,
            timestep=model_timesteps,
            text_bth=text_bth,
            text_lens=text_lens,
            return_dict=False,
        )

        return output[0] if isinstance(output, tuple) else output

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify + BN-normalize VAE latents into the model's input space.

        VAE encode produces ``[B, 32, H, W]``; we 2x2-patchify to
        ``[B, 128, H/2, W/2]`` (matching ``transformer.in_channels=128``)
        then apply the VAE's BN running statistics so the latents have
        unit variance in the patched space.
        """
        from app.engine.models.families.ernie_image.utils import (
            bn_normalize,
            patchify_latents,
        )

        patched = patchify_latents(latents)
        return bn_normalize(patched, self.vae).to(self.device)

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Patchify noise WITHOUT BN normalization.

        Flow matching interpolates between clean BN-normalized latents
        (zero mean / unit variance) and raw ``N(0, 1)`` noise.  Both
        endpoints already have unit variance, so noise must stay raw
        after patchification.
        """
        from app.engine.models.families.ernie_image.utils import patchify_latents

        return patchify_latents(noise).to(self.device)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return ERNIE-Image ComfyUI-compatible LoRA saver."""
        from app.engine.models.families.ernie_image.saver import ErnieImageSaver

        return ErnieImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """ERNIE-Image block topology: a single ``layers`` group."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            layers = getattr(model, "layers", None)
            if layers is not None:
                topology.append({
                    "name": "layers",
                    "attr_path": "layers",
                    "count": len(layers),
                    "approx_vram_mb": 440,
                })
        return topology
