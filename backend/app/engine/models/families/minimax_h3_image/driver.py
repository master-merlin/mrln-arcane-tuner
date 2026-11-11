"""MiniMax-H3-Image model driver: the family's training surface.

The target list, block topology and head grid come from the ``ModelDefinition``.
On top of that the driver owns every model-specific step of a training
iteration:

- **Text**: the shared Qwen3-VL layer tap (``minimax_h3_shared.text``; tap
  index and token cap from the definition), cached under the key prefix
  ``minimax_h3_image|`` so no entry is ever shared with the video family.
- **Clock**: the shared inverted flow-match convention
  (``minimax_h3_shared.flow``): ``t = 1 - sigma``, ``x_t = t*x0 + (1-t)*noise``,
  target ``v = x0 - noise``, ``t`` unscaled in ``[0, 1]``.
- **One sigma per batch**: the transformer takes ONE head index per forward
  (``forward(..., pdd_step_index: int)``), so a step draws one sigma from the
  shared ``TimestepSampler`` and uses it, with its head, for every item.
- **The head for a sigma**: the nearest sigma of the definition's head grid
  (``settings.head_for_sigma``).
- **The packed sequence** ``[text | target]`` of the reference pipeline
  (diffusers PR #15230 ``modular_pipelines/minimax_h3_image/before_denoise.py``
  ``build_image_packed_sequence``): one sequence per item with that item's TRUE
  text length (no padding row ever enters the transformer, which runs
  unmasked), text rows tagged 1 and image rows 0 (the AdaLN row is
  ``timestep_index * 3 + tag`` inside the model), a float64 ``(t, h, w)``
  rotary grid: text at ``(k, 0, 0)``, the image as one frame at
  ``t = num_text`` with its ``h``/``w`` axes aspect-normalised by the square
  root of the latent area and scaled by 32.
- **The loss on the target rows**: the model returns the image rows only
  (``index_select(1, image_indices)``), unpatchified back to the latent's
  shape, so the shared MSE sees nothing else.

``init_scheduler`` returns ``None`` and must keep exactly that body: flow
matching needs no scheduler object at train time, and ``return None`` is the
one body ``hook_dispatch.TRIVIAL_BODIES`` reads as "not overridden". The
CLOBBER hooks this driver does override (``add_noise``, ``compute_target``,
``sample_timesteps``) are each delegated explicitly by the trainer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.models.families.minimax_h3_shared import flow as h3_flow
from app.engine.models.families.minimax_h3_shared import text as h3_text

from . import settings as h3i_settings

# The modality tags of the packed sequence (the transformer's docstring: "0 image, 1 text").
IMAGE_TAG = 0
TEXT_TAG = 1
# The spatial rotary axes are normalised by the square root of the latent area and scaled by 32.
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class H3ImageLayout:
    """The structural inputs of one packed ``[text | target]`` sequence."""

    position_ids: torch.Tensor  # (seq_len, 3) float64
    token_tags: torch.Tensor  # (seq_len,) long
    timestep_indices: torch.Tensor  # (seq_len,) long, every row at the one timestep
    image_indices: torch.Tensor  # (num_target_rows,) long
    text_indices: torch.Tensor  # (num_text,) long


def _spatial_axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalised rotary axis: ``dim // patch`` coordinates centred on
    the unit interval, right endpoint excluded, scaled by 32 (float64)."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


class MiniMaxH3ImageDriver(IModelDriver):
    """MiniMax-H3-Image driver (text-to-image)."""

    TE_CACHE_PREFIX = "minimax_h3_image|"

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}
        self._layouts: dict[tuple[int, int, int, tuple[int, int, int]], H3ImageLayout] = {}

    # --- Loading & component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire the loader's components: ``tokenizer`` (the processor),
        ``text_encoder`` (released before the DiT loads), ``vae`` (pixel-adapted)
        and ``transformer`` (deferred; ``unet`` is the pipeline's alias of it)."""
        self._components = components
        self.transformer = components.get("transformer", components.get("unet"))
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {"text_encoder": self.text_encoder} if self.text_encoder is not None else {}

    def get_lora_targets(self) -> list[str]:
        """The definition's curated target list, verbatim (the YAML is the one source)."""
        return list(self.definition.lora_targetable_modules)

    def get_te_lora_targets(self) -> list[str]:
        """The text encoder never trains (released before the transformer loads)."""
        return []

    def get_block_topology(self) -> list[dict[str, Any]]:
        """The definition's block topology, verbatim (authoritative with or without weights)."""
        return list(self.definition.block_topology)

    def init_scheduler(self) -> Any:
        # Flow matching: no scheduler object at train time. This exact body is the
        # trivial baseline hook_dispatch recognises; see the module docstring.
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Every MiniMax-H3-Image component ships bf16 (the transformer keeps its
        ``_keep_in_fp32_modules`` in float32 itself)."""
        return torch.bfloat16

    def get_saver(self) -> Any:
        """The native-layout LoRA saver (``saver.py``, keys from ``lora_keys.py``)."""
        from .saver import MiniMaxH3ImageSaver

        return MiniMaxH3ImageSaver()

    # --- Text encoding ---

    def _tap_index(self) -> int:
        return h3i_settings.architecture_int(self.definition, "te.hidden_state_tap_index")

    def _prompt_max_tokens(self) -> int:
        return h3i_settings.architecture_int(self.definition, "te.max_length")

    def tokenizer_fingerprint(self) -> str:
        cached = getattr(self, "_tokenizer_fp", None)
        if cached is None:
            cached = self._tokenizer_fp = h3_text.tokenizer_fingerprint(self.tokenizer)
        return cached

    def te_cache_scope(self) -> str:
        """The disk cache's path segment: tap index + tokenizer."""
        return h3_text.te_cache_scope(self._tap_index(), self.tokenizer_fingerprint())

    def te_cache_key(self, prompt: str) -> str:
        """Every input the embedding is a function of, under this family's prefix."""
        return h3_text.te_cache_key(
            self.TE_CACHE_PREFIX, self.definition.id, self._tap_index(), self.tokenizer_fingerprint(), prompt
        )

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        """The layer-tap conditioning, each caption encoded alone, right-padded."""
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError(
                "minimax_h3_image encode_text: text_encoder/tokenizer not assigned; "
                "the encoder was released or assign_components() never ran"
            )
        return h3_text.encode_captions(
            self.tokenizer,
            self.text_encoder,
            captions,
            dtype,
            tap=self._tap_index(),
            max_tokens=self._prompt_max_tokens(),
            device=self.device,
        )

    def text_encoder_weight_bytes(self) -> int:
        return h3_text.module_weight_bytes(self.text_encoder)

    def release_text_encoders(self) -> None:
        """Drop the driver's attribute and its entry in the shared component dict."""
        self.text_encoder = None
        self._components.pop("text_encoder", None)

    def assert_text_encoder_released(self) -> None:
        """The encoder and the DiT never coexist: refuse the DiT load while the
        encoder is still held anywhere the driver can see."""
        if self.text_encoder is not None or "text_encoder" in self._components:
            gb = self.text_encoder_weight_bytes() / 1e9
            raise RuntimeError(
                f"text encoder still resident ({gb:.2f} GB) at DiT load; "
                "release it (cache the embeddings) before materialising the transformer"
            )

    # --- The latent cache ---

    def latent_cache_fingerprint(self, bucket_divisibility: int) -> str:
        """Short hash of every input a cached latent depends on: the pixel convention
        (``PIXEL_ADAPTER_VERSION``, read at call time), the VAE's class and config (its
        normalisation statistics included) and the bucket rule (the multiple every bucket side
        snaps to). Nothing that only shapes the sequence at train time is in it, so a settings
        change never re-encodes a dataset."""
        from app.engine.models.families.minimax_h3_shared import pixel_adapter

        vae = self.vae
        if vae is None:
            raise RuntimeError(
                "minimax_h3_image latent_cache_fingerprint: no VAE assigned; the cache key "
                "cannot be derived without the encoder's identity"
            )
        inner = getattr(vae, "inner", vae)
        config = getattr(inner, "config", None)
        if config is None:
            config_dict: dict[str, Any] = {}
        elif hasattr(config, "items"):  # diffusers FrozenDict or a plain dict
            config_dict = dict(config.items())
        else:
            config_dict = dict(vars(config))
        payload = json.dumps(
            {
                "pixel_adapter": pixel_adapter.PIXEL_ADAPTER_VERSION,
                "vae_class": type(inner).__name__,
                "vae_config": config_dict,
                "bucket_divisibility": int(bucket_divisibility),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    # --- The head grid ---

    def head_sigmas(self) -> list[float]:
        return h3i_settings.head_sigmas(self.definition)

    def head_for_sigma(self, sigma: float | torch.Tensor) -> int:
        return h3i_settings.head_for_sigma(sigma, self.head_sigmas())

    # --- Training-loop hooks (CLOBBER: each delegated by the trainer) ---

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """ONE sigma for the whole batch from the shared ``TimestepSampler`` in
        the config's mode, returned as ``t = 1 - sigma`` for every item."""
        from app.engine.strategies.timestep_sampling import TimestepSampler

        mode = config.get("timestep_sampling")
        if not mode:
            raise ValueError(
                "minimax_h3_image sample_timesteps: the config names no timestep_sampling; "
                "the trainer sets the family default at setup"
            )
        kwargs: dict[str, Any] = {"latents": latents, "progress": progress}
        if generator is not None:
            kwargs["generator"] = generator
        sigma = TimestepSampler.sample(mode, 1, device, config, **kwargs)
        t = h3_flow.sigma_to_t(sigma.float()).clamp(0.0, 1.0).reshape(1)
        return t.expand(int(batch_size)).contiguous()

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """``x_t = t*x0 + (1-t)*noise`` (the shared convention; ``t = 1`` clean)."""
        return h3_flow.add_noise(latents, noise, timesteps)

    def compute_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """``v = x0 - noise``: the opposite sign of the house default."""
        return h3_flow.velocity_target(latents, noise)

    # --- The packed sequence ---

    def patch_size(self) -> tuple[int, int, int]:
        # A PEFT wrapper forwards ``config`` to the model it wraps.
        config = getattr(self.transformer, "config", None)
        size = getattr(config, "patch_size", None) if config is not None else None
        if size is None:
            raise RuntimeError("minimax_h3_image: the transformer (and its patch_size) is not assigned")
        return tuple(int(p) for p in size)  # type: ignore[return-value]

    @staticmethod
    def patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
        """``(B, C, H, W)`` latents -> ``(B, rows, C * pt * ph * pw)`` rows,
        row-major over the patch grid (one frame)."""
        pt, ph, pw = patch_size
        b, c, h, w = latents.shape
        if pt != 1 or h % ph or w % pw:
            raise ValueError(f"latents of shape {tuple(latents.shape)} do not split into {patch_size} patches")
        x = latents.reshape(b, c, 1, 1, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(b, (h // ph) * (w // pw), c * ph * pw).contiguous()

    @staticmethod
    def unpatchify(rows: torch.Tensor, height: int, width: int, patch_size: tuple[int, int, int]) -> torch.Tensor:
        """The inverse of :meth:`patchify`: rows -> ``(B, C, H, W)``."""
        _, ph, pw = patch_size
        b, _, dim = rows.shape
        c = dim // (ph * pw)
        x = rows.reshape(b, height // ph, width // pw, c, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, height, width).contiguous()

    def packed_layout(
        self, num_text: int, latent_height: int, latent_width: int, patch_size: tuple[int, int, int]
    ) -> H3ImageLayout:
        """The ``[text | target]`` layout of one item (cached per geometry)."""
        key = (int(num_text), int(latent_height), int(latent_width), tuple(patch_size))
        layout = self._layouts.get(key)
        if layout is not None:
            return layout
        _, ph, pw = patch_size
        rows = (latent_height // ph) * (latent_width // pw)
        seq_len = num_text + rows
        position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
        position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
        sqrt_area = float(np.sqrt(latent_height * latent_width))
        h_axis = _spatial_axis(latent_height, ph, sqrt_area)
        w_axis = _spatial_axis(latent_width, pw, sqrt_area)
        grid_h, grid_w = torch.meshgrid(h_axis, w_axis, indexing="ij")
        position_ids[num_text:, 0] = float(num_text)
        position_ids[num_text:, 1] = grid_h.reshape(-1)
        position_ids[num_text:, 2] = grid_w.reshape(-1)
        token_tags = torch.full((seq_len,), IMAGE_TAG, dtype=torch.long)
        token_tags[:num_text] = TEXT_TAG
        layout = H3ImageLayout(
            position_ids=position_ids,
            token_tags=token_tags,
            timestep_indices=torch.zeros(seq_len, dtype=torch.long),
            image_indices=torch.arange(num_text, seq_len),
            text_indices=torch.arange(num_text),
        )
        self._layouts[key] = layout
        return layout

    def predict_velocity(
        self, noisy: torch.Tensor, text_rows: torch.Tensor, t: torch.Tensor, head: int
    ) -> torch.Tensor:
        """ONE item through the transformer: ``noisy`` ``(1, C, H, W)``,
        ``text_rows`` ``(1, L, D)`` (its true length), ``t`` the one timestep;
        returns the velocity of the target rows in ``(1, C, H, W)``."""
        if self.transformer is None:
            raise RuntimeError("minimax_h3_image forward_pass: transformer not assigned")
        patch = self.patch_size()
        height, width = int(noisy.shape[-2]), int(noisy.shape[-1])
        layout = self.packed_layout(int(text_rows.shape[1]), height, width, patch)
        device = noisy.device
        out = self.transformer(
            hidden_states=self.patchify(noisy, patch),
            encoder_hidden_states=text_rows,
            timestep=t.reshape(1).to(device=device, dtype=torch.float32),
            timestep_indices=layout.timestep_indices.to(device),
            token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device),
            image_indices=layout.image_indices.to(device),
            text_indices=layout.text_indices.to(device),
            pdd_step_index=int(head),
            return_dict=False,
        )
        rows = out[0] if isinstance(out, (tuple, list)) else out
        return self.unpatchify(rows, height, width, patch)

    @staticmethod
    def _text_rows(text_embeddings: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb = getattr(text_embeddings, "embeddings", None)
        mask = getattr(text_embeddings, "attention_mask", None)
        if emb is None:
            emb, mask = text_embeddings
        return emb, mask

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """The batch's velocity at its ONE timestep, through the head of that
        sigma: one packed sequence per item, each with its true text rows.
        Items at different timesteps are refused (one head per forward)."""
        t = timesteps.reshape(-1).to(device=noisy_input.device, dtype=torch.float32)
        if t.numel() != noisy_input.shape[0]:
            raise ValueError(
                f"minimax_h3_image forward_pass: {t.numel()} timesteps for a batch of {noisy_input.shape[0]}"
            )
        if bool((t != t[0]).any()):
            raise ValueError(
                f"minimax_h3_image trains one sigma per batch (one output head per forward), got timesteps {t.tolist()}"
            )
        head = self.head_for_sigma(h3_flow.t_to_sigma(t[0]))
        emb, mask = self._text_rows(text_embeddings)
        out = []
        for i in range(noisy_input.shape[0]):
            n_text = int(mask[i].sum()) if mask is not None else int(emb.shape[1])
            out.append(self.predict_velocity(noisy_input[i : i + 1], emb[i : i + 1, :n_text], t[:1], head))
        return torch.cat(out, dim=0)
