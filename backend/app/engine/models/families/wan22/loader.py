"""WAN 2.2 loader — dual-transformer MoE, manifest-driven.

Components (diffusers-format repo, ``Wan-AI/Wan2.2-{T2V,I2V}-A14B-Diffusers``):

- ``tokenizer``     : ``AutoTokenizer`` (UMT5)
- ``text_encoder``  : ``UMT5EncoderModel``
- ``vae``           : ``AutoencoderKLWan`` — kept fp32 (temporal VAE precision)
- ``unet``          : ``WanTransformer3DModel`` from ``transformer/`` — the
                      **high-noise** expert (active for ``t >= boundary``)
- ``unet_low``      : ``WanTransformer3DModel`` from ``transformer_2/`` — the
                      **low-noise** expert (active for ``t < boundary``)

Diffusers convention: ``transformer`` = high-noise, ``transformer_2`` =
low-noise (WAN 2.2 dual transformers selected by ``boundary_ratio``).

Unlike WAN 2.1 I2V, **WAN 2.2 I2V has NO CLIP image encoder** — diffusers
asserts ``image_embeds is None`` and conditions on the first-frame latent only.
So even the I2V manifest never loads an image encoder; the 36-channel concat is
built from the first-frame latent with ``encoder_hidden_states_image=None``.

Single-expert training (``expert_mode`` = ``"high"``/``"low"``) loads ONLY the
chosen transformer — the real VRAM save (the other ~14B expert is never read
from disk). The chosen expert is always loaded under the ``"unet"`` key so the
generic loop + driver treat it as the single active model; for ``"low"`` that
means ``transformer_2/`` is loaded as ``unet``.
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Wan22Loader(GenericComponentLoader):
    """Load WAN 2.2 dual-transformer components from a diffusers-format repo.

    Args:
        device: Target device for loaded components.
        expert_mode: ``"both"`` (default) loads both experts; ``"high"`` loads
            only ``transformer/`` and ``"low"`` only ``transformer_2/`` — each
            mapped to ``unet`` — so a single-expert run uses ~half the VRAM.
    """

    def __init__(self, device, expert_mode: str = "both") -> None:
        super().__init__(device)
        self.expert_mode = str(expert_mode or "both").lower()

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        manifest: list[ComponentSpec] = [
            # -- Tokenizer (UMT5) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder (UMT5-XXL) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.UMT5EncoderModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- VAE (Wan-VAE) — kept fp32 for temporal-decode precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLWan",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
        ]

        # Transformer specs depend on expert_mode. Single-expert loads exactly
        # ONE transformer under "unet" (the generic loop's primary model); the
        # driver knows the mode and wires it into the right expert slot.
        high_spec = ComponentSpec(
            key="unet",
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer",
            candidates=["transformer"],
            fallback_to_root=True,
        )
        low_spec = ComponentSpec(
            key="unet_low",
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer_2",
            candidates=["transformer_2"],
            fallback_to_root=True,
        )
        if self.expert_mode == "high":
            manifest.append(high_spec)
        elif self.expert_mode == "low":
            # Load transformer_2/ AS "unet" so it becomes the single primary.
            manifest.append(
                ComponentSpec(
                    key="unet",
                    hf_class="diffusers.WanTransformer3DModel",
                    subfolder="transformer_2",
                    candidates=["transformer_2"],
                    fallback_to_root=True,
                )
            )
        else:  # both (default) — high → "unet", low → "unet_low"
            manifest.append(high_spec)
            manifest.append(low_spec)

        # NOTE: no image_encoder/image_processor even for I2V — WAN 2.2 I2V is
        # first-frame-latent only (no CLIP-vision conditioning).
        return manifest
