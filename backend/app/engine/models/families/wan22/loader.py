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

Host-RAM sequencing (``defer_second_expert``)
---------------------------------------------
WAN 2.2 A14B is the repo's only dual-transformer family: TWO ~14B experts,
~28 GB bf16 EACH once resident. The phased loader stages every component on CPU
first (``initial_device="cpu"``) and they stay there through the whole TE/VAE
caching stretch, so a naive ``both`` load pins ~2×28 GB = ~56 GB of transformer
in host RAM (plus the ~10.6 GB text encoder) for the entire load — enough to
fill a 64 GB box and hang it (the reported bug). ``wan21`` (one expert, ~28 GB)
fits fine on the same machine.

When ``defer_second_expert=True`` (set by the trainer for ``both`` runs) the
**low-noise** expert is OMITTED from the Phase-A manifest and materialised on
demand via :meth:`load_second_expert` — the trainer calls that AFTER the high
expert has moved to the GPU, so host RAM holds at most ONE ~28 GB expert at any
instant. Peak host RAM for a ``both`` run therefore drops from ~67 GB to ≈ the
single-expert / ``wan21`` figure (~39 GB incl. TE). Everything downstream of the
materialisation is byte-identical to eager loading: the low expert is CPU-resident
exactly as it would have been just before ``prepare_for_training`` wraps it.
"""

from __future__ import annotations

from typing import Any

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
        defer_second_expert: When ``True`` (dual-expert runs only), the low-noise
            expert is left OUT of the Phase-A manifest and loaded later via
            :meth:`load_second_expert`, capping peak host RAM at one expert. See
            the module docstring for the rationale.
    """

    def __init__(
        self, device, expert_mode: str = "both", defer_second_expert: bool = False
    ) -> None:
        super().__init__(device)
        self.expert_mode = str(expert_mode or "both").lower()
        self.defer_second_expert = bool(defer_second_expert)

    # ── Transformer specs (shared by the manifest + deferred load) ──────────

    @staticmethod
    def _high_expert_spec() -> ComponentSpec:
        """High-noise expert (``transformer/``) under the primary ``unet`` key."""
        return ComponentSpec(
            key="unet",
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer",
            candidates=["transformer"],
            fallback_to_root=True,
        )

    @staticmethod
    def _low_expert_spec(key: str = "unet_low") -> ComponentSpec:
        """Low-noise expert (``transformer_2/``); ``key`` is ``unet_low`` in
        ``both`` mode, or ``unet`` when ``low`` is the single loaded expert."""
        return ComponentSpec(
            key=key,
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer_2",
            candidates=["transformer_2"],
            fallback_to_root=True,
        )

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
        if self.expert_mode == "high":
            manifest.append(self._high_expert_spec())
        elif self.expert_mode == "low":
            # Load transformer_2/ AS "unet" so it becomes the single primary.
            manifest.append(self._low_expert_spec(key="unet"))
        else:  # both (default) — high → "unet", low → "unet_low"
            manifest.append(self._high_expert_spec())
            # Host-RAM: in a dual-expert run the low expert is deferred out of
            # Phase A (loaded later by load_second_expert) so both ~28 GB experts
            # never sit on CPU together. See the module docstring.
            if not self.defer_second_expert:
                manifest.append(self._low_expert_spec())

        # NOTE: no image_encoder/image_processor even for I2V — WAN 2.2 I2V is
        # first-frame-latent only (no CLIP-vision conditioning).
        return manifest

    # ── Deferred low-noise expert (host-RAM sequencing) ─────────────────────

    def load_second_expert(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype,
        initial_device: str = "cpu",
    ) -> Any:
        """Materialise the deferred low-noise expert (``transformer_2/``).

        Reuses the root/path resolution + ``from_pretrained`` + meta-device +
        device-placement machinery from the initial :meth:`load` (which ran on
        THIS same loader instance and cached ``_root_path`` /
        ``_raw_safetensors_mode`` / ``_local_files_only``). Called by the trainer
        AFTER the high expert has moved to the GPU, so host RAM holds at most one
        ~28 GB expert at a time.

        Args:
            definition: The model definition (for path resolution fallbacks).
            torch_dtype: Loading dtype — pass the SAME dtype the initial load
                used (``driver.resolve_loading_dtype()`` → bf16) so the two
                experts are byte-identical in precision.
            initial_device: Placement after load. ``"cpu"`` (default) matches the
                low expert's device at ``prepare_for_training`` time under eager
                loading, keeping everything downstream identical.

        Returns:
            The loaded ``WanTransformer3DModel`` on ``initial_device``.
        """
        spec = self._low_expert_spec()
        root_path = getattr(self, "_root_path", None) or self._resolve_root(definition)
        self.logger.info(
            "wan22_load_second_expert",
            subfolder=spec.subfolder,
            dtype=str(torch_dtype),
            device=str(initial_device),
        )
        return self._load_single_spec(
            spec, definition, root_path, torch_dtype, initial_device
        )
