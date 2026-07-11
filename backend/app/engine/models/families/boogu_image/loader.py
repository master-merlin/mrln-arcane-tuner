"""Boogu-Image model loader — manifest-driven via GenericComponentLoader.

Checkpoint layout (both ``Boogu-Image-0.1-Base`` / ``-Turbo`` repos, verified
task-3-brief.md): a single diffusers-style repo root with subfolders
``transformer/`` (vendored ``BooguImageTransformer2DModel``), ``mllm/``
(``Qwen3VLForConditionalGeneration``, ~8.77B), ``processor/``
(``Qwen3VLProcessor``), ``vae/`` (FLUX-style ``AutoencoderKL``, 16ch), and
``scheduler/`` (the vendored custom scheduler). ``model_index.json``'s
``_class_name`` is ``BooguImagePipeline``/``BooguImageTurboPipeline`` — a
class we do NOT have installed, so a naive ``DiffusionPipeline.from_pretrained``
would crash; every component below is loaded individually instead (the
krea2/ideogram4 pattern).

Stock/near-stock components load through the generic manifest path:

- ``text_encoder`` (Qwen3-VL mllm, ``mllm/`` subfolder) — stock
  ``Qwen3VLForConditionalGeneration.from_pretrained``. Unlike krea2's
  text_encoder, Boogu's mllm ``config.json`` is ALREADY transformers-4.57
  shaped (verified, task-3-brief.md) — there is NO ``rope_parameters`` ->
  ``rope_scaling`` translation shim here (krea2 has one for its
  transformers-5.2-format checkpoint; do not copy it for Boogu).
- ``processor`` (``Qwen3VLProcessor``, ``processor/`` subfolder) — drives
  ``apply_chat_template`` in ``BooguImageDriver.encode_text`` (used by both
  the trainer's TE cache and the sampler); the loader just loads it as-is.
  Not a torch model.
- ``vae`` (FLUX-style ``AutoencoderKL``, ``vae/`` subfolder, 16ch, scaling
  0.3611 / shift 0.1159 — both come from ``architecture_params``, not the
  loader).
- ``scheduler`` — the VENDORED
  ``vendor/schedulers/scheduling_flow_match_euler_discrete_time_shifting.py::
  FlowMatchEulerDiscreteScheduler``, loaded via ITS OWN
  ``ConfigMixin.from_pretrained`` against the checkpoint's
  ``scheduler/scheduler_config.json`` (``do_shift``, ``dynamic_time_shift``,
  ``time_shift_version``, ``seq_len``). This is deliberately NOT the stock
  ``diffusers.FlowMatchEulerDiscreteScheduler`` — same class *name*,
  incompatible config keys and shift math. Not a torch model (no
  parameters/buffers to place on device), so ``is_torch_model=False``.

The transformer is NOT in the manifest — :meth:`BooguImageLoader.load`
overrides the base ``load()`` and loads the vendored
``BooguImageTransformer2DModel`` directly via its own
``from_pretrained(transformer_dir, torch_dtype=...)`` (mirrors krea2's
clean-``ModelMixin`` pattern: no fp8 dequant or manual shard stitching
required, unlike Ideogram 4).

``use_prompt_tuning: false`` in both shipped definitions — the checkpoint
ships no ``PromptEmbedding`` weights and none are loaded here; a
``BooguImageTransformer2DModel`` built with ``prompt_tuning_configs={
"use_prompt_tuning": False}`` never instantiates that submodule, so
``from_pretrained`` neither expects nor requires those weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

_SCHEDULER_CLASS = (
    "app.engine.models.families.boogu_image.vendor.schedulers."
    "scheduling_flow_match_euler_discrete_time_shifting.FlowMatchEulerDiscreteScheduler"
)


class BooguImageLoader(GenericComponentLoader):
    """Load Boogu-Image components; transformer via from_pretrained override."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        """Return component manifest for text_encoder(mllm), processor, vae, scheduler.

        The transformer is loaded by the overridden :meth:`load` (not in the
        manifest), because it needs the hand-rolled steps in :meth:`load`
        rather than the generic ``from_pretrained`` string resolver.
        """
        return [
            # -- Text encoder (Qwen3-VL mllm) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="mllm",
                candidates=["mllm"],
            ),
            # -- Processor (Qwen3VLProcessor) --
            ComponentSpec(
                key="processor",
                hf_class="transformers.Qwen3VLProcessor",
                subfolder="processor",
                candidates=["processor"],
                is_torch_model=False,
            ),
            # -- VAE (FLUX-style AutoencoderKL, 16ch) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Scheduler: VENDORED class, never the stock diffusers one of
            #    the same name (see module docstring). Not a torch model. --
            ComponentSpec(
                key="scheduler",
                hf_class=_SCHEDULER_CLASS,
                subfolder="scheduler",
                candidates=["scheduler"],
                is_torch_model=False,
            ),
            # NOTE: transformer is NOT listed here — loaded by hand in load().
        ]

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load Boogu-Image components: manifest path, transformer by hand.

        Loads text_encoder / processor / vae / scheduler through the generic
        manifest path, then loads ``BooguImageTransformer2DModel`` directly
        via its own ``from_pretrained(transformer_dir, torch_dtype=...)`` call
        (clean ``ModelMixin``, no fp8 dequant required).

        Args:
            definition: Model definition with component paths / repo IDs.
            torch_dtype: Dtype for the transformer weights. Defaults to
                ``bfloat16``.
            initial_device: Device to place the transformer on after load.
                ``None`` defaults to ``self.device``.

        Returns:
            Dict of loaded components including ``"unet"`` for the transformer.
        """
        # 1. Load stock/manifest components (text_encoder, processor, vae,
        #    scheduler) via the generic manifest path. This also sets
        #    self._root_path.
        components = await super().load(definition, torch_dtype, initial_device)

        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )
        root = Path(self._root_path)

        # 2. Load the vendored transformer by hand.
        transformer_dir = root / "transformer"
        if not transformer_dir.is_dir():
            if not (root / "config.json").is_file():
                raise FileNotFoundError(
                    f"No 'transformer/' subfolder and no config.json at root: {root}. "
                    "Place the Boogu-Image transformer weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "boogu_image.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )

        self.logger.info(
            "boogu_image.loading_transformer",
            path=str(transformer_dir),
            dtype=str(dtype),
        )
        model = BooguImageTransformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=dtype,
        )
        model = model.to(target_device)
        model.eval()

        components["unet"] = model

        self.logger.info(
            "boogu_image.load.complete",
            components=list(components.keys()),
        )
        return components
