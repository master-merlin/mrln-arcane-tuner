"""OmniGen2 model loader — manifest-driven via GenericComponentLoader.

Checkpoint layout (``OmniGen2/OmniGen2``, verified via the HF API
2026-07-13): a diffusers-style repo root with subfolders ``transformer/``
(vendored ``OmniGen2Transformer2DModel`` — ``model_index.json`` maps it to
a repo-local ``transformer_omnigen2`` module diffusers cannot resolve
without the external ``omnigen2`` package), ``mllm/``
(``Qwen2_5_VLForConditionalGeneration``, Qwen2.5-VL-3B-Instruct lineage),
``processor/`` (``Qwen2_5_VLProcessor``), ``vae/`` (the FLUX.1-dev
``AutoencoderKL`` verbatim, 16ch, scaling 0.3611 / shift 0.1159) and
``scheduler/`` (upstream's OWN FlowMatchEuler variant — ``model_index.json``
maps it to a repo-local ``scheduling_flow_match_euler_discrete`` module).
``model_index.json``'s ``_class_name`` is ``OmniGen2Pipeline`` — a class
that exists in NO installed library, so a naive
``DiffusionPipeline.from_pretrained`` would crash; every component is
loaded individually (krea2/boogu_image pattern).

Stock components load through the generic manifest path:

- ``text_encoder`` (``mllm/`` subfolder) — stock transformers-4.57
  ``Qwen2_5_VLForConditionalGeneration.from_pretrained`` (the same class
  qwen_image/longcat_image/kandinsky5 already load; ``mllm/config.json``
  is standard transformers format, no krea2-style rope-config shim).
- ``processor`` (``Qwen2_5_VLProcessor``, ``processor/`` subfolder) — used
  ONLY for its tokenizer + chat template (``OmniGen2Driver.encode_text``;
  the vision side is never invoked — recon §1 in driver.py). Not a torch
  model.
- ``vae`` — stock ``diffusers.AutoencoderKL``.
- ``scheduler`` — the VENDORED
  ``vendor/schedulers/scheduling_flow_match_euler_discrete.py::
  FlowMatchEulerDiscreteScheduler``, loaded via ITS OWN
  ``ConfigMixin.from_pretrained`` against the checkpoint's
  ``scheduler/scheduler_config.json`` (``dynamic_time_shift: true``).
  Deliberately NOT the stock diffusers class of the same name —
  incompatible time direction and config keys (see the vendored module's
  header). Not a torch model.

The transformer is NOT in the manifest — :meth:`OmniGen2Loader.load`
overrides the base ``load()`` and loads the vendored
``OmniGen2Transformer2DModel`` directly via its own
``from_pretrained(transformer_dir, torch_dtype=...)`` (clean ``ModelMixin``;
the checkpoint's ``transformer/config.json`` keys map 1:1 onto the vendored
class's ``__init__`` signature — verified field-by-field 2026-07-13).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

_SCHEDULER_CLASS = (
    "app.engine.models.families.omnigen2.vendor.schedulers."
    "scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler"
)


class OmniGen2Loader(GenericComponentLoader):
    """Load OmniGen2 components; transformer via from_pretrained override."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        """Manifest for text_encoder (mllm), processor, vae, scheduler.

        The transformer is loaded by the overridden :meth:`load` (not in
        the manifest) — it needs the vendored class, not the generic
        ``from_pretrained`` string resolver.
        """
        return [
            # -- Text encoder (Qwen2.5-VL mllm, text-only use) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen2_5_VLForConditionalGeneration",
                subfolder="mllm",
                candidates=["mllm"],
            ),
            # -- Processor (tokenizer + chat template) --
            ComponentSpec(
                key="processor",
                hf_class="transformers.Qwen2_5_VLProcessor",
                subfolder="processor",
                candidates=["processor"],
                is_torch_model=False,
            ),
            # -- VAE (FLUX.1-dev AutoencoderKL, 16ch) --
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
        """Load OmniGen2 components: manifest path, transformer by hand.

        Args:
            definition: Model definition with component paths / repo IDs.
            torch_dtype: Dtype for the transformer weights. Defaults to
                ``bfloat16``.
            initial_device: Device to place the transformer on after load.
                ``None`` defaults to ``self.device``.

        Returns:
            Dict of loaded components including ``"unet"`` for the
            transformer.
        """
        # 1. Manifest components (text_encoder, processor, vae, scheduler).
        #    This also sets self._root_path.
        components = await super().load(definition, torch_dtype, initial_device)

        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )
        root = Path(self._root_path)

        # 2. Vendored transformer by hand.
        transformer_dir = root / "transformer"
        if not transformer_dir.is_dir():
            if not (root / "config.json").is_file():
                raise FileNotFoundError(
                    f"No 'transformer/' subfolder and no config.json at root: {root}. "
                    "Place the OmniGen2 transformer weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "omnigen2.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.omnigen2.vendor.models.transformers.transformer_omnigen2 import (  # noqa: PLC0415
            OmniGen2Transformer2DModel,
        )

        self.logger.info(
            "omnigen2.loading_transformer",
            path=str(transformer_dir),
            dtype=str(dtype),
        )
        model = OmniGen2Transformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=dtype,
        )
        model = model.to(target_device)
        model.eval()

        components["unet"] = model

        self.logger.info(
            "omnigen2.load.complete",
            components=list(components.keys()),
        )
        return components
