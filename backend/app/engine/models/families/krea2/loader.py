"""Krea2 model loader — manifest-driven via GenericComponentLoader.

Stock components (tokenizer, text_encoder, vae) load through the generic
manifest path.  The transformer is NOT in the manifest because
``Krea2Transformer2DModel`` is a vendored class absent from the
``diffusers 0.38`` namespace; the generic ``from_pretrained`` string
resolver cannot build it.

Instead, :meth:`Krea2Loader.load` overrides the base ``load()`` and loads
the vendored transformer by calling
``Krea2Transformer2DModel.from_pretrained(transformer_dir, ...)`` directly
(it is a clean ``ModelMixin`` with ``register_to_config`` — no fp8 dequant
or manual shard stitching required, unlike Ideogram 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Krea2Loader(GenericComponentLoader):
    """Load Krea2 components; transformer via vendored from_pretrained override."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        """Return component manifest for tokenizer, text_encoder, and vae.

        The transformer is loaded by the overridden :meth:`load` (not in the
        manifest), because ``Krea2Transformer2DModel`` is vendored and is not
        registered in the ``diffusers 0.38`` namespace.
        """
        vae_class = self._detect_vae_class()

        return [
            # -- Tokenizer --
            # AutoTokenizer (NOT the slow Qwen2Tokenizer named in model_index.json):
            # the checkpoint ships only a fast tokenizer.json (no vocab.json/merges.txt),
            # so the slow class fails with vocab_file=None. AutoTokenizer loads the fast one.
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                # The checkpoint's tokenizer_config.json stores
                # ``extra_special_tokens`` as a LIST (newer-transformers format),
                # but transformers 4.57 expects a dict and crashes on
                # ``list.keys()``. Override with {} — the 13 ChatML/vision
                # special tokens are already defined as special added-tokens in
                # tokenizer.json, so they remain functional (verified:
                # <|im_start|>=151644, <|im_end|>=151645 stay single tokens).
                load_kwargs={"extra_special_tokens": {}},
            ),
            # NOTE: text_encoder (Qwen3-VL) is NOT in the manifest — it is loaded
            # by hand in load() with a config translation. The checkpoint's
            # text_encoder/config.json was saved by transformers 5.2 (it uses the
            # 5.x ``rope_parameters`` key); pinned transformers 4.57 reads
            # ``rope_scaling`` and crashes on ``NoneType.get``. The weights load
            # fine on 4.57 once the config is translated (verified: 4.44B params).
            # -- VAE (AutoencoderKLQwenImage — same as qwen_image) --
            ComponentSpec(
                key="vae",
                hf_class=vae_class,
                subfolder="vae",
                candidates=["vae"],
            ),
            # NOTE: transformer is NOT listed here — loaded by hand in load().
        ]

    @staticmethod
    def _detect_vae_class() -> str:
        """Detect whether the QwenImage-specific VAE class is available."""
        try:
            from diffusers.models import AutoencoderKLQwenImage  # noqa: F401
            return "diffusers.models.AutoencoderKLQwenImage"
        except ImportError:
            return "diffusers.AutoencoderKL"

    @staticmethod
    def _translate_qwen3vl_rope_config(config: Any) -> Any:
        """Translate a transformers-5.x Qwen3-VL config for pinned 4.57.

        The Krea-2 text_encoder config is saved by transformers 5.2, which stores
        rotary config under ``text_config.rope_parameters``. transformers 4.57's
        ``Qwen3VLTextRotaryEmbedding`` reads ``config.rope_scaling`` and calls
        ``.get(...)`` on it unconditionally → ``AttributeError`` when it is None.
        Inject a 4.57-shaped ``rope_scaling`` derived from ``rope_parameters`` so
        the (otherwise-compatible) weights load. No-op if ``rope_scaling`` is
        already present. Mutates and returns ``config``.
        """
        tc = getattr(config, "text_config", config)
        rp = getattr(tc, "rope_parameters", None)
        if rp is not None and getattr(tc, "rope_scaling", None) is None:
            tc.rope_scaling = {
                "rope_type": rp.get("rope_type", "default"),
                "mrope_section": rp.get("mrope_section", [24, 20, 20]),
                "mrope_interleaved": rp.get("mrope_interleaved", True),
            }
            if "rope_theta" in rp:
                tc.rope_theta = rp["rope_theta"]
        return config

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load Krea2 components: stock via manifest, transformer via vendored class.

        Loads tokenizer / text_encoder / vae through the generic manifest path,
        then loads the vendored ``Krea2Transformer2DModel`` directly via its own
        ``from_pretrained(transformer_dir, torch_dtype=...)`` call (clean ModelMixin,
        no fp8 dequant required).

        Args:
            definition: Model definition with component paths / repo IDs.
            torch_dtype: Dtype for the transformer weights. Defaults to ``bfloat16``.
            initial_device: Device to place the transformer on after load. ``None``
                defaults to ``self.device``.

        Returns:
            Dict of loaded components including ``"unet"`` for the transformer.
        """
        # 1. Load stock components (tokenizer, vae) via the generic manifest path.
        #    (text_encoder + transformer are loaded by hand below.)
        #    This also sets self._root_path.
        components = await super().load(definition, torch_dtype, initial_device)

        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )
        root = Path(self._root_path)

        # 2. Load the Qwen3-VL text encoder by hand with a config translation
        #    (the checkpoint config is transformers-5.2 format; see manifest note).
        te_dir = root / "text_encoder"
        if te_dir.is_dir():
            from transformers import AutoConfig, Qwen3VLModel

            te_config = self._translate_qwen3vl_rope_config(
                AutoConfig.from_pretrained(str(te_dir)),
            )
            self.logger.info("krea2.loading_text_encoder", path=str(te_dir))
            text_encoder = Qwen3VLModel.from_pretrained(
                str(te_dir),
                config=te_config,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            text_encoder = text_encoder.to(target_device)
            text_encoder.eval()
            components["text_encoder"] = text_encoder

        # 3. Load the vendored transformer by hand.
        transformer_dir = root / "transformer"
        if not transformer_dir.is_dir():
            if not (root / "config.json").is_file():
                raise FileNotFoundError(
                    f"No 'transformer/' subfolder and no config.json at root: {root}. "
                    "Place the Krea-2 transformer weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "krea2.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.krea2.vendor.transformer_krea2 import (
            Krea2Transformer2DModel,
        )

        self.logger.info(
            "krea2.loading_transformer",
            path=str(transformer_dir),
            dtype=str(dtype),
        )
        model = Krea2Transformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=dtype,
        )
        model = model.to(target_device)
        model.eval()

        components["unet"] = model

        self.logger.info(
            "krea2.load.complete",
            components=list(components.keys()),
        )
        return components
