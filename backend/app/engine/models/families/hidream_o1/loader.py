"""HiDream-O1 model loader — direct safetensors load into vendored custom class.

The HF repo ``HiDream-ai/HiDream-O1-Image`` is a flat weights-only repo
declaring ``architectures=["Qwen3VLForConditionalGeneration"]``, but the
checkpoint contains extra weights for the custom heads (``x_embedder``,
``final_layer2``, ``t_embedder1``) that the stock transformers class
silently drops. Our vendored ``Qwen3VLForConditionalGeneration``
(``app.engine.models.families.hidream_o1.vendor.qwen3_vl_transformers``)
adds those modules.

**Why we override ``load()``:** ``PreTrainedModel.from_pretrained()`` hangs
silently on our vendored class (PR A Task 4 finding). We bypass it entirely
using the standard ``accelerate.init_empty_weights`` + per-shard
``safetensors.torch.load_file`` + ``model.load_state_dict(strict=False)``
pattern. Validated working in PR A's spike — see ``vendor/spike_notes.md``
Task 4.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)

logger = structlog.get_logger(__name__)


class HiDreamO1Loader(GenericComponentLoader):
    """Load the unified HiDream-O1 model via direct-safetensors path.

    Single-component manifest (``unet``). Overrides ``load()`` to bypass
    ``transformers.PreTrainedModel.from_pretrained`` which hangs on our
    vendored class.
    """

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        # Single entry — informs the base pipeline that this family has
        # one primary model and no separate TE/VAE/tokenizer components.
        return [
            ComponentSpec(
                key="unet",
                # ``hf_class`` is informational here; we don't actually
                # use the standard from_pretrained path. Document the
                # vendored class location for traceability.
                hf_class=(
                    "app.engine.models.families.hidream_o1.vendor."
                    "qwen3_vl_transformers.Qwen3VLForConditionalGeneration"
                ),
                subfolder=None,
                candidates=[""],
                is_torch_model=True,
            ),
        ]

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load the unified model via init_empty_weights + per-shard safetensors.

        Args:
            definition: Model definition with the HF repo + revision.
            torch_dtype: Dtype for the loaded weights. Defaults to bfloat16
                (recipe default per ``spike_notes.md`` Task 3a).
            initial_device: Where to place the model after load. ``None``
                defaults to ``self.device``. Pass ``"cpu"`` for phased loading.

        Returns:
            ``{"unet": <model>}`` — driver consumes this directly.
        """
        # Resolve dtype + target device
        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )

        # Pull component spec from the definition for the repo + revision
        unet_spec = definition.components.get("unet") if definition.components else None
        repo_id = (
            getattr(unet_spec, "repo", None) or getattr(unet_spec, "path", None)
            if unet_spec else None
        ) or "HiDream-ai/HiDream-O1-Image"
        revision = getattr(unet_spec, "revision", None) if unet_spec else None

        self.logger.info(
            "hidream_o1.load.start",
            repo_id=repo_id,
            revision=revision,
            dtype=str(dtype),
            target_device=str(target_device),
        )

        # Step 1: resolve snapshot dir (HF cache); does NOT re-download if cached.
        t0 = time.time()
        try:
            snap_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_files_only=False,  # allow download if missing; existing cache hits will short-circuit
                ),
            )
        except OSError as e:
            self.logger.error(
                "hidream_o1.load.snapshot_failed",
                repo_id=repo_id,
                revision=revision,
                error=str(e),
            )
            raise RuntimeError(
                f"snapshot_download for {repo_id} failed: {e}",
            ) from e
        self.logger.info(
            "hidream_o1.load.snapshot_resolved",
            snap_dir=str(snap_dir),
            seconds=round(time.time() - t0, 2),
        )

        # Step 2: import our vendored class (lazy to keep import cost off the registry path)
        from app.engine.models.families.hidream_o1.vendor.qwen3_vl_transformers import (
            Qwen3VLForConditionalGeneration,
        )
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(snap_dir, trust_remote_code=False)
        config.torch_dtype = dtype

        # Step 3: instantiate the class with empty weights (no allocation)
        t1 = time.time()
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)
        model = model.to(dtype=dtype)  # set the dtype tag without materializing
        self.logger.info(
            "hidream_o1.load.empty_init",
            seconds=round(time.time() - t1, 2),
        )

        # Step 4: load each safetensors shard and accumulate into one state_dict
        shard_files = sorted(snap_dir.glob("model-*-of-*.safetensors"))
        if not shard_files:
            # Fallback: single-file safetensors
            single = list(snap_dir.glob("*.safetensors"))
            if not single:
                raise FileNotFoundError(
                    f"No safetensors found in {snap_dir}",
                )
            shard_files = single

        t2 = time.time()
        state_dict: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            try:
                shard_state = load_file(str(shard))
            except Exception as e:
                self.logger.error(
                    "hidream_o1.load.shard_failed",
                    shard=shard.name,
                    error=str(e),
                )
                raise RuntimeError(
                    f"failed to load shard {shard.name}: {e}",
                ) from e
            state_dict.update(shard_state)
            self.logger.debug(
                "hidream_o1.load.shard_loaded",
                shard=shard.name,
                keys=len(shard_state),
            )
        self.logger.info(
            "hidream_o1.load.all_shards_loaded",
            shards=len(shard_files),
            total_keys=len(state_dict),
            seconds=round(time.time() - t2, 2),
        )

        # Step 5: assign weights — strict=False because of any naming mismatches;
        # assign=True to avoid double-allocation (state_dict tensors become params).
        t3 = time.time()
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if unexpected:
            self.warnings.append(
                f"{len(unexpected)} unexpected key(s) in checkpoint "
                f"(first 5: {list(unexpected)[:5]})",
            )
            self.logger.warning(
                "hidream_o1.load.unexpected_keys",
                count=len(unexpected),
                sample=list(unexpected)[:5],
            )
        if missing:
            self.warnings.append(
                f"{len(missing)} missing key(s) — these modules will be "
                f"randomly initialized (first 5: {list(missing)[:5]})",
            )
            self.logger.warning(
                "hidream_o1.load.missing_keys",
                count=len(missing),
                sample=list(missing)[:5],
            )
        self.logger.info(
            "hidream_o1.load.state_assigned",
            missing=len(missing),
            unexpected=len(unexpected),
            seconds=round(time.time() - t3, 2),
        )

        # Step 5b: materialize any parameters still on meta device (params whose
        # names weren't in the checkpoint). load_state_dict(strict=False) silently
        # leaves them on meta; .to(device) would crash. Initialize them to zeros so
        # they're at least usable downstream (caller is responsible for proper init).
        meta_params = [
            (name, p) for name, p in model.named_parameters() if p.device.type == "meta"
        ]
        meta_buffers = [
            (name, b) for name, b in model.named_buffers() if b.device.type == "meta"
        ]
        if meta_params or meta_buffers:
            self.warnings.append(
                f"{len(meta_params)} parameter(s) and {len(meta_buffers)} buffer(s) "
                f"still on meta device after load — zeroing them (sample: "
                f"{[n for n, _ in meta_params[:3]]})",
            )
            self.logger.warning(
                "hidream_o1.load.meta_residual",
                params=len(meta_params),
                buffers=len(meta_buffers),
                sample_params=[n for n, _ in meta_params[:5]],
            )
            # Materialize meta tensors as zeros on CPU; the subsequent .to(device)
            # call moves them to the final device.
            for name, p in meta_params:
                new_p = torch.zeros(p.shape, dtype=p.dtype if p.dtype != torch.float32 or dtype is None else dtype)
                # Walk to the parent module and assign
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], torch.nn.Parameter(new_p, requires_grad=p.requires_grad))
            for name, b in meta_buffers:
                new_b = torch.zeros(b.shape, dtype=b.dtype)
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                # buffers: register again (preserves persistent flag is harder to detect; default True)
                parent.register_buffer(parts[-1], new_b)

        # Step 6: move to target device
        if target_device.type != "meta":
            t4 = time.time()
            model = model.to(target_device)
            self.logger.info(
                "hidream_o1.load.moved_to_device",
                target=str(target_device),
                seconds=round(time.time() - t4, 2),
            )

        model.eval()  # default eval; trainer flips to .train() in Task 11
        self.logger.info(
            "hidream_o1.load.complete",
            total_seconds=round(time.time() - t0, 2),
        )

        return {"unet": model}
