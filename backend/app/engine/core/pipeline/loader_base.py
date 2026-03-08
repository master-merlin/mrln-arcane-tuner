"""Generic component loader — one loader for all model families.

Each family declares a **component manifest** (list of ``ComponentSpec``)
describing *what* to load.  ``GenericComponentLoader`` handles *how*:
path resolution, ``from_pretrained``, device placement, error handling.

Family-specific loaders become thin subclasses that override only
``get_component_manifest()`` and (optionally) a few hooks.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelLoader
from app.engine.utils.model_utils import ModelPathResolver

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Component specification
# ---------------------------------------------------------------------------

@dataclass
class ComponentSpec:
    """Declares how to load one model component.

    Family loaders return a list of these from ``get_component_manifest()``.
    The generic loader iterates them to produce the final components dict.
    """

    key: str
    """Key in the returned components dict, e.g. ``"unet"``, ``"vae"``, ``"tokenizer"``."""

    hf_class: str
    """Fully qualified class name, e.g. ``"diffusers.UNet2DConditionModel"``."""

    subfolder: str | None = None
    """Subfolder within the repo root (e.g. ``"transformer"``, ``"vae"``)."""

    definition_key: str | None = None
    """Key in ``definition.components`` for explicit path override."""

    candidates: list[str] | None = None
    """Candidate subfolders for ``ModelPathResolver.find_component()``."""

    is_torch_model: bool = True
    """False for tokenizers / processors that should NOT be ``.to(device).eval()``."""

    dtype_override: torch.dtype | None = None
    """Force a specific dtype (e.g. ``torch.float32`` for VAE)."""

    load_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs passed to ``from_pretrained`` (e.g. ``device_map="auto"``)."""

    post_load_hook: str | None = None
    """Method name on the loader to call after loading (receives model + definition)."""

    use_subfolder_kwarg: bool = False
    """If True, pass ``subfolder=`` to ``from_pretrained`` instead of joining the path."""

    fallback_to_root: bool = False
    """If True and subfolder path doesn't exist, fall back to repo root."""

    root_key: str | None = None
    """If set on the FIRST spec, use this definition component key to resolve
    the repo root (e.g. ``"unet"`` for SDXL).  Only checked on the first spec."""

    separate_repo: bool = False
    """If True, this component lives in a separate repository.
    Uses ``definition_key`` to resolve its own root path independently."""


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

class GenericComponentLoader(IModelLoader):
    """Loads model components from a manifest — one loader for all families.

    Subclasses MUST override:
    - ``get_component_manifest()``

    Subclasses MAY override:
    - ``_resolve_root()``  (defaults to ``definition.components["repo"].path``)
    - ``_resolve_component_path()``  (for edge cases like separate VAE repos)
    - ``_resolve_dtype()``  (for family-specific dtype selection)
    - ``_post_load_*()``  (any method named via ``post_load_hook``)
    """

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        """Return the component manifest for this family.

        Subclasses MUST override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_component_manifest()",
        )

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load all components described by the manifest.

        Args:
            definition: Model definition with ``components`` paths.
            torch_dtype: Global dtype override (may be overridden per-component).
            initial_device: Device to place torch models on after loading.
                ``None`` (default) uses ``self.device`` (typically CUDA).
                Pass ``"cpu"`` for phased loading where components are
                moved to GPU individually as needed.

        Returns:
            Dict of loaded components keyed by ``ComponentSpec.key``.
        """
        target_device = initial_device or self.device
        root_path = self._resolve_root(definition)
        dtype = torch_dtype or self._resolve_dtype(definition)
        manifest = self.get_component_manifest(definition)

        self.logger.info(
            "generic_load_start",
            family=definition.family,
            root=root_path,
            dtype=str(dtype),
            initial_device=str(target_device),
            components=[s.key for s in manifest],
        )

        components: dict[str, Any] = {}

        for spec in manifest:
            # 1. Resolve path
            path = self._resolve_component_path(spec, definition, root_path)

            # 2. Determine dtype
            comp_dtype = spec.dtype_override or dtype

            # 3. Status broadcast
            display_name = spec.key.replace("_", " ").title()
            self.logger.info("loading_component", component=spec.key, display_name=display_name)

            # 4. Import HF class
            cls = self._import_class(spec.hf_class)

            # 5. Load via from_pretrained
            try:
                model = self._load_component(cls, path, comp_dtype, spec)
            except (OSError, ValueError, RuntimeError) as e:
                self.logger.error(
                    "component_load_failed",
                    key=spec.key,
                    path=path,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Failed to load {spec.key} from {path}: {e}",
                ) from e

            # 6. Device placement
            if spec.is_torch_model and isinstance(model, nn.Module):
                # Guard: models loaded with low_cpu_mem_usage (diffusers
                # default) may leave *missing* checkpoint parameters on the
                # ``meta`` device.  For example, Klein 9B has no
                # ``guidance_embedder`` weights so those stay on ``meta``
                # after ``from_pretrained``.
                #
                # The old approach called ``model.to_empty(device=...)``
                # which is **destructive** — it allocates new *un-initialised*
                # memory for EVERY parameter, wiping the loaded checkpoint.
                #
                # Fix: materialise only the individual meta-device params
                # (zeros), then move the whole model normally.
                meta_names = [
                    n for n, p in model.named_parameters()
                    if p.device.type == "meta"
                ]
                meta_bufs = [
                    n for n, b in model.named_buffers()
                    if b.device.type == "meta"
                ]
                if meta_names or meta_bufs:
                    self.logger.warning(
                        "meta_device_detected",
                        key=spec.key,
                        meta_params=meta_names,
                        meta_buffers=meta_bufs,
                        message=(
                            f"{len(meta_names)} param(s) and "
                            f"{len(meta_bufs)} buffer(s) on meta device — "
                            "materialising individually (checkpoint may lack "
                            "these weights)"
                        ),
                    )
                    for name in meta_names:
                        # Walk the module tree to the parent
                        parts = name.split(".")
                        parent = model
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        old = getattr(parent, parts[-1])
                        # Replace with a real tensor (zeros) on the target
                        new = torch.zeros(
                            old.shape, dtype=old.dtype, device=target_device,
                        )
                        setattr(parent, parts[-1], nn.Parameter(
                            new, requires_grad=old.requires_grad,
                        ))
                    for name in meta_bufs:
                        parts = name.split(".")
                        parent = model
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        old = getattr(parent, parts[-1])
                        new = torch.zeros(
                            old.shape, dtype=old.dtype, device=target_device,
                        )
                        parent.register_buffer(parts[-1], new)

                model = model.to(target_device)
                model.eval()

            # 7. Post-load hook
            if spec.post_load_hook:
                hook = getattr(self, spec.post_load_hook, None)
                if hook:
                    model = hook(model, definition) or model
                else:
                    self.logger.warning(
                        "post_load_hook_missing",
                        hook=spec.post_load_hook,
                        key=spec.key,
                    )

            # 8. Log param count for torch models
            if spec.is_torch_model and isinstance(model, nn.Module):
                n_params = sum(p.numel() for p in model.parameters()) / 1e9
                self.logger.info(
                    "component_loaded",
                    key=spec.key,
                    type=type(model).__name__,
                    params_B=round(n_params, 2),
                    device=str(target_device),
                )
            else:
                self.logger.info(
                    "component_loaded",
                    key=spec.key,
                    type=type(model).__name__,
                )

            components[spec.key] = model

        self.logger.info(
            "generic_load_complete",
            components=list(components.keys()),
        )
        return components

    # ── Overridable helpers ───────────────────────────────────────────────

    def _resolve_root(self, definition: ModelDefinition) -> str:
        """Resolve the root repo path from the definition.

        Default: looks for ``components["repo"]`` or ``components["path"]``
        or falls back to ``components["unet"]`` (SDXL convention).
        """
        for key in ("repo", "path", "unet"):
            # Check if any spec declares a root_key
            manifest = self.get_component_manifest(definition)
            for spec in manifest:
                if spec.root_key:
                    comp = definition.components.get(spec.root_key)
                    if comp:
                        resolved = ModelPathResolver.resolve(comp.path)
                        self._root_path = resolved
                        return resolved
                break  # only check first spec

            comp = definition.components.get(key)
            if comp:
                resolved = ModelPathResolver.resolve(comp.path)
                self._root_path = resolved
                return resolved
        raise ValueError(
            f"Definition '{definition.id}' must specify a 'repo', 'path', or 'unet' component.",
        )

    def _resolve_dtype(self, definition: ModelDefinition) -> torch.dtype:
        """Determine loading dtype from definition metadata."""
        precision = getattr(definition, "detected_precision", {}) or {}
        unet_prec = precision.get("unet", "torch.bfloat16")
        if "float16" in str(unet_prec):
            return torch.float16
        return torch.bfloat16

    def _resolve_component_path(
        self,
        spec: ComponentSpec,
        definition: ModelDefinition,
        root_path: str,
    ) -> str:
        """Resolve a single component's filesystem path.

        Resolution order:
        1. Explicit path from ``definition.components[spec.definition_key]``
        2. ``ModelPathResolver.find_component()`` with ``spec.candidates``
        3. ``os.path.join(root_path, spec.subfolder)``
        4. ``root_path`` (flat-layout fallback if ``fallback_to_root=True``)
        """
        # 1. Separate repository (e.g. SDXL VAE in its own repo)
        if spec.separate_repo and spec.definition_key:
            comp = definition.components.get(spec.definition_key)
            if comp:
                resolved = ModelPathResolver.resolve(comp.path)
                if resolved != root_path:
                    # Truly separate repo — load from its root
                    return resolved

        # 2. Explicit definition override
        if spec.definition_key:
            comp = definition.components.get(spec.definition_key)
            if comp:
                return ModelPathResolver.resolve(comp.path)

        # If using subfolder kwarg, the root is the path
        if spec.use_subfolder_kwarg:
            return root_path

        # 2. Smart discovery via find_component
        if spec.candidates:
            discovered = ModelPathResolver.find_component(
                definition, spec.key, root_path, candidates=spec.candidates,
            )
            if discovered:
                return discovered

        # 3. Subfolder join
        if spec.subfolder:
            joined = os.path.join(root_path, spec.subfolder)
            if os.path.isdir(joined):
                return joined
            # Flat-layout fallback
            if spec.fallback_to_root:
                return root_path

        # 4. Root path as final fallback
        return root_path

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _import_class(dotted_path: str) -> type:
        """Dynamically import a class from a dotted path like ``diffusers.UNet2DConditionModel``."""
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Invalid class path: {dotted_path}")
        module_path, class_name = parts
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _load_component(
        cls: type,
        path: str,
        dtype: torch.dtype,
        spec: ComponentSpec,
    ) -> Any:
        """Call ``from_pretrained`` with the appropriate kwargs."""
        kwargs: dict[str, Any] = {}

        # dtype for torch models
        if spec.is_torch_model:
            kwargs["torch_dtype"] = dtype

        # subfolder kwarg (e.g. SDXL's from_pretrained(root, subfolder="unet"))
        if spec.use_subfolder_kwarg and spec.subfolder:
            kwargs["subfolder"] = spec.subfolder

        # Extra kwargs from spec
        kwargs.update(spec.load_kwargs)

        return cls.from_pretrained(path, **kwargs)
