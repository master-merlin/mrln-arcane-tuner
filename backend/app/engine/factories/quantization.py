"""
Quantization Factory — quantizes model components for efficient LoRA training.

Supports multiple backends (torchao, optimum-quanto, bitsandbytes) and bit-widths
via a pluggable strategy pattern.
"""
import json
import os
import time
from typing import Any

import structlog
import torch
import torch.nn as nn

from .quantization_base import QuantizationBase
from .quantizers.torchao import TorchAOBackend
from .quantizers.quanto import QuantoBackend
from .quantizers.bitsandbytes import BitsAndBytesBackend

logger = structlog.get_logger(__name__)

# Fallback chain: if the requested scheme is unavailable, try these in order.
_FALLBACK_MAP: dict[str, list[str]] = {
    "nvfp4":        ["nf4", "int4", "none"],
    "fp8":          ["int8", "nf4", "none"],
    "nf4":          ["int4", "none"],
    "int4":         ["nf4", "none"],
    "int5":         ["int4", "nf4", "none"],
    "int6":         ["int5", "int4", "nf4", "none"],
    "int7":         ["int6", "int5", "int4", "nf4", "none"],
    "int8":         ["fp8", "int7", "int6", "nf4", "none"],
    "qint4":        ["int4", "nf4", "none"],
    "qint8":        ["int8", "fp8", "nf4", "none"],
    "qfloat8":      ["fp8", "int8", "none"],
    "qfloat8_e4m3fn": ["fp8", "int8", "none"],
    "qfloat8_e5m2": ["fp8", "int8", "none"],
}

class QuantizationFactory:
    """Factory and registry for quantization backends."""

    _backends: dict[str, type[QuantizationBase]] = {
        "torchao": TorchAOBackend,
        "optimum-quanto": QuantoBackend,
        "bitsandbytes": BitsAndBytesBackend,
    }

    @classmethod
    def get_backend(cls, name: str) -> type[QuantizationBase]:
        """Retrieve a specific backend by name."""
        if name not in cls._backends:
            raise ValueError(f"Unknown quantization backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def _resolve_backend_and_scheme(cls, requested_backend: str, requested_scheme: str) -> tuple[str, str]:
        """Resolves 'auto' logic to find the best backend and scheme."""
        scheme = requested_scheme.lower().strip()
        backend_name = requested_backend.lower().strip()

        if scheme in ("none", "bf16"):
            return "auto", scheme

        # Specific backend requested
        if backend_name != "auto":
            backend = cls.get_backend(backend_name)
            if not backend.is_available(scheme):
                raise RuntimeError(f"Scheme '{scheme}' is not available on backend '{backend_name}'.")
            return backend_name, scheme

        # Auto resolution - try to find the first capable backend
        # Prioritize torchao for int4/int8/fp8/nvfp4, bitsandbytes for nf4, optimum-quanto as fallback
        for b_name in ["torchao", "bitsandbytes", "optimum-quanto"]:
            backend = cls.get_backend(b_name)
            if scheme in backend.supported_schemes() and backend.is_available(scheme):
                return b_name, scheme

        raise RuntimeError(f"No backend available for scheme '{scheme}'. Check hardware/dependencies.")

    @classmethod
    def validate_and_fallback(cls, scheme: str, backend_name: str = "auto") -> tuple[str, str]:
        """Validate and find an available backend/scheme combo, walking fallback map if necessary."""
        scheme = scheme.lower().strip()
        backend_name = backend_name.lower().strip()

        if scheme in ("none", "bf16"):
            return backend_name, scheme

        try:
            return cls._resolve_backend_and_scheme(backend_name, scheme)
        except RuntimeError:
            pass # Try fallbacks

        # Walk the fallback chain
        for fallback in _FALLBACK_MAP.get(scheme, ["none"]):
            try:
                b_name, res_scheme = cls._resolve_backend_and_scheme(backend_name, fallback)
                logger.warning(
                    "quantization_fallback",
                    requested_scheme=scheme,
                    fallback=res_scheme,
                    backend=b_name
                )
                return b_name, res_scheme
            except RuntimeError:
                pass

        logger.warning("quantization_no_fallback", requested_scheme=scheme)
        return "auto", "none"

    @classmethod
    def quantize(
        cls,
        module: nn.Module,
        scheme: str,
        backend_name: str = "auto",
        device: str | None = None,
    ) -> nn.Module:
        """
        Apply weight-only quantization to a frozen module using the strategy pattern.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if scheme in ("none", "bf16"):
            logger.info("quantization_skipped", scheme=scheme)
            return module

        resolved_backend, resolved_scheme = cls._resolve_backend_and_scheme(backend_name, scheme)
        backend = cls.get_backend(resolved_backend)

        param_count = sum(p.numel() for p in module.parameters())
        logger.info("quantizing_model", backend=resolved_backend, scheme=resolved_scheme, params=param_count)

        return backend.quantize(module, resolved_scheme, device=device)

    @classmethod
    def get_supported_capabilities(cls) -> dict[str, list[str]]:
        """Return a map of available backends to their supported, hardware-capable schemes."""
        capabilities = {}
        for name, backend in cls._backends.items():
            capable_schemes = [
                scheme for scheme in backend.supported_schemes()
                if backend.is_available(scheme)
            ]
            if capable_schemes:
                capabilities[name] = capable_schemes
        return capabilities

    @classmethod
    def estimate_vram(
        cls,
        module: nn.Module,
        scheme: str,
        backend_name: str = "auto"
    ) -> dict[str, Any]:
        """Estimate VRAM usage before and after quantization."""
        total_params = sum(p.numel() for p in module.parameters())

        # Current size (assume all params are the same dtype)
        sample_param = next(module.parameters(), None)
        if sample_param is None:
            return {"before_mb": 0, "after_mb": 0, "savings_pct": 0}

        bytes_per_param = sample_param.element_size()
        before_bytes = total_params * bytes_per_param

        if scheme in ("none", "bf16"):
            bits_after = bytes_per_param * 8
        else:
             try:
                 resolved_backend, resolved_scheme = cls._resolve_backend_and_scheme(backend_name, scheme)
                 backend = cls.get_backend(resolved_backend)
                 bits_after = backend.get_bits(resolved_scheme)
             except RuntimeError:
                 bits_after = bytes_per_param * 8

        after_bytes = total_params * bits_after / 8

        before_mb = round(before_bytes / (1024 * 1024), 1)
        after_mb = round(after_bytes / (1024 * 1024), 1)
        savings = round((1 - after_bytes / before_bytes) * 100, 1) if before_bytes > 0 else 0

        return {
            "before_mb": before_mb,
            "after_mb": after_mb,
            "savings_pct": savings,
        }

    # ── Disk Cache API (Preserved for compatibility) ──────────────────────
    @staticmethod
    def _get_cache_root() -> str:
        here = os.path.dirname(__file__)
        backend_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        return os.path.join(backend_root, "models", ".quantized")

    @staticmethod
    def resolve_cache_path(definition_id: str, component: str, scheme: str) -> str:
        root = QuantizationFactory._get_cache_root()
        safe_id = definition_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(root, safe_id, scheme, component)

    @staticmethod
    def _get_source_fingerprint(source_path: str | None) -> dict:
        if not source_path or not os.path.exists(source_path):
            return {}
        try:
            if os.path.isfile(source_path):
                stat = os.stat(source_path)
                return {"source_path": source_path, "source_mtime": stat.st_mtime, "source_size": stat.st_size}
            max_mtime = 0.0
            total_size = 0
            for entry in os.scandir(source_path):
                if entry.is_file() and entry.name.endswith((".safetensors", ".bin", ".pt")):
                    s = entry.stat()
                    max_mtime = max(max_mtime, s.st_mtime)
                    total_size += s.st_size
            if max_mtime > 0:
                return {"source_path": source_path, "source_mtime": max_mtime, "source_size": total_size}
        except OSError:
            pass
        return {}

    @staticmethod
    def save_quantized(
        module: nn.Module,
        cache_path: str,
        scheme: str,
        source_path: str | None = None,
    ) -> None:
        os.makedirs(cache_path, exist_ok=True)

        metadata = {
            "scheme": scheme,
            "torch_version": torch.__version__,
            "timestamp": time.time(),
            "param_count": sum(p.numel() for p in module.parameters()),
        }

        fingerprint = QuantizationFactory._get_source_fingerprint(source_path)
        if fingerprint:
            metadata.update(fingerprint)

        try:
            from safetensors.torch import save_file
            tensors = {k: v.contiguous() for k, v in module.state_dict().items()}
            save_file(tensors, os.path.join(cache_path, "model.safetensors"))
            metadata["format"] = "safetensors"
        except Exception as e:
            if not isinstance(e, ImportError):
                logger.info("safetensors_save_fallback", reason=str(e)[:200])
            torch.save(module.state_dict(), os.path.join(cache_path, "model.pt"))
            metadata["format"] = "torch"

        with open(os.path.join(cache_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "quantized_model_cached",
            path=cache_path,
            scheme=scheme,
            format=metadata["format"],
        )

    @staticmethod
    def load_quantized(
        module: nn.Module,
        cache_path: str,
        scheme: str,
        source_path: str | None = None,
    ) -> nn.Module | None:
        meta_path = os.path.join(cache_path, "metadata.json")
        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("quantized_cache_corrupt", path=cache_path)
            return None

        if metadata.get("scheme") != scheme:
            logger.info("quantized_cache_scheme_mismatch", path=cache_path, cached=metadata.get("scheme"), requested=scheme)
            return None

        param_count = sum(p.numel() for p in module.parameters())
        cached_count = metadata.get("param_count")
        if cached_count and cached_count != param_count:
             logger.warning("quantized_cache_param_mismatch", cached=cached_count, current=param_count)
             return None

        if source_path:
            fingerprint = QuantizationFactory._get_source_fingerprint(source_path)
            if fingerprint.get("source_mtime") != metadata.get("source_mtime") or \
               fingerprint.get("source_size") != metadata.get("source_size"):
                logger.info("quantized_cache_invalidated", reason="source weights changed")
                return None

        format = metadata.get("format", "torch")
        try:
            if format == "safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(cache_path, "model.safetensors"))
            else:
                state_dict = torch.load(os.path.join(cache_path, "model.pt"), weights_only=False)

            module.load_state_dict(state_dict, strict=False, assign=True)
            logger.info("quantized_model_loaded", path=cache_path, scheme=scheme)
            return module
        except Exception as e:
            logger.error("quantized_cache_load_error", error=str(e))
            return None
