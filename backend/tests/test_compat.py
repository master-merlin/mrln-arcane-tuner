"""Tests for the diffusers / torchao compatibility patch.

Covers: patch detection, idempotency, and no-op when already fixed.

Note: ``apply_diffusers_patches()`` runs at ``app`` init time via
``app/__init__.py``, so by the time these tests execute the patch has
already been applied to the on-disk file.  The tests therefore validate
the *current state* of that file and re-invoke the patch to verify
idempotency.
"""

import importlib
import importlib.metadata
import importlib.util
from unittest.mock import patch



MODULE_TARGET = "diffusers.quantizers.torchao.torchao_quantizer"


class TestApplyDiffusersPatches:
    """Tests for ``app.core.compat.apply_diffusers_patches``."""

    @staticmethod
    def _read_target_source() -> str:
        """Return the current on-disk source of the target module."""
        spec = importlib.util.find_spec(MODULE_TARGET)
        assert spec is not None and spec.origin is not None
        with open(spec.origin, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_logger_defined_before_usage(self):
        """After the patch, ``logger`` must be defined before
        ``_update_torch_safe_globals()``."""
        source = self._read_target_source()
        func_pos = source.find("_update_torch_safe_globals()")
        logger_pos = source.find("\nlogger = logging.get_logger(__name__)\n")

        assert func_pos != -1, "Could not find _update_torch_safe_globals"
        assert logger_pos != -1, "Could not find logger definition"
        assert logger_pos < func_pos, (
            "logger must be defined BEFORE _update_torch_safe_globals()"
        )

    def test_patch_is_idempotent(self):
        """Re-running the patch should not duplicate logger or corrupt the file."""
        from app.core.compat import apply_diffusers_patches

        source_before = self._read_target_source()
        apply_diffusers_patches()
        source_after = self._read_target_source()

        assert source_before == source_after, "Patch is not idempotent"

    def test_single_logger_definition(self):
        """There should be exactly one ``logger = logging.get_logger(...)``."""
        source = self._read_target_source()
        count = source.count("logger = logging.get_logger(__name__)")
        assert count == 1, f"Expected 1 logger definition, found {count}"

    def test_noop_when_diffusers_not_installed(self):
        """If diffusers is not installed, the patch should silently return."""
        from app.core.compat import apply_diffusers_patches

        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            # Should not raise
            apply_diffusers_patches()

    def test_module_importable(self):
        """The target module should be importable after patching."""
        mod = importlib.import_module(MODULE_TARGET)
        assert hasattr(mod, "TorchAoHfQuantizer"), (
            "Module broken — TorchAoHfQuantizer not found"
        )
        assert hasattr(mod, "logger"), (
            "Module broken — logger not found"
        )
