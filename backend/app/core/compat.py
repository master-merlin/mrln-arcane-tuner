"""Compatibility patches for third-party library bugs.

Applied once at package init (``app/__init__.py``) **before** any
``from diffusers import …`` executes, so the patch is in place by the
time module-level code runs.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import structlog

_logger = structlog.get_logger(__name__)

# Sentinel: track whether the on-disk patch was applied this session.
_PATCH_APPLIED = False


def apply_diffusers_patches() -> None:
    """Fix ``diffusers.quantizers.torchao.torchao_quantizer`` logger bug.

    diffusers ≤ 0.36.0 calls ``_update_torch_safe_globals()`` at module
    scope *before* ``logger`` is defined.  When ``torchao ≥ 0.16`` removes
    ``torchao.dtypes.uintx.uint4_layout``, the resulting ``ImportError``
    triggers an ``except`` block that references the undefined ``logger``,
    crashing the entire import chain.

    This patch edits the installed source file **on disk** (idempotently)
    so that ``logger`` is defined before it is referenced.  The patch is
    safe to re-apply and becomes a no-op once the file is already fixed
    or diffusers ships an upstream fix.
    """
    global _PATCH_APPLIED  # noqa: PLW0603

    try:
        diffusers_version = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return  # diffusers not installed — nothing to patch

    # Derive the file path from the package location without importing
    # any diffusers submodule (which would trigger the crash).
    try:
        dist = importlib.metadata.distribution("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return

    # dist.locate_file gives us a path relative to the package install root
    pkg_root = Path(str(dist.locate_file("")))
    target_file = (
        pkg_root
        / "diffusers"
        / "quantizers"
        / "torchao"
        / "torchao_quantizer.py"
    )

    if not target_file.is_file():
        return

    _apply_logger_fix(str(target_file), diffusers_version)


def _apply_logger_fix(filepath: str, diffusers_version: str) -> None:
    """Idempotently move ``logger`` above ``_update_torch_safe_globals``."""
    global _PATCH_APPLIED  # noqa: PLW0603

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return

    # ── Detect whether the bug is present ───────────────────────────────
    marker_func = "_update_torch_safe_globals()"
    marker_logger = "\nlogger = logging.get_logger(__name__)\n"

    func_pos = source.find(marker_func)
    logger_pos = source.find(marker_logger)

    if func_pos == -1 or logger_pos == -1:
        return  # Module structure changed — bail
    if logger_pos < func_pos:
        return  # Logger is already before the function call — no-op

    # ── Patch the source on disk ────────────────────────────────────────
    # 1. Remove logger from its current (too-late) position.
    patched = source.replace(marker_logger, "\n", 1)

    # 2. Insert it just before `def _update_torch_safe_globals():`.
    anchor = "\ndef _update_torch_safe_globals():"
    anchor_pos = patched.find(anchor)
    if anchor_pos == -1:
        return

    patched = (
        patched[:anchor_pos]
        + "\n\nlogger = logging.get_logger(__name__)\n"
        + patched[anchor_pos:]
    )

    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(patched)
        _PATCH_APPLIED = True
        _logger.info(
            "diffusers_torchao_patch_applied",
            diffusers_version=diffusers_version,
            file=filepath,
        )
    except OSError as exc:
        _logger.warning("diffusers_patch_write_failed", error=str(exc))


def apply_hpsv2_patches() -> None:
    """Download missing BPE vocabulary file for hpsv2's vendored open_clip.

    The ``hpsv2`` pip package bundles its own ``open_clip`` copy but fails
    to include ``bpe_simple_vocab_16e6.txt.gz`` in the wheel.  Without this
    file, tokenization crashes on first use.

    This patch downloads the file from OpenAI's CLIP repository (idempotent).
    """
    try:
        importlib.metadata.version("hpsv2")
    except importlib.metadata.PackageNotFoundError:
        return  # hpsv2 not installed — nothing to patch

    try:
        dist = importlib.metadata.distribution("hpsv2")
    except importlib.metadata.PackageNotFoundError:
        return

    pkg_root = Path(str(dist.locate_file("")))
    bpe_path = pkg_root / "hpsv2" / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"

    if bpe_path.is_file():
        return  # Already present — no-op

    bpe_url = (
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    )

    import urllib.request

    try:
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(bpe_url, str(bpe_path))
        _logger.info("hpsv2_bpe_vocab_downloaded", bpe_path=str(bpe_path))
    except OSError as exc:
        _logger.warning("hpsv2_bpe_vocab_download_failed", error=str(exc))
