"""MRLN Arcane Tuner — root application package."""

# ── Third-party compatibility patches ──────────────────────────────────
# Must run before any `from diffusers import …` in the codebase.
from app.core.compat import apply_diffusers_patches, apply_hpsv2_patches  # noqa: E402

apply_diffusers_patches()
apply_hpsv2_patches()

__version__ = "0.6.5-beta"
