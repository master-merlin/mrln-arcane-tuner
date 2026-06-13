"""Pure helpers for assembling paired-control inventory items.

Edit/kontext runs train the *effective target* of a pair while
conditioning on its *effective controls* (the resolved roles from the
``/pairs`` payload — role ordering is metadata, so a control slot can be
the target). These helpers compute the control resize dims, the
latent-cache variant (keyed by physical slot so role flips don't thrash
unrelated caches), and the per-item control fields — with no torch / I/O
so they're unit-testable in isolation.
"""

from __future__ import annotations

import os
from typing import Any, Callable

# Latent-cache variant for a root image that is acting as a control (role
# flip). Distinct from the target's "original" variant so the same file
# cached as target vs as control never collide.
CONTROL_VARIANT_ROOT = "root_ctl"


def control_variant(rel_path: str) -> str:
    """Latent-cache variant for a control image, keyed by its physical slot.

    ``control/img.jpg`` -> ``control``; ``control_2/img.png`` -> ``control_2``;
    a bare root image (role-flipped pair) -> :data:`CONTROL_VARIANT_ROOT`.
    Physical-slot keying means re-ordering a pair only re-keys the two files
    whose roles swapped, never the rest of the cache.
    """
    rel = rel_path.replace("\\", "/")
    return rel.split("/", 1)[0] if "/" in rel else CONTROL_VARIANT_ROOT


def control_target_dims(
    target_w: int,
    target_h: int,
    control_resolution: int,
    bucket_for: Callable[[int, int, int], dict] | None = None,
) -> tuple[int, int]:
    """Dims to resize control images to for one batch item.

    ``control_resolution == 0`` (default) → controls follow the target's
    bucket, so a Kontext batch packs control + target on the same grid.
    Nonzero → re-bucket the *target's aspect* at the control base resolution
    (Qwen-Edit's fixed 1024 convention) — keying off the target aspect keeps
    every control in a bucket-homogeneous batch the same size (stackable).
    """
    if not control_resolution:
        return target_w, target_h
    if bucket_for is None:
        return control_resolution, control_resolution
    b = bucket_for(target_w, target_h, control_resolution)
    return int(b["width"]), int(b["height"])


def build_control_fields(
    effective_controls: list[str],
    ds_path: str,
    control_inputs: int,
    target_w: int,
    target_h: int,
    control_resolution: int,
    cache_dir_for: Callable[[str, str], str],
    bucket_for: Callable[[int, int, int], dict] | None = None,
) -> dict[str, Any] | None:
    """Assemble per-item control inventory fields, or ``None`` for a partial pair.

    A partial pair (fewer resolved controls than the model needs) returns
    ``None`` so the caller can skip + count it. ``cache_dir_for(res_str,
    variant)`` resolves a latent-cache directory; ``bucket_for(w, h, base)``
    returns a bucket dict (only used when ``control_resolution > 0``).
    """
    controls = list(effective_controls or [])
    if len(controls) < control_inputs:
        return None
    controls = controls[:control_inputs]

    cw, ch = control_target_dims(target_w, target_h, control_resolution, bucket_for)
    res_str = f"{cw}x{ch}"

    paths: list[str] = []
    dims: list[tuple[int, int]] = []
    variants: list[str] = []
    cache_dirs: list[str] = []
    for rel in controls:
        variant = control_variant(rel)
        paths.append(os.path.join(ds_path, rel))
        dims.append((cw, ch))
        variants.append(variant)
        cache_dirs.append(cache_dir_for(res_str, variant))

    return {
        "control_rel_paths": controls,
        "control_paths": paths,
        "control_dims": dims,
        "control_variants": variants,
        "control_cache_dirs": cache_dirs,
    }
