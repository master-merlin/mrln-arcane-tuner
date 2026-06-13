"""Archetype-based capability + defaults templates for model families.

A model family declares an `archetype` (and optional `capability_overrides`).
The archetype is the generic template; per-model YAML `defaults` and per-family
overrides layer on top. This is the single machine-readable source the Training
UI reads to decide which config fields to show and what defaults to pre-fill.
Additive only — it changes nothing about how jobs actually train.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Archetype:
    id: str
    has_vae: bool
    has_external_te: bool
    latent_cache: bool
    te_cache: bool
    supports_train_te: bool
    supports_te_quantization: bool
    supports_block_swap: bool
    config_defaults: dict = field(default_factory=dict)


LATENT_DIFFUSION = Archetype(
    id="latent_diffusion",
    has_vae=True,
    has_external_te=True,
    latent_cache=True,
    te_cache=True,
    supports_train_te=False,  # SDXL overrides to True via capability_overrides
    supports_te_quantization=True,
    supports_block_swap=True,
    config_defaults={"resolution": 1024, "scheduler": "euler_a", "learning_rate": 1e-5},
)

UNIFIED_TRANSFORMER = Archetype(
    id="unified_transformer",
    has_vae=False,
    has_external_te=False,
    latent_cache=False,
    te_cache=False,
    supports_train_te=False,
    supports_te_quantization=False,
    supports_block_swap=False,
    config_defaults={"resolution": 1024, "scheduler": "euler_a", "learning_rate": 5e-6},
)

ARCHETYPES: dict[str, Archetype] = {
    a.id: a for a in (LATENT_DIFFUSION, UNIFIED_TRANSFORMER)
}

# Maps a config field -> the capability flag that gates it + a human reason.
_FIELD_RULES: list[tuple[str, str, str]] = [
    ("cache_latents", "has_vae", "pixel-space model — no VAE/latents to cache"),
    ("low_vram", "has_vae", "no VAE to offload"),
    (
        "cache_text_embeddings",
        "te_cache",
        "text encoding is part of the unified forward pass",
    ),
    ("unload_text_encoder", "has_external_te", "no standalone text encoder"),
    (
        "train_text_encoder",
        "supports_train_te",
        "no trainable text encoder for this family",
    ),
    ("te_quantization", "supports_te_quantization", "no standalone text encoder"),
    (
        "te_quantization_backend",
        "supports_te_quantization",
        "no standalone text encoder",
    ),
    (
        "block_swap_config",
        "supports_block_swap",
        "no block topology declared for this family",
    ),
    # Paired edit models (control_inputs > 0): geometric augmentation breaks
    # control/target pixel correspondence, masked variants are mutually
    # exclusive with paired training, and control_resolution only applies here.
    (
        "h_flip",
        "supports_augmentation",
        "paired edit training — flip augmentation breaks control/target correspondence",
    ),
    (
        "v_flip",
        "supports_augmentation",
        "paired edit training — flip augmentation breaks control/target correspondence",
    ),
    (
        "masking_enabled",
        "supports_masking_variants",
        "masked variants are mutually exclusive with paired edit training",
    ),
    (
        "control_resolution",
        "is_edit",
        "control resolution only applies to paired edit models",
    ),
]


def build_field_visibility(caps) -> dict[str, dict]:
    """caps: an Archetype OR a merged-capabilities dict. Returns
    {field: {"supported": bool, "reason"?: str}} for every gated field."""
    flags = caps if isinstance(caps, dict) else asdict(caps)
    out: dict[str, dict] = {}
    for field_name, flag, reason in _FIELD_RULES:
        supported = bool(flags.get(flag, True))
        entry: dict = {"supported": supported}
        if not supported:
            entry["reason"] = reason
        out[field_name] = entry
    return out


def _coerce_number(value):
    """Coerce a numeric-looking string default to int/float.

    PyYAML 1.1 keeps unsigned-exponent scalars like ``1e-4`` as *strings*
    (it requires ``1.0e-4`` to parse a float).  The UI descriptor must
    expose ``learning_rate`` etc. as real numbers, so normalize here
    without mutating the source YAML.
    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_capabilities(definition) -> dict:
    """Merge archetype template -> per-model YAML defaults -> family capability
    overrides into the descriptor the Training UI consumes. Additive/read-only."""
    from app.engine.models.registry import registry

    family_cls = registry.get_family_class(definition.family)
    arch = ARCHETYPES[family_cls.archetype]
    caps = {k: v for k, v in asdict(arch).items() if k not in ("id", "config_defaults")}
    caps.update(getattr(family_cls, "capability_overrides", {}))

    # Paired edit conditioning (per-definition, not per-archetype). Surface
    # control_inputs + an ``is_edit`` flag and the derived gates the field
    # rules consume (augmentation/masking are disabled for edit models).
    control_inputs = int(getattr(definition, "control_inputs", 0) or 0)
    is_edit = control_inputs > 0
    caps["control_inputs"] = control_inputs
    caps["is_edit"] = is_edit
    caps["supports_augmentation"] = not is_edit
    caps["supports_masking_variants"] = not is_edit

    merged = {**arch.config_defaults, **(definition.defaults or {})}
    defaults = {k: _coerce_number(v) for k, v in merged.items()}
    return {
        "archetype": arch.id,
        "capabilities": caps,
        "field_visibility": build_field_visibility(caps),
        "defaults": defaults,
    }
