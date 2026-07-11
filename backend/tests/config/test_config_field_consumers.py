"""W4-1 dead-key regression guards for ``BaseTrainingConfig``.

Two static invariants keep the training-config surface honest. The config
pipeline is ``dict[str, Any]`` end-to-end and the frontend renders whatever the
backend-served schema declares, so a field can rot in three silent ways that no
runtime test catches:

* it loses its last consumer and becomes dead weight in the schema/UI, or
* it never had one (a forward-compat stub declared "just in case"), or
* it is added without any capability gating and therefore silently reaches
  *every* family (the opt-in-gating hole).

**Consumer guard** — every field must be *read* somewhere in ``backend/app``
(attribute access ``.field`` or dict-key access ``"field"`` / ``'field'``),
excluding the model definition itself and the gating/visibility plumbing
(``archetypes.py``), OR be listed in :data:`ALLOWED_STUBS` with a one-line
justification. All five W4-1 dead fields (``quantization_strategy``,
``resolution_strategy``, ``boundary_ratio_override``, ``still_resolutions`` and
``radc_seqlen_influence``) were deleted, so they must NOT be needed here. The
fifth, ``radc_seqlen_influence``, was initially retained for a self-referential
``video_contract`` guard (it only read the field to reject it unless the RADC
sampler was selected — a Phase-3 stub that never consumes the value); the fix
wave recognised that guard as circular and deleted the field. It will be
re-added with Phase 3.

**Gating-coverage guard** — every field must either be gated in
``_FIELD_RULES`` OR be explicitly enrolled in :data:`UNIVERSAL_FIELDS`. A new
field that is neither trips the guard, forcing a deliberate classification
instead of silently defaulting to "shown for all families".

Both guards are pure source scans — no GPU, no registry, no model instantiation
beyond reading ``model_fields``.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.engine.core.archetypes import _FIELD_RULES
from app.engine.models.base import BaseTrainingConfig, DatasetItem

# ── Source-scan scaffolding ──────────────────────────────────────────────
# This file lives at backend/tests/config/ → parents[2] is the backend root.
_BACKEND = Path(__file__).resolve().parents[2]
_APP = (_BACKEND / "app").resolve()

# Reading a field's *value* here does not make the file a real consumer:
#  - base.py     — the model definition (the field lives here)
#  - archetypes.py — _FIELD_RULES gating + build_field_visibility plumbing
_EXCLUDED_FILES = {
    (_APP / "engine" / "models" / "base.py").resolve(),
    (_APP / "engine" / "core" / "archetypes.py").resolve(),
}


def _consumer_texts() -> dict[Path, str]:
    """Every ``backend/app`` python source that may legitimately consume a
    config field, keyed by resolved path."""
    out: dict[Path, str] = {}
    for p in _APP.rglob("*.py"):
        rp = p.resolve()
        if rp in _EXCLUDED_FILES:
            continue
        if "tests" in rp.parts:  # defensive; app currently ships no tests dir
            continue
        out[rp] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def _has_consumer(field: str, texts: dict[Path, str]) -> bool:
    """True iff ``field`` is referenced as a quoted key (``"field"`` /
    ``'field'``) or an attribute access (``.field``) anywhere in ``texts``.

    Covers the three real access shapes in this codebase: ``config.get("f")``,
    ``config["f"]`` and ``cfg.f``. The word boundary on the attribute form
    prevents ``.foo`` from matching ``.foobar``.
    """
    pat = re.compile(
        r'["\']' + re.escape(field) + r'["\']'  # "field" or 'field'
        r'|\.' + re.escape(field) + r'\b'  # .field
    )
    return any(pat.search(t) for t in texts.values())


# ── ALLOWED_STUBS: fields with no backend/app consumer, retained on purpose ──
# Each entry needs a one-line justification. A dead field must NOT hide here;
# the five W4-1 deletions were removed from the model, not parked here.
ALLOWED_STUBS: dict[str, str] = {
    "lora_prefix": (
        "frontend-only: lora-naming UI input, composed into lora_name (which "
        "the backend consumes); backend never reads it directly"
    ),
    "lora_suffix": (
        "frontend-only: lora-naming UI input, composed into lora_name (which "
        "the backend consumes); backend never reads it directly"
    ),
    "i2v_image_dropout": (
        "declared + gated(is_video) + tested i2v conditioning-dropout stub; no "
        "runtime consumer wired yet (video-training backlog)"
    ),
    "audio_loss_weight": (
        "declared + gated(has_audio) + tested audio loss-weight stub; no "
        "runtime consumer wired yet (video/audio-training backlog)"
    ),
}


def test_every_field_has_a_consumer_or_is_an_allowed_stub():
    texts = _consumer_texts()
    dead = sorted(
        f
        for f in BaseTrainingConfig.model_fields
        if f not in ALLOWED_STUBS and not _has_consumer(f, texts)
    )
    assert not dead, (
        "BaseTrainingConfig fields with NO consumer in backend/app and not "
        f"declared in ALLOWED_STUBS (dead schema/UI weight): {dead}. Either "
        "wire a consumer, delete the field, or (if intentionally inert) add it "
        "to ALLOWED_STUBS with a justification."
    )


def test_allowed_stubs_are_current_fields_and_still_stubs():
    """ALLOWED_STUBS must not go stale: every entry names a real field, carries
    a justification, and has NOT quietly gained a consumer (in which case it
    should graduate out of the stub list)."""
    texts = _consumer_texts()
    fields = set(BaseTrainingConfig.model_fields)
    stale = sorted(f for f in ALLOWED_STUBS if f not in fields)
    assert not stale, f"ALLOWED_STUBS names fields that no longer exist: {stale}"
    for field, why in ALLOWED_STUBS.items():
        assert why.strip(), f"ALLOWED_STUBS[{field!r}] needs a justification"
        assert not _has_consumer(field, texts), (
            f"{field} now HAS a real consumer in backend/app — remove it from "
            "ALLOWED_STUBS so the consumer guard covers it."
        )


def test_consumer_detector_has_no_false_positive_on_a_synthetic_field():
    """Durable red-first proof: a field name that appears nowhere in the tree
    is reported as having no consumer (the guard's negative branch works)."""
    texts = _consumer_texts()
    assert not _has_consumer("zzz_synthetic_field_that_appears_nowhere", texts)


# ── Gating-coverage guard ────────────────────────────────────────────────
_GATED_FIELDS = frozenset(name for (name, _flag, _reason) in _FIELD_RULES)

# Fields intentionally shown for EVERY family (no capability gate). Enrolling a
# field here is a deliberate act: it documents that the field is universal.
UNIVERSAL_FIELDS = frozenset(
    {
        "lora_prefix",
        "lora_suffix",
        "lora_name",
        "global_triggerword",
        "mixed_precision",
        "save_precision",
        "model_family",
        "definition_id",
        "quantization_backend",
        "quantization",
        "store_quantized_version",
        "output_dir",
        "datasets",
        "max_train_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "save_every_n_steps",
        "keep_last_checkpoints",
        "persist_latents",
        "persist_embeddings",
        "resume_from_checkpoint",
        "use_cached_latents",
        "use_cached_embeddings",
        "resolutions",
        "bucketing_mode",
        "timestep_sampling",
        "logit_normal_mu",
        "logit_normal_sigma",
        "model_shift_std",
        "timestep_uniform_prob",
        "mode_scale",
        "flux_shift_base",
        "flux_shift_max",
        "radc_start",
        "radc_end",
        "radc_width",
        "radc_res_influence",
        "network_rank",
        "network_alpha",
        "optimizer_type",
        "learning_rate",
        "weight_decay",
        "lr_scale_mode",
        "beta1",
        "beta2",
        "d_coef",
        "growth_rate",
        "decouple",
        "safeguard_warmup",
        "use_bias_correction",
        "lr_warmup_steps",
        "lr_scheduler",
        "ppsf_d_coef",
        "ppsf_prodigy_steps",
        "ppsf_use_bias_correction",
        "ppsf_use_stableadamw",
        "ppsf_factored",
        "ppsf_eps",
        "ppsf_use_cautious",
        "ppsf_use_grams",
        "ppsf_use_adopt",
        "ppsf_use_orthograd",
        "ppsf_use_focus",
        "ppsf_use_speed",
        "ppsf_split_groups",
        "sophia_rho",
        "sophia_p",
        "sophia_update_period",
        "sophia_num_samples",
        "sophia_hessian_distribution",
        "sophia_maximize",
        "sophia_capturable",
        "adafactor_relative_step",
        "adafactor_warmup_init",
        "adafactor_clip_threshold",
        "adafactor_decay_rate",
        "radam_n_sma_threshold",
        "shampoo_preconditioning_compute_steps",
        "stableadamw_kahan_sum",
        "ademamix_beta3",
        "ademamix_alpha",
        "ema",
        "ema_decay",
        "noise_offset",
        "min_snr_gamma",
        "offload_to_cpu",
        "vram_safe_bucket_order",
        "targeted_layers",
        "sample_every_n_steps",
        "sample_skip_first_n_steps",
        "sample_prompts",
        "sampling_min_free_vram_fraction",
    }
)


def test_every_field_is_gated_or_declared_universal():
    """The opt-in-gating hole: a new field that is neither gated in
    ``_FIELD_RULES`` nor enrolled in ``UNIVERSAL_FIELDS`` silently reaches every
    family. Force a deliberate choice."""
    unclassified = sorted(
        f
        for f in BaseTrainingConfig.model_fields
        if f not in _GATED_FIELDS and f not in UNIVERSAL_FIELDS
    )
    assert not unclassified, (
        "BaseTrainingConfig fields that are neither gated in _FIELD_RULES nor "
        f"declared in UNIVERSAL_FIELDS: {unclassified}. Add a _FIELD_RULES gate "
        "(family-specific) or enroll it in UNIVERSAL_FIELDS (shown everywhere)."
    )


def test_no_field_is_both_gated_and_universal():
    overlap = sorted(_GATED_FIELDS & UNIVERSAL_FIELDS)
    assert not overlap, f"fields both gated and marked universal: {overlap}"


def test_universal_and_gated_markers_name_real_fields():
    base_fields = set(BaseTrainingConfig.model_fields)
    # _FIELD_RULES legitimately gates fields on BOTH the per-run config and the
    # per-dataset item (e.g. masking_enabled, h_flip live on DatasetItem).
    gateable_fields = base_fields | set(DatasetItem.model_fields)
    stale_universal = sorted(UNIVERSAL_FIELDS - base_fields)
    stale_gated = sorted(_GATED_FIELDS - gateable_fields)
    assert not stale_universal, f"stale UNIVERSAL_FIELDS entries: {stale_universal}"
    assert not stale_gated, f"stale _FIELD_RULES entries: {stale_gated}"


def test_gating_guard_rejects_a_synthetic_unclassified_field():
    """Durable red-first proof: a hypothetical new field lands in neither the
    gated set nor the universal set, so the classification check would flag it."""
    fake = "brand_new_totally_ungated_field"
    assert fake not in _GATED_FIELDS
    assert fake not in UNIVERSAL_FIELDS
