"""Metadata every saver must write identically, in one place.

Each family saver builds its own ``ss_*`` map. That is fine for the keys a
family genuinely owns, but a value with ONE correct spelling across all of
them drifts the moment a saver is copied — ltx2 shipped without the trigger
word at all because its copy predated that line. Anything in here is imported,
never re-typed.
"""

from __future__ import annotations

from typing import Any

#: Kohya's conventional home for the trigger word. CivitAI and most LoRA
#: browsers read this one, so it stays even though it is a free-text comment
#: field being used to carry a specific value.
TRIGGER_COMMENT_KEY = "ss_training_comment"

#: The field the Stability ModelSpec defines for exactly this, mirrored by
#: kohya's own ``library/sai_model_spec.py``. A tool that follows the spec
#: looks here and finds nothing if only the comment is written.
TRIGGER_PHRASE_KEY = "modelspec.trigger_phrase"


def trigger_metadata(config: Any) -> dict[str, str]:
    """Both keys a downstream tool might read the trigger word from.

    Returns an EMPTY dict when the run has no trigger word. Writing an empty
    string would advertise a trigger phrase that does not exist, and a reader
    cannot tell "" apart from "the author left it blank on purpose" — an absent
    key is unambiguous.
    """
    if not isinstance(config, dict):
        return {}
    trigger = config.get("global_triggerword")
    if trigger is None:
        return {}
    trigger = str(trigger).strip()
    if not trigger:
        return {}
    return {TRIGGER_COMMENT_KEY: trigger, TRIGGER_PHRASE_KEY: trigger}


#: Training-config → Kohya ``ss_*`` keys, the map ``GenericLoRASaver.save``
#: (``core/pipeline/saver_base.py``) carries inline. Shared here so a family
#: saver that cannot subclass the generic one (minimax_h3 fuses and renames
#: its tensors) writes the SAME keys; the string-valued half is skipped when
#: blank, the numeric half whenever present.
KOHYA_STR_KEYS: dict[str, str] = {
    "optimizer_type": "ss_optimizer",
    "lr_scheduler": "ss_lr_scheduler",
    "mixed_precision": "ss_mixed_precision",
    "lora_name": "ss_output_name",
    "definition_id": "ss_sd_model_name",
    "model_family": "ss_base_model_version",
    "timestep_sampling": "ss_timestep_sampling",
}
KOHYA_NUM_KEYS: dict[str, str] = {
    "learning_rate": "ss_learning_rate",
    "max_train_steps": "ss_steps",
    "train_batch_size": "ss_batch_size_per_device",
    "gradient_accumulation_steps": "ss_gradient_accumulation_steps",
    "noise_offset": "ss_noise_offset",
    "min_snr_gamma": "ss_min_snr_gamma",
    "lr_warmup_steps": "ss_warmup_steps",
    "weight_decay": "ss_weight_decay",
    "seed": "ss_seed",
}


def kohya_config_metadata(
    config: Any,
    *,
    rank: int | None = None,
    alpha: float | None = None,
) -> dict[str, str]:
    """The Kohya ``ss_*`` header fields for a run: the config map above, the
    network dim / alpha when given, the first resolution as ``"(W,H)"`` and the
    dataset repeat count — every value stringified the way safetensors wants."""
    out: dict[str, str] = {}
    if rank is not None:
        out["ss_network_dim"] = str(int(rank))
    if alpha is not None:
        out["ss_network_alpha"] = str(float(alpha))
    if not isinstance(config, dict) or not config:
        return out
    for cfg_key, ss_key in KOHYA_STR_KEYS.items():
        val = config.get(cfg_key)
        if val is not None and str(val).strip():
            out[ss_key] = str(val)
    for cfg_key, ss_key in KOHYA_NUM_KEYS.items():
        val = config.get(cfg_key)
        if val is not None:
            out[ss_key] = str(val)
    resolutions = config.get("resolutions")
    if resolutions and isinstance(resolutions, list):
        first = resolutions[0]
        out["ss_resolution"] = f"({first},{first})"
    datasets = config.get("datasets")
    if datasets and isinstance(datasets, list):
        out["ss_num_train_images"] = str(
            sum(d.get("num_repeats", 1) for d in datasets if isinstance(d, dict))
        )
    return out
