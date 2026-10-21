"""LANE-168: the ONE visibility source gates the engine fields a family does not apply.

``core/archetypes.py`` ``_FIELD_RULES`` -> ``field_visibility`` is what the training form reads
(``model-capabilities.service.ts``) and what the submit allowlist (``config_allowlist.py``) drops by.
Before this lane no rule named ``min_snr_gamma`` (applied by SDXL alone), ``noise_offset`` (ignored by
hidream_o1's own patch-space noise) or the flow-matching timestep fields (SDXL draws a discrete uniform
timestep), so each was shown -- and written into the LoRA metadata -- for families that never used it.

Spec scenarios 8-11. Real registry, real shipped definitions, real savers writing real files.
"""

from __future__ import annotations

import pytest
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch import nn

from app.engine.core.archetypes import FLOW_TIMESTEP_FIELDS, _FIELD_RULES, resolve_capabilities
from app.engine.core.config_allowlist import apply_capability_allowlist
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.utils.lora_metadata import kohya_config_metadata


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


def _definitions():
    registry.initialize()
    defs = list(registry._definitions.values())
    assert len(defs) >= 32, f"only {len(defs)} definitions -- the registry did not load"
    return defs


def _visibility(defn) -> dict:
    return resolve_capabilities(defn)["field_visibility"]


# ── FLOW_TIMESTEP_FIELDS is derived from the schema, and the rules cover exactly it ──


def test_flow_timestep_fields_are_every_timestep_field_of_the_schema():
    expected = {
        name for name in BaseTrainingConfig.model_fields
        if name in {"timestep_sampling", "timestep_uniform_prob", "mode_scale"}
        or name.startswith(("radc_", "flux_shift_", "model_shift_", "logit_normal_"))
    }
    assert set(FLOW_TIMESTEP_FIELDS) == expected
    # the ones the spec names by hand are all there (a derivation that found none would be vacuous)
    for name in ("timestep_sampling", "timestep_uniform_prob", "mode_scale", "logit_normal_mu",
                 "logit_normal_sigma", "model_shift_std", "flux_shift_base", "flux_shift_max",
                 "radc_start", "radc_end", "radc_width", "radc_res_influence"):
        assert name in FLOW_TIMESTEP_FIELDS, name


def test_the_rules_gate_exactly_the_flow_timestep_fields_by_the_one_flag():
    gated = {f for f, flag, _ in _FIELD_RULES if flag == "supports_flow_timestep_sampling"}
    assert gated == set(FLOW_TIMESTEP_FIELDS)
    assert len([f for f, _, _ in _FIELD_RULES]) == len({f for f, _, _ in _FIELD_RULES}), "a field ruled twice"


# ── Scenario 10: the flow-matching timestep fields are hidden for sdxl only ──


def test_flow_timestep_fields_hidden_for_sdxl_and_dropped_by_the_allowlist():
    sdxl = [d for d in _definitions() if d.family == "sdxl"]
    assert len(sdxl) == 3, [d.id for d in sdxl]
    for defn in sdxl:
        vis = _visibility(defn)
        for name in FLOW_TIMESTEP_FIELDS:
            assert vis[name]["supported"] is False, f"{defn.id} {name}"
            assert "discrete uniform" in vis[name]["reason"], f"{defn.id} {name}: {vis[name]['reason']}"
        config = {name: BaseTrainingConfig.model_fields[name].default for name in FLOW_TIMESTEP_FIELDS}
        config["learning_rate"] = 1e-4
        dropped = apply_capability_allowlist(config, defn)
        assert set(dropped) == set(FLOW_TIMESTEP_FIELDS), defn.id
        assert config == {"learning_rate": 1e-4}, defn.id


def test_flow_timestep_fields_shown_for_every_other_definition():
    for defn in _definitions():
        if defn.family == "sdxl":
            continue
        vis = _visibility(defn)
        for name in FLOW_TIMESTEP_FIELDS:
            assert vis[name] == {"supported": True}, f"{defn.family}/{defn.id} {name}: {vis[name]}"
        config = {"timestep_sampling": "radc", "radc_start": 0.8}
        assert apply_capability_allowlist(config, defn) == [], defn.id
        assert config == {"timestep_sampling": "radc", "radc_start": 0.8}


# ── Scenario 9: min_snr_gamma is SDXL-only ──


def test_min_snr_gamma_shown_for_sdxl_hidden_with_an_sdxl_reason_elsewhere():
    for defn in _definitions():
        rule = _visibility(defn)["min_snr_gamma"]
        if defn.family == "sdxl":
            assert rule == {"supported": True}, defn.id
        else:
            assert rule["supported"] is False, f"{defn.family}/{defn.id}"
            assert "SDXL" in rule["reason"], f"{defn.family}/{defn.id}: {rule['reason']}"


def test_allowlist_drops_min_snr_gamma_for_flux2_and_keeps_it_for_sdxl():
    flux2 = next(d for d in _definitions() if d.family == "flux2")
    config = {"min_snr_gamma": 7.0, "learning_rate": 1e-4}
    assert apply_capability_allowlist(config, flux2) == ["min_snr_gamma"]
    assert config == {"learning_rate": 1e-4}
    sdxl = registry.get_definition("sdxl_base_1.0")
    config = {"min_snr_gamma": 7.0}
    assert apply_capability_allowlist(config, sdxl) == []
    assert config == {"min_snr_gamma": 7.0}


# ── Scenario 8: noise_offset hidden for hidream_o1, minimax_h3 keeps its refusal ──


def test_noise_offset_hidden_and_dropped_for_hidream_o1_only():
    seen_hidream = 0
    for defn in _definitions():
        rule = _visibility(defn)["noise_offset"]
        config = {"noise_offset": 0.1}
        dropped = apply_capability_allowlist(config, defn)
        if defn.family == "hidream_o1":
            seen_hidream += 1
            assert rule["supported"] is False, defn.id
            assert "patch-space noise" in rule["reason"], rule["reason"]
            assert dropped == ["noise_offset"] and config == {}, defn.id
        else:
            # minimax_h3 included: its value is REFUSED by its own validation, never silently dropped
            assert rule == {"supported": True}, f"{defn.family}/{defn.id}: {rule}"
            assert dropped == [] and config == {"noise_offset": 0.1}, defn.id
    assert seen_hidream >= 1


def _minimax_trainer(**config):
    defn = next(d for d in _definitions() if d.family == "minimax_h3")
    family = object.__new__(registry.get_family_class(defn.family))
    family.definition = defn
    family.config = {}
    trainer = object.__new__(family.get_trainer_class())
    trainer.definition = defn
    trainer.config = {"mixed_precision": "bf16", "te_quantization": "none", **config}
    return trainer


def test_minimax_h3_still_refuses_a_noise_offset():
    _minimax_trainer(noise_offset=0.0)._refuse_unproven_settings()  # the negative: no offset, no refusal
    with pytest.raises(ValueError, match="noise_offset"):
        _minimax_trainer(noise_offset=0.1)._refuse_unproven_settings()


# ── Scenario 11: ss_min_snr_gamma only when the run carried (applied) min_snr_gamma ──


def _flux2_config_after_allowlist() -> dict:
    flux2 = next(d for d in _definitions() if d.family == "flux2")
    config = {"min_snr_gamma": 5.0, "noise_offset": 0.05, "learning_rate": 1e-4, "lora_name": "x"}
    apply_capability_allowlist(config, flux2)
    return config


def test_shared_metadata_helper_writes_min_snr_only_when_carried():
    assert "ss_min_snr_gamma" not in kohya_config_metadata(_flux2_config_after_allowlist())
    sdxl = kohya_config_metadata({"min_snr_gamma": 5.0, "learning_rate": 1e-4})
    assert sdxl["ss_min_snr_gamma"] == "5.0"


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)

    def forward(self, x):
        return self.to_k(self.to_q(x))


def _peft_model() -> nn.Module:
    return get_peft_model(_Tiny(), LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k"]))


def _hidream_model() -> nn.Module:
    from app.engine.models.families.hidream_o1.lora_wrapper import inject_lora_layers

    class Mini(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8))

    m = Mini()
    inject_lora_layers(m, rank=4, alpha=8.0)
    return m


def _savers():
    from app.engine.core.pipeline.saver_base import GenericLoRASaver
    from app.engine.models.families.flux2.saver import Flux2Saver
    from app.engine.models.families.hidream_o1.saver import HiDreamO1Saver
    from app.engine.models.families.microsoft_lens.saver import MicrosoftLensSaver
    from app.engine.models.families.sdxl.saver import SDXLSaver

    return {
        "generic": (GenericLoRASaver, _peft_model),
        "flux2": (Flux2Saver, _peft_model),
        "hidream_o1": (HiDreamO1Saver, _hidream_model),
        "microsoft_lens": (MicrosoftLensSaver, _peft_model),
        "sdxl": (SDXLSaver, _peft_model),
    }


def _saved_header(saver_cls, make_model, config, tmp_path) -> dict:
    path = tmp_path / "lora.safetensors"
    saver_cls().save(components={"unet": make_model(), "config": dict(config)}, path=path)
    assert path.exists(), f"{saver_cls.__name__} wrote no file"
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


@pytest.mark.parametrize("saver_key", ["generic", "flux2", "hidream_o1", "microsoft_lens", "sdxl"])
def test_saver_writes_min_snr_only_when_the_config_carries_it(saver_key, tmp_path):
    saver_cls, make_model = _savers()[saver_key]
    without = _saved_header(saver_cls, make_model, _flux2_config_after_allowlist(), tmp_path / "a")
    assert "ss_min_snr_gamma" not in without, f"{saver_key}: {without.get('ss_min_snr_gamma')}"
    assert without.get("ss_learning_rate") == "0.0001", f"{saver_key} wrote no config metadata at all"
    carried = _saved_header(saver_cls, make_model, {"min_snr_gamma": 5.0, "learning_rate": 1e-4},
                            tmp_path / "b")
    assert carried.get("ss_min_snr_gamma") == "5.0", saver_key


def test_lora_tools_reverse_map_still_reads_ss_min_snr_gamma():
    from app.engine.utils.lora_tools import _parse_training_params

    assert _parse_training_params({"ss_min_snr_gamma": "5.0"})["min_snr_gamma"] == 5.0
    assert "min_snr_gamma" not in _parse_training_params({"ss_learning_rate": "0.0001"})
