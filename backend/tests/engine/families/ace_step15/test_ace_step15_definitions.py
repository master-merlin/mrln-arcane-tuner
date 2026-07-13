"""ACE-Step 1.5 definition YAML sanity — mirrors kandinsky5's
test_kandinsky5_definitions.py. Pins the id/family/repo/architecture_params
contract and the non-empty LoRA target-list guard (registry.enrich_definition
would otherwise auto-fill it with the introspector's exhaustive Linear
catalog — dreamlite 2026-07-08 precedent, see test_lora_target_lists_shipped.py).
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry


def _get_definition():
    ModelRegistry.initialize()
    return next(
        d for d in ModelRegistry._definitions.values() if d.id == "ace-step-1.5"
    )


def test_definition_identity():
    defn = _get_definition()
    assert defn.family == "ace_step15"
    assert defn.control_inputs == 0


def test_definition_repo_is_diffusers_native():
    defn = _get_definition()
    repo = defn.components["repo"]
    assert repo.type == "diffusers"
    # Deliberate deviation from the GitHub repo id (trust_remote_code layout)
    # — see definitions/base.yaml's header comment + driver.py module
    # docstring for why the diffusers-native repo is required.
    assert repo.path == "huggingface:ACE-Step/acestep-v15-xl-turbo-diffusers"


def test_definition_architecture_params_audio_contract():
    defn = _get_definition()
    arch = defn.architecture_params
    assert arch["audio.sample_rate"] == 48000
    assert arch["audio.channels"] == 2
    assert arch["audio.latent_hz"] == 25.0
    assert arch["audio.acoustic_hidden_dim"] == 64
    assert arch["transformer._class_name"] == "AceStepTransformer1DModel"
    assert arch["transformer.in_channels"] == 192
    assert arch["condition_encoder._class_name"] == "AceStepConditionEncoder"
    assert arch["vae._class_name"] == "AutoencoderOobleck"
    assert arch["transformer.is_turbo"] is True


def test_definition_defaults_match_upstream_preset():
    defn = _get_definition()
    d = defn.defaults
    assert d["timestep_sampling"] == "logit_normal"
    assert d["logit_normal_mu"] == -0.4
    assert d["logit_normal_sigma"] == 1.0
    assert d["train_batch_size"] == 1
    assert d["gradient_accumulation_steps"] == 4
    assert d["guidance_scale"] == 1.0  # turbo — CFG off
    assert d["num_inference_steps"] == 8
    assert d["duration_s"] == 30.0
    assert d["genre_ratio"] == 0.15


def test_definition_ships_nonempty_lora_target_list():
    defn = _get_definition()
    assert defn.lora_targetable_modules == ["to_q", "to_k", "to_v", "to_out.0"]


def test_definition_detected_precision():
    defn = _get_definition()
    assert defn.detected_precision["vae"] == "torch.float32"
    assert defn.detected_precision["unet"] == "torch.bfloat16"
    assert defn.detected_precision["text_encoder"] == "torch.bfloat16"
    assert defn.detected_precision["condition_encoder"] == "torch.bfloat16"
