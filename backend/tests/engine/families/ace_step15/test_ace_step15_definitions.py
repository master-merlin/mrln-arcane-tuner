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


# ── xl_base.yaml (task C2 — non-turbo, true-CFG/APG checkpoint) ──────────


def _get_xl_base_definition():
    ModelRegistry.initialize()
    return next(
        d
        for d in ModelRegistry._definitions.values()
        if d.id == "ace-step-1.5-xl-base"
    )


def test_xl_base_definition_identity():
    defn = _get_xl_base_definition()
    assert defn.family == "ace_step15"
    assert defn.control_inputs == 0
    assert defn.name == "ACE-Step 1.5 XL (Base)"


def test_xl_base_definition_repo_is_diffusers_native():
    defn = _get_xl_base_definition()
    repo = defn.components["repo"]
    assert repo.type == "diffusers"
    assert repo.path == "huggingface:ACE-Step/acestep-v15-xl-base-diffusers"


def test_xl_base_definition_architecture_params_match_byte_verified_config():
    """Pins the RECON-VERIFIED real transformer shape (hidden_size=2560,
    32 layers, 32 heads) — deliberately NOT the smaller shape the shipped
    turbo definition documents (hidden_size=2048/24/16), which task C2's
    recon found is sourced from the wrong upstream config file (see
    xl_base.yaml's header comment)."""
    defn = _get_xl_base_definition()
    arch = defn.architecture_params
    assert arch["audio.sample_rate"] == 48000
    assert arch["audio.latent_hz"] == 25.0
    assert arch["transformer._class_name"] == "AceStepTransformer1DModel"
    assert arch["transformer.hidden_size"] == 2560
    assert arch["transformer.intermediate_size"] == 9728
    assert arch["transformer.num_hidden_layers"] == 32
    assert arch["transformer.num_attention_heads"] == 32
    assert arch["transformer.in_channels"] == 192
    assert arch["transformer.is_turbo"] is False
    assert arch["transformer.model_version"] == "base"
    assert arch["condition_encoder._class_name"] == "AceStepConditionEncoder"
    assert arch["vae._class_name"] == "AutoencoderOobleck"
    assert arch["scheduler.shift"] == 3.0


def test_xl_base_definition_defaults_match_model_card():
    """num_inference_steps=50 / guidance_scale=7.0 / shift=3.0 are the
    model card's own recommended defaults (task C2 recon) — guidance_scale
    > 1.0 combined with is_turbo=False is what flips sampler.py onto the
    real APG path (see test_ace_step15_sampler.py)."""
    defn = _get_xl_base_definition()
    d = defn.defaults
    assert d["timestep_sampling"] == "logit_normal"
    assert d["logit_normal_mu"] == -0.4
    assert d["logit_normal_sigma"] == 1.0
    assert d["num_inference_steps"] == 50
    assert d["guidance_scale"] == 7.0
    assert d["duration_s"] == 30.0
    assert d["genre_ratio"] == 0.15


def test_xl_base_definition_ships_nonempty_lora_target_list():
    defn = _get_xl_base_definition()
    assert defn.lora_targetable_modules == ["to_q", "to_k", "to_v", "to_out.0"]


def test_xl_base_definition_ships_own_model_size_mb():
    """Unlike turbo's empty ``model_size_mb: {}``, xl_base ships real
    on-disk sizes so the VRAM estimator does not fall back to
    ``_FAMILY_PARAMS["ace_step15"]["transformer"]`` (1.575 B) — that
    fallback is calibrated to turbo's incorrect, smaller documented shape
    (see xl_base.yaml header comment) and would understate this
    definition's real ~4.17 B-parameter DiT by ~2.6x."""
    defn = _get_xl_base_definition()
    size_mb = defn.model_size_mb
    assert size_mb["transformer"] == 7952
    assert size_mb["text_encoder"] == 2298
    assert size_mb["vae"] == 322


def test_xl_base_definition_block_topology_matches_real_layer_count():
    defn = _get_xl_base_definition()
    topo = defn.block_topology
    assert len(topo) == 1
    assert topo[0]["count"] == 32


def test_xl_base_and_turbo_share_condition_encoder_and_vae_config():
    """The condition encoder + VAE are byte-identical across checkpoint
    variants (task C2 recon, HF LFS content-hash comparison) — only the
    transformer differs. Both definitions must agree on these shared
    component shapes."""
    turbo = _get_definition()
    base = _get_xl_base_definition()
    for key in (
        "condition_encoder.hidden_size",
        "condition_encoder.text_hidden_dim",
        "condition_encoder.timbre_hidden_dim",
        "condition_encoder.num_lyric_encoder_hidden_layers",
        "condition_encoder.num_timbre_encoder_hidden_layers",
        "te.hidden_size",
        "vae.sampling_rate",
        "vae.audio_channels",
        "vae.decoder_input_channels",
    ):
        assert turbo.architecture_params[key] == base.architecture_params[key], key
