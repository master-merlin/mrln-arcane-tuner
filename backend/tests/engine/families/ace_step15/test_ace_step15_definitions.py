"""ACE-Step 1.5 definition YAML sanity — mirrors kandinsky5's
test_kandinsky5_definitions.py. Pins the id/family/repo/architecture_params
contract and the non-empty LoRA target-list guard (registry.enrich_definition
would otherwise auto-fill it with the introspector's exhaustive Linear
catalog — dreamlite 2026-07-08 precedent, see test_lora_target_lists_shipped.py).
"""

from __future__ import annotations

import pytest

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


def test_definition_transformer_params_match_real_checkpoint_config():
    """Task C4a fix: the turbo definition's transformer.* must match the
    REAL `ACE-Step/acestep-v15-xl-turbo-diffusers/transformer/config.json`
    (hidden_size=2560/32L/32H, byte-verified via the HF tree API) — C1
    originally documented 2048/24L/16H, sourced from the legacy
    combined-wrapper repo's root config (a differently-scoped model)."""
    defn = _get_definition()
    arch = defn.architecture_params
    assert arch["transformer.hidden_size"] == 2560
    assert arch["transformer.intermediate_size"] == 9728
    assert arch["transformer.num_hidden_layers"] == 32
    assert arch["transformer.num_attention_heads"] == 32
    assert arch["transformer.num_key_value_heads"] == 8
    assert arch["transformer.head_dim"] == 128
    assert arch["transformer.model_version"] == "turbo"
    assert defn.block_topology[0]["count"] == 32


def test_definition_ships_real_model_size_mb():
    """Task C4a: turbo now ships concrete on-disk sizes (previously an
    intentionally-empty dict routed to the — then wrong — family fallback).
    Same byte-derived numbers as xl_base (shards are size-identical across
    the two checkpoint variants)."""
    defn = _get_definition()
    size_mb = defn.model_size_mb
    assert size_mb["transformer"] == pytest.approx(7951.6)
    assert size_mb["text_encoder"] == 2298
    assert size_mb["vae"] == 322


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
    32 layers, 32 heads). Since the C4a fix, the turbo definition documents
    the SAME (correct) shape — see
    test_definition_transformer_params_match_real_checkpoint_config."""
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
    """xl_base ships real on-disk sizes so the VRAM estimator prefers them
    over the ``_FAMILY_PARAMS["ace_step15"]`` fallback (since the C4a fix
    both the fallback and turbo's own model_size_mb agree with these
    numbers — the shards are size-identical across checkpoint variants)."""
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


# ── enrichment pins (GPU UAT regression, 2026-07-14) ─────────────────────────


def test_both_definitions_pin_scheduler_shift_against_enrichment():
    """`scheduler.shift: 3.0` is the MODEL CARD's recommended value; the
    repo's own scheduler_config.json ships shift=1.0. Enrichment's
    "harvested wins" policy overwrote the pin during the first real load
    (GPU UAT 2026-07-14) which silently degrades every preview. Both
    definitions must declare the key pinned."""
    ModelRegistry.initialize()
    for def_id in ("ace-step-1.5", "ace-step-1.5-xl-base"):
        defn = next(
            d for d in ModelRegistry._definitions.values() if d.id == def_id
        )
        assert "scheduler.shift" in defn.enrich_pinned_keys, def_id
        assert defn.architecture_params["scheduler.shift"] == 3.0, def_id


def test_enrich_definition_respects_pinned_keys(tmp_path, monkeypatch):
    """enrich_definition must keep the YAML value for keys listed in
    `enrich_pinned_keys` even when the harvested checkpoint config
    disagrees — the drift is still logged, the value is not clobbered."""
    from app.engine.core.definitions import ModelDefinition

    ModelRegistry.initialize()
    defn = ModelDefinition(
        id="pin-test", family="ace_step15", name="pin test",
        architecture_params={"scheduler.shift": 3.0, "transformer.hidden_size": 1},
        lora_targetable_modules=["q_proj"],
        enrich_pinned_keys=["scheduler.shift"],
    )
    ModelRegistry._definitions["pin-test"] = defn

    class _FakeResult:
        detected_precision = {}
        architecture_params = {
            "scheduler.shift": 1.0,          # pinned — must NOT win
            "transformer.hidden_size": 2560,  # unpinned — must win
        }
        lora_targetable_modules = []

    class _FakeIntrospector:
        def introspect(self, components):
            return _FakeResult()

    import app.engine.utils.introspection as intro_mod

    monkeypatch.setattr(intro_mod, "ModelIntrospector", _FakeIntrospector)
    try:
        ModelRegistry.enrich_definition("pin-test", {})
        arch = ModelRegistry.get_definition("pin-test").architecture_params
        assert arch["scheduler.shift"] == 3.0
        assert arch["transformer.hidden_size"] == 2560
    finally:
        ModelRegistry._definitions.pop("pin-test", None)
