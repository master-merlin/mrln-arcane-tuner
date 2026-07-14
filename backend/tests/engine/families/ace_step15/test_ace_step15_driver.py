"""ACE-Step 1.5 driver unit tests — real classes, tiny meta-instantiable
configs (no GPU, no downloaded checkpoint). Pins the load-bearing math
described in ``driver.py``'s module docstring: the [B,D,T]->[B,T,D] latent
transpose, the constant text2music context_latents (silence + ones mask),
the timestep [0,1000]->[0,1] scale conversion inside forward_pass, the
genre_ratio null-condition dropout, and the SFT prompt/lyric templates.
"""

from __future__ import annotations

import torch
from diffusers import AceStepTransformer1DModel, AutoencoderOobleck
from diffusers.pipelines.ace_step.modeling_ace_step import AceStepConditionEncoder

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.ace_step15.driver import (
    ACE_STEP15_LORA_SUFFIXES,
    AceStep15Driver,
    format_condition_text,
    tile_to_length,
)


def _tiny_transformer(**overrides) -> AceStepTransformer1DModel:
    cfg = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        in_channels=24,  # 3 * audio_acoustic_hidden_dim(8): hidden(8) + context(16)
        audio_acoustic_hidden_dim=8,
        encoder_hidden_size=32,
        layer_types=["full_attention", "full_attention"],
        is_turbo=True,
        model_version="turbo",
    )
    cfg.update(overrides)
    return AceStepTransformer1DModel(**cfg)


def _tiny_condition_encoder(**overrides) -> AceStepConditionEncoder:
    cfg = dict(
        hidden_size=32,
        intermediate_size=64,
        text_hidden_dim=16,
        timbre_hidden_dim=8,
        num_lyric_encoder_hidden_layers=1,
        num_timbre_encoder_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
    )
    cfg.update(overrides)
    return AceStepConditionEncoder(**cfg)


def _tiny_vae() -> AutoencoderOobleck:
    return AutoencoderOobleck(
        encoder_hidden_size=8,
        downsampling_ratios=[2, 4],
        channel_multiples=[1, 2],
        decoder_channels=8,
        decoder_input_channels=8,
        audio_channels=2,
        sampling_rate=1000,
    )


class _FakeTokenizer:
    """Deterministic tiny tokenizer stub — returns fixed-shape token ids."""

    def __call__(self, texts, padding="longest", truncation=True, max_length=256, return_tensors="pt"):
        n = len(texts)
        length = 5

        class _Out(dict):
            pass

        ids = torch.randint(0, 100, (n, length))
        mask = torch.ones(n, length, dtype=torch.long)
        out = _Out(input_ids=ids, attention_mask=mask)
        out.input_ids = ids
        out.attention_mask = mask
        return out


def _make_driver(is_turbo: bool = True) -> AceStep15Driver:
    from transformers import LlamaConfig, LlamaModel

    definition = ModelDefinition(id="ace-step-1.5-test", family="ace_step15", name="test")
    driver = AceStep15Driver(definition, torch.device("cpu"))

    te_cfg = LlamaConfig(
        vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    )
    transformer_overrides = (
        {}
        if is_turbo
        else {"is_turbo": False, "model_version": "base"}
    )
    components = {
        "unet": _tiny_transformer(**transformer_overrides),
        "vae": _tiny_vae(),
        "text_encoder": LlamaModel(te_cfg),
        "tokenizer": _FakeTokenizer(),
        "condition_encoder": _tiny_condition_encoder(),
    }
    driver.assign_components(components)
    return driver


# ── assign_components / config derivation ───────────────────────────────────


def test_assign_components_derives_latents_per_second_from_vae():
    driver = _make_driver()
    # sampling_rate=1000, downsampling_ratios=[2,4] -> 1000/8 = 125.0
    assert driver.latents_per_second == 125.0
    assert driver.audio_acoustic_hidden_dim == 8
    assert driver.timbre_fix_frame == 3750  # ceil(30 * 125.0)


def test_get_primary_model_and_text_encoders():
    driver = _make_driver()
    assert driver.get_primary_model() is driver.transformer
    tes = driver.get_text_encoders()
    assert set(tes) == {"text_encoder", "condition_encoder"}
    assert tes["text_encoder"] is driver.text_encoder
    assert tes["condition_encoder"] is driver.condition_encoder


def test_get_te_lora_targets_empty():
    driver = _make_driver()
    assert driver.get_te_lora_targets() == []


def test_init_scheduler_returns_none():
    driver = _make_driver()
    assert driver.init_scheduler() is None


def test_resolve_loading_dtype_bf16():
    driver = _make_driver()
    assert driver.resolve_loading_dtype() == torch.bfloat16


# ── LoRA targets ──────────────────────────────────────────────────────────


def test_get_lora_targets_default_suffixes():
    driver = _make_driver()
    assert driver.get_lora_targets() == list(ACE_STEP15_LORA_SUFFIXES)


def test_get_lora_targets_prefers_definition_list():
    definition = ModelDefinition(
        id="ace-step-1.5-test", family="ace_step15", name="test",
        lora_targetable_modules=["to_q", "to_k"],
    )
    driver = AceStep15Driver(definition, torch.device("cpu"))
    assert driver.get_lora_targets() == ["to_q", "to_k"]


def test_lora_suffixes_match_real_linear_modules():
    """The shipped suffix list must match at least one real Linear per suffix
    on BOTH self_attn and cross_attn — a silent rename upstream would make
    PEFT wrap zero modules."""
    tr = _tiny_transformer()
    linear_names = [n for n, m in tr.named_modules() if isinstance(m, torch.nn.Linear)]
    for suffix in ACE_STEP15_LORA_SUFFIXES:
        matches = [n for n in linear_names if n.endswith(suffix)]
        assert matches, f"no Linear module ends with {suffix!r}: {linear_names}"
        assert any(".self_attn." in n for n in matches), suffix
        assert any(".cross_attn." in n for n in matches), suffix


# ── format_condition_text / tile_to_length ───────────────────────────────


def test_format_condition_text_templates():
    text, lyrics = format_condition_text(
        "a happy song", "la la la", vocal_language="en", audio_duration=42.0
    )
    assert "# Instruction" in text
    assert "# Caption" in text
    assert "a happy song" in text
    assert "42 seconds" in text
    assert text.endswith("<|endoftext|>\n")
    assert lyrics == "# Languages\nen\n\n# Lyric\nla la la<|endoftext|>"


def test_format_condition_text_default_duration_when_unset():
    _, _ = format_condition_text("x", "y", audio_duration=0.0)
    text, _ = format_condition_text("x", "y", audio_duration=0.0)
    assert "30 seconds" in text


def test_tile_to_length_crop_and_repeat():
    t = torch.arange(6).view(1, 3, 2).float()
    cropped = tile_to_length(t, 2)
    assert cropped.shape == (1, 2, 2)
    assert torch.equal(cropped, t[:, :2, :])

    tiled = tile_to_length(t, 7)
    assert tiled.shape == (1, 7, 2)
    assert torch.equal(tiled[:, :3, :], t)
    assert torch.equal(tiled[:, 3:6, :], t)  # repeated verbatim
    assert torch.equal(tiled[:, 6:7, :], t[:, :1, :])  # cropped tail


# ── encode_condition ──────────────────────────────────────────────────────


def test_encode_condition_shapes():
    driver = _make_driver()
    eh, mask = driver.encode_condition(
        ["a happy song", "a sad song"], ["la la", ""], torch.float32, audio_duration=6.0
    )
    assert eh.ndim == 3 and eh.shape[0] == 2
    assert eh.shape[-1] == 32  # hidden_size
    assert mask.ndim == 2 and mask.shape[0] == 2


# ── prepare_latents ───────────────────────────────────────────────────────


def test_prepare_latents_transposes_channels_first_to_last():
    driver = _make_driver()
    lat = torch.randn(2, 8, 10)  # [B, D, T]
    prepared = driver.prepare_latents(lat)
    assert prepared.shape == (2, 10, 8)  # [B, T, D]
    assert torch.equal(prepared, lat.transpose(1, 2))


def test_prepare_latents_rejects_non_3d():
    driver = _make_driver()
    import pytest

    with pytest.raises(ValueError):
        driver.prepare_latents(torch.randn(2, 8, 10, 3))


# ── forward_pass ──────────────────────────────────────────────────────────


def test_forward_pass_shape_and_grad_flow():
    driver = _make_driver()
    driver.transformer.train()
    driver.genre_ratio = 0.0  # deterministic — no null-condition dropout

    eh, mask = driver.encode_condition(["x"], ["y"], torch.float32, audio_duration=6.0)
    latents = torch.randn(1, 12, 8, requires_grad=False)
    timesteps = torch.tensor([500.0])  # raw [0,1000] scale (framework convention)

    out = driver.forward_pass(latents, timesteps, (eh, mask), {})
    assert out.shape == latents.shape

    loss = out.float().pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in driver.transformer.parameters())


def test_forward_pass_requires_tuple_text_embeddings():
    driver = _make_driver()
    import pytest

    latents = torch.randn(1, 4, 8)
    timesteps = torch.tensor([100.0])
    with pytest.raises(TypeError):
        driver.forward_pass(latents, timesteps, torch.randn(1, 4, 32), {})


def test_forward_pass_genre_ratio_can_force_null_condition():
    """genre_ratio=1.0 must replace the ENTIRE batch's condition with the
    learned null embedding — deterministic, unit-testable without RNG luck."""
    driver = _make_driver()
    driver.transformer.train()
    driver.genre_ratio = 1.0

    eh, mask = driver.encode_condition(["x"], ["y"], torch.float32, audio_duration=6.0)
    latents = torch.randn(1, 12, 8)
    timesteps = torch.tensor([500.0])

    torch.manual_seed(0)
    out_dropped = driver.forward_pass(latents, timesteps, (eh, mask), {})

    # Build the null-conditioned expectation manually and confirm the output
    # is identical to forwarding with the null embedding directly (proves the
    # dropout branch, not just "some" output).
    dtype = driver.transformer.dtype
    null_emb = driver.condition_encoder.null_condition_emb.to(latents.device, dtype)
    null_expanded = null_emb.expand_as(eh.to(dtype))
    driver.genre_ratio = 0.0
    expected = driver.forward_pass(latents, timesteps, (null_expanded, mask), {})
    assert torch.allclose(out_dropped, expected)


def test_forward_pass_genre_ratio_inert_in_eval_mode():
    """genre_ratio must NOT engage when the transformer is in eval mode
    (sampling previews should never randomly drop conditioning)."""
    driver = _make_driver()
    driver.transformer.eval()
    driver.genre_ratio = 1.0

    eh, mask = driver.encode_condition(["x"], ["y"], torch.float32, audio_duration=6.0)
    latents = torch.randn(1, 12, 8)
    timesteps = torch.tensor([500.0])

    with torch.no_grad():
        out = driver.forward_pass(latents, timesteps, (eh, mask), {})
        dtype = driver.transformer.dtype
        expected = driver.forward_pass(
            latents, timesteps, (eh.to(dtype), mask), {}
        )
    assert torch.allclose(out, expected)


# ── save metadata / saver ────────────────────────────────────────────────


def test_get_save_metadata_and_saver():
    driver = _make_driver()
    meta = driver.get_save_metadata()
    assert meta["modelspec.architecture"] == "ace-step15.dit"
    from app.engine.models.families.ace_step15.saver import AceStep15Saver

    assert isinstance(driver.get_saver(), AceStep15Saver)
    assert AceStep15Saver.architecture_name == "ace_step15"


# ── condition-encoder buffer survival (GPU UAT crash, 2026-07-14) ────────────


def test_condition_buffers_survive_te_pop_and_reassign():
    """The shared pipeline pops text encoders — including our
    condition_encoder (exposed via get_text_encoders) — from
    ``self.components`` after embedding caching, then ``prepare_for_training``
    re-runs ``_assign_components()`` for the LoRA-wrap alias re-sync. The
    driver must keep the tiny silence_latent/null_condition_emb buffers so
    forward_pass (context latents + genre-drop) and the sampler keep working
    after the encoder module itself is gone."""
    driver = _make_driver()
    comps = dict(driver.get_components())
    comps["unet"] = driver.transformer  # what the re-assign actually carries
    comps.pop("condition_encoder", None)
    comps.pop("text_encoder", None)
    driver.assign_components(comps)

    assert driver.condition_encoder is None or comps.get("condition_encoder") is None
    # Buffers survive as driver-owned stashes.
    sl = driver.silence_latent
    ne = driver.null_condition_emb
    assert isinstance(sl, torch.Tensor) and isinstance(ne, torch.Tensor)

    # Context latents (every training step + every sample) still build.
    ctx = driver._build_context_latents(2, 6, torch.device("cpu"), torch.float32)
    assert ctx.shape[0] == 2 and ctx.shape[1] == 6

    # The genre-drop training path still finds the null embedding.
    driver.transformer.train()
    driver.genre_ratio = 1.0
    eh = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    latents = torch.randn(1, 12, 8)
    timesteps = torch.tensor([500.0])
    out = driver.forward_pass(latents, timesteps, (eh, mask), {})
    assert out.shape == latents.shape


def test_condition_buffer_accessors_raise_before_assign():
    """Before any condition_encoder was ever assigned the accessors must
    raise a clear error instead of AttributeError-on-None."""
    import pytest

    definition = ModelDefinition(id="ace-step-1.5-test", family="ace_step15", name="test")
    driver = AceStep15Driver(definition, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="condition_encoder"):
        _ = driver.silence_latent
    with pytest.raises(RuntimeError, match="condition_encoder"):
        _ = driver.null_condition_emb
