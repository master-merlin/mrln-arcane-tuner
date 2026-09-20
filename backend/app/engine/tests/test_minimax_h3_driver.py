"""minimax_h3 family registration + capability flags + driver non-training surface."""

from __future__ import annotations

import ast
from pathlib import Path

from app.engine.models.registry import ModelRegistry


def _reload_registry() -> type[ModelRegistry]:
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()
    return ModelRegistry


def test_family_is_auto_discovered():
    assert "minimax_h3" in _reload_registry()._families


def test_capability_flags_declare_video_and_audio():
    family_cls = _reload_registry()._families["minimax_h3"]
    caps = family_cls.capability_overrides
    assert caps["is_video"] is True
    assert caps["has_audio"] is True
    assert caps["has_image_encoder"] is True
    # H3 is single-stream: there is no second expert to schedule.
    assert caps["dual_expert"] is False
    # supports_train_te is intentionally NOT asserted here: only sdxl may put
    # that key in capability_overrides (test_only_sdxl_overrides_train_te);
    # minimax_h3 relies on the latent_diffusion archetype's False default.
    # The ~66.7 GB Qwen3-VL TE must be cacheable.
    assert caps["te_cache"] is True


# ---------------------------------------------------------------------------
# Task 6: driver non-training surface
# ---------------------------------------------------------------------------

def _driver(def_id: str = "minimax-h3-t2va"):
    import torch

    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return MiniMaxH3Driver(ModelRegistry._definitions[def_id], torch.device("cpu"))


def test_definition_ships_curated_target_list_matching_driver():
    """The nucleus_image contract: YAML and driver must agree EXACTLY.

    If they drift, the introspector's exhaustive catalog silently overwrites
    the curated list at first model load and the run adapts the wrong modules.
    """
    import pathlib
    import yaml

    defs_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "models" / "families" / "minimax_h3" / "definitions"
    )
    for path in defs_dir.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        driver = _driver(data["id"])
        assert sorted(driver.get_lora_targets()) == sorted(
            data["lora_targetable_modules"]
        ), f"{data['id']}: driver/YAML target-list drift"


def test_block_topology_is_read_from_the_definition():
    topo = _driver().get_block_topology()
    assert sum(entry["count"] for entry in topo) == 52


def test_init_scheduler_still_returns_none():
    """The deliberate exception to "raise, never a plausible default":
    ``None`` IS the real answer (H3 trains with flow matching, no external
    scheduler) — and the ONLY body ``hook_dispatch.TRIVIAL_BODIES`` treats as
    the ``init_scheduler`` no-op baseline. Any other body (including a raise)
    would enroll minimax_h3 in the auto-delegation allowlist and trip
    ``test_autodelegated_family_hook_set_is_exactly_expected``. See
    ``driver.py``'s module docstring."""
    assert _driver().init_scheduler() is None


def test_resolve_loading_dtype_is_bf16():
    import torch

    assert _driver().resolve_loading_dtype() is torch.bfloat16


def test_get_te_lora_targets_is_empty_te_never_trains():
    assert _driver().get_te_lora_targets() == []


def test_assign_components_wires_the_five_manifest_keys():
    driver = _driver()
    components = {
        "tokenizer": object(),
        "text_encoder": object(),
        "vae": object(),
        "audio_vae": object(),
        "transformer": object(),
    }
    driver.assign_components(components)
    assert driver.get_components() is components
    assert driver.get_primary_model() is components["transformer"]
    assert driver.get_text_encoders() == {"text_encoder": components["text_encoder"]}


# ---------------------------------------------------------------------------
# Plan row 2.1: text encoding (Qwen3-VL layer tap), the TE cache key, the
# release-before-DiT check, and the INCREMENTAL PR0-refusal guard.
# ---------------------------------------------------------------------------

def _text_driver(def_id: str = "minimax-h3-t2va", *, tokenizer=None, num_layers: int = 64):
    """A driver with a stub Qwen3-VL + processor assigned, no weights."""
    from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL

    driver = _driver(def_id)
    te = StubQwen3VL(num_hidden_layers=num_layers)
    components = {
        "tokenizer": StubProcessor(tokenizer),
        "text_encoder": te,
        "vae": object(),
        "audio_vae": object(),
    }
    driver.assign_components(components)
    return driver, te


def test_encode_text_returns_the_layer_tap_embedding():
    """The embedding IS ``hidden_states[te.hidden_state_tap_index]`` (50 for the
    shipped definitions: MiniMax README "hidden states from its 50th layer";
    ai-toolkit ``text_encoder.py`` "unnormalized hidden_states[50]") — read off
    the returned tensor, whose every element equals the tap index by the stub's
    construction. Raw tokens (no chat template), the caption trimmed to
    ``te.max_length`` (512, ai-toolkit's evidenced default), the empty prompt
    encoded as ONE pad token, per-caption masks re-padded to the batch max."""
    import torch

    driver, te = _text_driver()
    arch = driver.definition.architecture_params
    assert arch["te.hidden_state_tap_index"] == 50
    assert arch["te.max_length"] == 512

    long_caption = " ".join(["word"] * 600)  # 600 tokens under the stub tokenizer
    out = driver.encode_text(["a cat sits", "", long_caption], torch.bfloat16)

    emb, mask = out.embeddings, out.attention_mask
    assert emb.dtype is torch.bfloat16
    assert emb.shape == (3, 512, te.hidden_size), emb.shape
    assert mask.shape == (3, 512)
    # Every real token row carries the tap index, not the last layer (64).
    assert torch.all(emb[mask.bool()].float() == 50.0), "not the layer-50 tap"
    # Lengths: 3 tokens, 1 pad token for the empty prompt, 512 after the trim.
    assert mask.sum(dim=1).tolist() == [3, 1, 512]
    # Padding rows are zero, not the tap value.
    assert torch.all(emb[0, 3:] == 0)
    # Raw, per-caption encodes: 3 ids, 1 pad id, 512 after the trim (never 600).
    assert sorted(t.shape[1] for t in te.seen_input_ids) == [1, 3, 512]


def test_te_cache_key_includes_tap_and_tokenizer():
    """Two encoders that differ ONLY in the tap index, or ONLY in the tokenizer
    vocabulary, must never share a cached embedding (key-collision test)."""
    from app.engine.tests.h3_text_stubs import StubTokenizer

    base, _ = _text_driver()
    same, _ = _text_driver()
    other_tap, _ = _text_driver()
    other_tap.definition = other_tap.definition.model_copy(
        update={
            "architecture_params": {
                **other_tap.definition.architecture_params,
                "te.hidden_state_tap_index": 49,
            }
        }
    )
    other_tok, _ = _text_driver(tokenizer=StubTokenizer(vocab_tag="v2"))

    prompt = "a cat sits"
    assert base.te_cache_key(prompt) == same.te_cache_key(prompt)
    assert base.te_cache_key(prompt) != base.te_cache_key("a dog sits")
    assert base.te_cache_key(prompt) != other_tap.te_cache_key(prompt), (
        "tap index missing from the key: layer-49 and layer-50 embeddings collide"
    )
    assert base.te_cache_key(prompt) != other_tok.te_cache_key(prompt), (
        "tokenizer fingerprint missing from the key"
    )
    assert "tap=50" in base.te_cache_key(prompt)


def test_text_encoder_released_before_dit_load():
    """The DiT never loads while the 63 GB encoder is resident: the check
    names the bytes; after ``release_text_encoders()`` the driver reports no
    encoder and the check passes."""
    import pytest

    driver, te = _text_driver()
    with pytest.raises(RuntimeError, match=r"text encoder still resident \(.* GB\) at DiT load"):
        driver.assert_text_encoder_released()
    assert driver.text_encoder_weight_bytes() == te.weight.numel() * te.weight.element_size()

    driver.release_text_encoders()
    assert driver.get_text_encoders() == {}
    assert driver.text_encoder is None
    driver.assert_text_encoder_released()  # no raise


def test_cache_fingerprint_includes_pixel_adapter_and_vae_identity(monkeypatch):
    """Row 2.3 (D10 "keyed on every input"): the RAW VAE latent depends on the
    pixel convention (`PIXEL_ADAPTER_VERSION`), the VAE class and its config —
    and on nothing else (a sigma shift or the packed-layout version must NOT
    re-encode a dataset)."""
    from app.engine.models.families.minimax_h3 import pixel_adapter
    from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
    from app.engine.tests.h3_text_stubs import StubVisualVAE

    def _with_vae(inner):
        driver = _driver()
        driver.assign_components({"vae": H3PixelAdaptedVAE(inner)})
        return driver

    base = _with_vae(StubVisualVAE())
    same = _with_vae(StubVisualVAE())
    fp = base.latent_cache_fingerprint()
    assert len(fp) == 8 and int(fp, 16) >= 0
    assert fp == same.latent_cache_fingerprint()

    other_config = _with_vae(StubVisualVAE(config={"latent_channels": 64}))
    assert other_config.latent_cache_fingerprint() != fp, "VAE config not in the key"

    class OtherVAE(StubVisualVAE):
        pass

    assert _with_vae(OtherVAE()).latent_cache_fingerprint() != fp, "VAE class not in the key"

    monkeypatch.setattr(pixel_adapter, "PIXEL_ADAPTER_VERSION", pixel_adapter.PIXEL_ADAPTER_VERSION + 1)
    assert _with_vae(StubVisualVAE()).latent_cache_fingerprint() != fp, (
        "two pixel conventions share a key"
    )


# ── Row 2.4: the joint forward + the three-number loss on the driver ──────


def _settings(config: dict | None = None):
    from app.engine.models.families.minimax_h3.settings import resolve_h3_settings

    cfg = {"train_audio": True}
    cfg.update(config or {})
    return resolve_h3_settings(_driver().definition, cfg)


def _forward_driver(build_tiny_transformer, config: dict | None = None):
    driver = _driver()
    driver.assign_components({"transformer": build_tiny_transformer().eval()})
    driver.apply_settings(_settings(config))
    return driver


def _text(lengths: list[int], dim: int = 16):
    import torch

    from app.engine.core.text_encoding import TextEncoderOutput

    g = torch.Generator().manual_seed(7)
    L = max(lengths)
    emb = torch.randn(len(lengths), L, dim, generator=g)
    mask = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
        emb[i, n:] = 0
    return TextEncoderOutput(embeddings=emb, attention_mask=mask)


def test_forward_pass_returns_video_and_audio_velocities(build_tiny_transformer):
    """One packed sequence per item (its OWN text length — the batch padding
    never enters the transformer), the distinct set `[t_v, t_a]` with `t_a`
    derived from `t_v` through the dual-shift schedule, and the velocities
    handed back in the raw latent shapes the training loop holds."""
    import torch

    from app.engine.models.families.minimax_h3.packing import (
        H3Geometry,
        build_layout,
        pack_audio,
        packed_forward,
        patchify_video,
        unpack_audio,
        unpatchify_video,
    )
    from app.engine.models.families.minimax_h3.schedule import remap_sigma, sigma_to_t, t_to_sigma

    # Scale 1.0 pinned: this test compares against ONE reference forward; the
    # scale-4.0 rearrangement has its own tests (row 3.1) below.
    driver = _forward_driver(build_tiny_transformer, {"cfg_augment_scale": 1.0})
    g = torch.Generator().manual_seed(3)
    video = torch.randn(2, 24, 2, 6, 4, generator=g)
    audio = torch.randn(2, 2, 32, 3, generator=g)
    text = _text([5, 8])
    t_v = torch.tensor([0.7, 0.2])

    t_a = driver.audio_timestep(t_v)
    expected_t_a = sigma_to_t(remap_sigma(t_to_sigma(t_v), 12.0, 3.0))
    assert torch.allclose(t_a, expected_t_a), "t_a is not the remapped video clock"

    with torch.no_grad():
        video_v, audio_v = driver.forward_pass(video, t_v, text, {"audio_noisy": audio})
    assert video_v.shape == video.shape and audio_v.shape == audio.shape
    assert torch.isfinite(video_v).all() and torch.isfinite(audio_v).all()

    # Reference: item 0 packed by hand with its 5 real text rows.
    layout = build_layout(H3Geometry(num_text=5, latent_frames=2, latent_height=6, latent_width=4, audio_latents=3))
    with torch.no_grad():
        ref_v, ref_a = packed_forward(
            driver.transformer,
            layout,
            patchify_video(video[:1], layout.patch_size),
            pack_audio(audio[:1]),
            text.embeddings[:1, :5],
            torch.stack([t_v[0], t_a[0]]),
        )
    assert torch.allclose(unpatchify_video(ref_v, layout), video_v[:1], atol=1e-5)
    assert torch.allclose(unpack_audio(ref_a, 3), audio_v[:1], atol=1e-5)

    # A batch of 5-token text padded to 8 is NOT the same sequence as 8 rows.
    layout8 = build_layout(H3Geometry(num_text=8, latent_frames=2, latent_height=6, latent_width=4, audio_latents=3))
    with torch.no_grad():
        padded_v, _ = packed_forward(
            driver.transformer,
            layout8,
            patchify_video(video[:1], layout8.patch_size),
            pack_audio(audio[:1]),
            text.embeddings[:1],
            torch.stack([t_v[0], t_a[0]]),
        )
    assert not torch.allclose(unpatchify_video(padded_v, layout8), video_v[:1], atol=1e-5), (
        "padding rows leaked into item 0's sequence"
    )


# ── CFG augmentation (plan row 3.1; research §3.7, re-implemented) ────────
#
# `out_aug = (out + (s − 1) · out_uncond) / s` on the MODEL OUTPUT (video and
# audio alike), the uncond forward on the empty-prompt TE row under
# `torch.no_grad()`; `s == 1.0` runs ONE forward. The uncond row is served
# by the trainer through `batch["text_embeddings_uncond"]` (the encoder is
# released before the DiT loads, so the driver cannot encode it itself).


class _CountingTransformer:
    """Wraps the tiny diffusers DiT and counts forwards (kwargs-only call,
    exactly the way `packing.packed_forward` invokes it)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.inner(**kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _cfg_driver(build_tiny_transformer, scale: float):
    driver = _driver()
    driver.assign_components({"transformer": _CountingTransformer(build_tiny_transformer().eval())})
    driver.apply_settings(_settings({"cfg_augment_scale": scale}))
    return driver


def _cfg_inputs():
    import torch

    g = torch.Generator().manual_seed(11)
    video = torch.randn(1, 24, 2, 6, 4, generator=g)
    audio = torch.randn(1, 2, 32, 3, generator=g)
    t_v = torch.tensor([0.6])
    cond = _text([6])
    uncond = _text([2])  # a DIFFERENT row, so a swapped branch is visible
    return video, audio, t_v, cond, uncond


def test_cfg_augment_scale_one_runs_single_forward(build_tiny_transformer):
    import torch

    driver = _cfg_driver(build_tiny_transformer, 1.0)
    video, audio, t_v, cond, uncond = _cfg_inputs()
    with torch.no_grad():
        driver.forward_pass(video, t_v, cond, {"audio_noisy": audio, "text_embeddings_uncond": uncond})
    assert driver.transformer.calls == 1, f"{driver.transformer.calls} forwards at scale 1.0"


def test_cfg_augment_scale_four_runs_two_forwards_and_rearranges_output(build_tiny_transformer):
    import torch

    video, audio, t_v, cond, uncond = _cfg_inputs()
    plain = _cfg_driver(build_tiny_transformer, 1.0)
    with torch.no_grad():
        v_cond, a_cond = plain.forward_pass(video, t_v, cond, {"audio_noisy": audio})
        v_unc, a_unc = plain.forward_pass(video, t_v, uncond, {"audio_noisy": audio})
    assert not torch.allclose(v_cond, v_unc), "cond and uncond rows give the same output; the check is vacuous"

    driver = _cfg_driver(build_tiny_transformer, 4.0)
    batch = {"audio_noisy": audio, "text_embeddings_uncond": uncond}
    with torch.no_grad():
        v_aug, a_aug = driver.forward_pass(video, t_v, cond, batch)
    assert driver.transformer.calls == 2, f"{driver.transformer.calls} forwards at scale 4.0"
    assert torch.allclose(v_aug, (v_cond + 3.0 * v_unc) / 4.0, atol=1e-5), "video output is not (out + 3·uncond)/4"
    assert torch.allclose(a_aug, (a_cond + 3.0 * a_unc) / 4.0, atol=1e-5), "audio output is not (out + 3·uncond)/4"
    assert torch.allclose(batch["video_pred_uncond"], v_unc, atol=1e-5)
    assert torch.allclose(batch["audio_pred_uncond"], a_unc, atol=1e-5)


def test_uncond_branch_does_not_participate_in_autograd(build_tiny_transformer):
    import torch

    driver = _cfg_driver(build_tiny_transformer, 4.0)
    video, audio, t_v, cond, uncond = _cfg_inputs()
    batch = {"audio_noisy": audio, "text_embeddings_uncond": uncond}
    v_aug, a_aug = driver.forward_pass(video, t_v, cond, batch)  # grad ENABLED, as in training
    assert batch["video_pred_uncond"].grad_fn is None, "uncond video branch is under grad"
    assert batch["audio_pred_uncond"].grad_fn is None, "uncond audio branch is under grad"
    assert v_aug.grad_fn is not None and a_aug.grad_fn is not None, "the cond branch lost its graph"
    noise = torch.randn_like(video)
    target = driver.compute_target(video, noise, t_v)
    assert target.requires_grad is False


def test_cfg_augment_without_uncond_row_refuses(build_tiny_transformer):
    import pytest
    import torch

    driver = _cfg_driver(build_tiny_transformer, 4.0)
    video, audio, t_v, cond, _ = _cfg_inputs()
    with pytest.raises(ValueError, match="text_embeddings_uncond"), torch.no_grad():
        driver.forward_pass(video, t_v, cond, {"audio_noisy": audio})


# ── Row 3.2: the step-0 banner + `train_audio=false` keeps the rows packed ──
#
# H3 is SINGLE-stream: the DiT always sees `[text | audio | video]`. Turning
# audio training off changes ONE number (`audio_loss_weight = 0.0`, source
# `train_audio_off`), never the sequence — a video-only sequence would be a
# layout the pretrained weights never saw.


def test_step0_banner_names_every_setting_with_source():
    driver = _driver()
    settings = _settings({"train_audio": True})
    driver.apply_settings(settings)
    banner = driver.step0_banner(u_seed=1234)
    src = settings.sources
    expected = (
        f"h3_settings video_shift={settings.sigma_shift_video} ({src['sigma_shift_video']}) "
        f"audio_shift={settings.sigma_shift_audio} ({src['sigma_shift_audio']}) "
        f"audio_loss_weight={settings.audio_loss_weight} ({src['audio_loss_weight']}) "
        f"cfg_augment_scale={settings.cfg_augment_scale} ({src['cfg_augment_scale']}) "
        f"train_audio={'true' if settings.train_audio else 'false'} ({src['train_audio']}) "
        "u_seed=1234"
    )
    assert banner == expected, f"banner != expected:\n{banner}\n{expected}"
    assert "train_audio=true (config)" in banner
    # Every value is followed by its provenance; an unsourced value is refused.
    for token in banner.split(" ")[1:-1]:
        if "=" in token:
            continue
        assert token.startswith("(") and token.endswith(")"), f"unsourced value before {token!r}"
    assert banner.count("(") == 5, "a value printed without its source"
    # Audio off: the weight is 0.0 and says WHY; unseeded runs say so.
    off = _driver()
    off.apply_settings(_settings({"train_audio": False}))
    banner_off = off.step0_banner(u_seed=None)
    assert "audio_loss_weight=0.0 (train_audio_off)" in banner_off
    assert "train_audio=false (config)" in banner_off
    assert banner_off.endswith("u_seed=unseeded")


def test_train_audio_off_keeps_audio_rows_packed(build_tiny_transformer):
    import torch

    driver = _cfg_driver(build_tiny_transformer, 1.0)
    driver.apply_settings(_settings({"train_audio": False, "cfg_augment_scale": 1.0}))
    assert driver.settings.audio_loss_weight == 0.0
    assert driver.settings.sources["audio_loss_weight"] == "train_audio_off"
    a = torch.randn(2, 32, 3)
    extra = driver.build_batch_extra([{"id": "x", "audio_latents": a}])
    assert "audio_clean" in extra and torch.equal(extra["audio_clean"][0], a), (
        "audio_rows == 0; H3 is single-stream — train_audio=false dropped the audio rows at the batch seam"
    )
    video, audio, t_v, cond, _ = _cfg_inputs()
    with torch.no_grad():
        v, a_vel = driver.forward_pass(video, t_v, cond, {"audio_noisy": audio})
    audio_rows = 0 if a_vel is None else int(a_vel.shape[-1])
    assert audio_rows == audio.shape[-1], "audio_rows == 0; H3 is single-stream"
    # The loss: the packed audio rows cost nothing — weight 0.0, loss == video.
    out = driver.compute_loss(v, torch.zeros_like(v), {}, audio_pred=a_vel, audio_target=torch.zeros_like(a_vel))
    assert torch.equal(out.loss, out.loss_video)


def test_build_batch_extra_stacks_audio_with_a_presence_mask():
    import torch

    driver = _driver()
    driver.apply_settings(_settings())
    a = torch.randn(2, 32, 3)
    items = [{"id": "x", "audio_latents": a}, {"id": "y"}]
    extra = driver.build_batch_extra(items)
    assert extra["audio_clean"].shape == (2, 2, 32, 3)
    assert torch.equal(extra["audio_clean"][0], a) and torch.all(extra["audio_clean"][1] == 0)
    assert extra["audio_mask"].tolist() == [1.0, 0.0]
    assert driver.build_batch_extra([{"id": "y"}]) == {}
    # `train_audio=false` does NOT unpack the rows (row 3.2; H3 is single-stream).
    driver.apply_settings(_settings({"train_audio": False}))
    assert driver.build_batch_extra(items)["audio_mask"].tolist() == [1.0, 0.0]


# ── Audio latent count: ONE formula, the diffusers reference (GATE-0 finding) ──
#
# The audio VAE pads a clip up to whole 800-sample latents (5 frames -> 6667
# samples -> 9 latents) while the reference trains AND samples on
# `round(frames / fps * 40)` (= 8): ai-toolkit `_fit_audio_rows`, diffusers
# `modular_pipeline.audio_latent_num_frames`. The cached latents are fitted
# to that count at the batch seam, and the sampler draws its noise from the
# same function.


def _frame_ladder(limit: int = 107) -> list[int]:
    return [17 * n + 5 for n in range((limit - 5) // 17 + 1)]


def test_audio_latent_num_frames_matches_the_diffusers_reference():
    import pytest

    reference = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.modular_pipeline")
    from app.engine.models.families.minimax_h3.packing import audio_latent_num_frames

    for frames in [1, *_frame_ladder()]:
        assert audio_latent_num_frames(frames) == reference.audio_latent_num_frames(frames), frames
    assert audio_latent_num_frames(5) == 8  # the VAE returns 9 for the same clip


def test_build_batch_extra_fits_audio_rows_to_the_reference_count():
    import torch

    driver = _driver()
    driver.apply_settings(_settings())
    long = torch.randn(2, 32, 9)  # what the audio VAE returns for 5 frames
    short = torch.randn(2, 32, 6)
    items = [
        {"id": "long", "audio_latents": long, "target_frames": 5},
        {"id": "short", "audio_latents": short, "target_frames": 5},
    ]
    extra = driver.build_batch_extra(items)
    assert extra["audio_clean"].shape == (2, 2, 32, 8), "fitted to round(5/24*40) = 8, not the VAE's 9"
    assert torch.equal(extra["audio_clean"][0], long[..., :8]), "the tail latent is cropped"
    assert torch.equal(extra["audio_clean"][1, ..., :6], short) and torch.all(extra["audio_clean"][1, ..., 6:] == 0), (
        "a short clip is zero-padded to the reference count"
    )
    assert extra["audio_mask"].tolist() == [1.0, 1.0]
    # No frame count on the item (a still): nothing to fit against, rows kept.
    assert driver.build_batch_extra([{"id": "raw", "audio_latents": long}])["audio_clean"].shape[-1] == 9


def test_sampler_draws_audio_noise_from_the_same_formula(monkeypatch):
    from types import SimpleNamespace

    from app.engine.models.families.minimax_h3 import packing, sampler as sampler_mod
    from app.engine.models.families.minimax_h3.sampler import MiniMaxH3Sampler

    definition = _driver().definition
    shell = MiniMaxH3Sampler.__new__(MiniMaxH3Sampler)
    shell.pipeline = SimpleNamespace(definition=definition)
    assert shell._audio_latents_for(5) == packing.audio_latent_num_frames(5) == 8
    monkeypatch.setattr(sampler_mod, "audio_latent_num_frames", lambda *a, **k: 4242, raising=False)
    monkeypatch.setattr(packing, "audio_latent_num_frames", lambda *a, **k: 4242)
    assert shell._audio_latents_for(107) == 4242, "the sampler has its own copy of the formula"


def _loss_inputs():
    import torch

    g = torch.Generator().manual_seed(5)
    vp, vt = torch.randn(2, 24, 2, 6, 4, generator=g), torch.randn(2, 24, 2, 6, 4, generator=g)
    ap, at = torch.randn(2, 2, 32, 3, generator=g), torch.randn(2, 2, 32, 3, generator=g)
    return vp, vt, ap, at


def test_compute_loss_reports_three_numbers():
    import torch

    driver = _driver()
    driver.apply_settings(_settings({"audio_loss_weight": 0.1}))
    vp, vt, ap, at = _loss_inputs()
    out = driver.compute_loss(vp, vt, {}, audio_pred=ap, audio_target=at)
    assert hasattr(out, "loss_video") and hasattr(out, "loss_audio"), "loss_video/loss_audio missing"
    assert out.loss_video.ndim == 0 and out.loss_audio.ndim == 0 and out.loss.ndim == 0
    assert torch.allclose(out.loss_video, torch.nn.functional.mse_loss(vp, vt))
    assert torch.allclose(out.loss_audio, torch.nn.functional.mse_loss(ap, at))
    assert torch.allclose(out.loss, out.loss_video + 0.1 * out.loss_audio)


def test_compute_loss_takes_the_audio_mask_from_the_host():
    """GATE-1 finding (plan row 2.12): `build_batch_extra` builds `audio_mask`
    on the host (the cached rows are CPU tensors), `forward_pass` moves only
    `audio_clean` to the card, and the loss met a CUDA prediction with a CPU
    mask — "Expected all tensors to be on the same device". The loss owns the
    mask's device. CUDA-only by nature: a CPU box cannot have two devices."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a second device to reproduce the mismatch")
    driver = _driver()
    driver.apply_settings(_settings({"audio_loss_weight": 0.1}))
    vp, vt, ap, at = (x.cuda() for x in _loss_inputs())
    out = driver.compute_loss(vp, vt, {}, audio_pred=ap, audio_target=at, audio_mask=torch.ones(ap.shape[0]))
    assert out.loss.device.type == "cuda"
    assert torch.allclose(out.loss_audio.cpu(), torch.nn.functional.mse_loss(ap, at).cpu())


def test_audio_loss_weight_scales_only_the_audio_term():
    import torch

    vp, vt, ap, at = _loss_inputs()
    low = _driver()
    low.apply_settings(_settings({"audio_loss_weight": 0.1}))
    high = _driver()
    high.apply_settings(_settings({"audio_loss_weight": 0.5}))
    a = low.compute_loss(vp, vt, {}, audio_pred=ap, audio_target=at)
    b = high.compute_loss(vp, vt, {}, audio_pred=ap, audio_target=at)
    assert torch.allclose(a.loss_video, b.loss_video) and torch.allclose(a.loss_audio, b.loss_audio)
    assert torch.allclose(b.loss - a.loss, 0.4 * a.loss_audio, atol=1e-6), (
        f"delta loss {float(b.loss - a.loss):.6f} != 0.4 * loss_audio {float(0.4 * a.loss_audio):.6f}"
    )


def test_train_audio_off_gives_video_loss_only():
    import torch

    driver = _driver()
    driver.apply_settings(_settings({"train_audio": False}))
    vp, vt, ap, at = _loss_inputs()
    out = driver.compute_loss(vp, vt, {}, audio_pred=ap, audio_target=at)
    assert torch.equal(out.loss, out.loss_video)
    assert float(out.loss_audio) == 0.0
    # And absent audio tensors with audio ON are refused, never silently zero.
    on = _driver()
    on.apply_settings(_settings())
    import pytest

    with pytest.raises(ValueError, match="audio"):
        on.compute_loss(vp, vt, {})


# ── The PR0-refusal guard, final shape (plan row 2.8) ──────────────────────
#
# Rows 2.4 (forward_pass), 2.5 (_setup_family) and 2.8 (get_saver) retired
# their refusals one by one; nothing in driver.py / trainer.py may raise a
# NotImplementedError (directly or through `_lands_in_pr1`) and no
# `*refuses_loudly_in_pr0` test may remain in this file. The positive control
# below proves the scanner still SEES a refusal.

_FAMILY_DIR = Path(__file__).resolve().parents[1] / "models" / "families" / "minimax_h3"
_REFUSAL_CONTROL = Path(__file__).resolve().parent / "fixtures" / "h3_refusal_control.py"


def _refusal_sites(path: Path, label: str) -> set[tuple[str, str]]:
    """Every function in ``path`` that RAISES a ``NotImplementedError`` (directly
    or through the ``_lands_in_pr1`` helper) — code only; docstrings and
    comments are never scanned (the row 1.2 lesson)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            src = ast.unparse(node.exc)
            if "NotImplementedError" in src or "_lands_in_pr1" in src:
                found.add((label, fn.name))
    return found


def _refusal_tests(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        ("DRV", fn.name)
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef) and "refuses_loudly_in_pr0" in fn.name
    }


def test_no_pr0_refusal_remains():
    found = sorted(
        _refusal_sites(_FAMILY_DIR / "driver.py", "driver.py")
        | _refusal_sites(_FAMILY_DIR / "trainer.py", "trainer.py")
        | _refusal_tests(Path(__file__))
    )
    assert not found, "PR0 refusal still present: " + ", ".join(f"{f}:{n}" for f, n in found)


def test_refusal_scanner_flags_the_positive_control():
    """The scanner must SEE a refusal: the control fixture raises one directly,
    one through the helper shape, and one inside a method; prose is ignored."""
    found = _refusal_sites(_REFUSAL_CONTROL, "h3_refusal_control.py")
    assert found == {
        ("h3_refusal_control.py", "direct_refusal"),
        ("h3_refusal_control.py", "helper_refusal"),
        ("h3_refusal_control.py", "still_refuses"),
    }, found


# ── The LoRA saver (plan row 2.8): the ORIGINAL-checkpoint layout ──────────
#
# Production caller: `MiniMaxH3Driver.get_saver()` → `pipeline_optimization.py`
# `CheckpointManager(saver_impl=…)` → `saver.save(components, path, metadata)`.
# The artifact is what ComfyUI and ai-toolkit consume and what diffusers 0.40
# converts (`lora_conversion_utils.py:3122`): `diffusion_model.` prefix,
# native module names, `lora_A`/`lora_B`, PEFT scaling folded into `lora_B`,
# no `.alpha` key. Reserved as a public id (ECOSYSTEM §6, REQUEST-16).

_SAVER_PY = Path(__file__).resolve().parents[1] / "models" / "families" / "minimax_h3" / "saver.py"


def _peft_model(build_tiny_transformer, *, r: int = 2, alpha: float = 2.0):
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        build_tiny_transformer(), LoraConfig(r=r, lora_alpha=alpha, target_modules=_driver().get_lora_targets())
    )
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:  # PEFT zero-inits B: randomise so every delta is non-zero
                p.copy_(torch.randn(p.shape, generator=g))
    return model.eval()


def _read_artifact(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        meta = dict(fh.metadata() or {})
    return tensors, meta


def _assert_original_layout(tensors: dict) -> None:
    assert tensors, "no artifact written"
    bad = [k for k in tensors if not k.startswith("diffusion_model.")]
    assert not bad, f"keys outside the diffusion_model. namespace: {bad[:3]}"
    assert any(k.startswith("diffusion_model.blocks.") for k in tensors), "no key starts with diffusion_model.blocks."
    assert not any(k.endswith(".alpha") for k in tensors), "an .alpha key was written (scaling must be folded)"
    assert all(k.endswith((".lora_A.weight", ".lora_B.weight")) for k in tensors)


def test_saver_writes_original_layout_with_shared_metadata(tmp_path, build_tiny_transformer):
    import torch

    saver = _driver().get_saver()
    config = {
        "global_triggerword": "sks",
        "save_precision": "fp32",
        "lora_name": "t",
        "optimizer_type": "adamw",
        "learning_rate": 1e-4,
    }
    out = tmp_path / "h3.safetensors"
    saver.save({"unet": _peft_model(build_tiny_transformer), "config": config}, out, metadata={"step": 3})
    tensors, meta = _read_artifact(out)
    _assert_original_layout(tensors)
    assert all(t.dtype is torch.float32 for t in tensors.values())
    # Shared metadata: BOTH trigger keys from the one helper, the Kohya ss_*
    # map, the rank, and the manager's own fields (stringified).
    assert meta.get("ss_training_comment") == "sks" and meta.get("modelspec.trigger_phrase") == "sks", meta
    assert meta.get("ss_optimizer") == "adamw" and meta.get("ss_learning_rate") == "0.0001"
    assert meta.get("ss_network_dim") == "2" and meta.get("ss_network_alpha") == "2.0"
    assert meta.get("step") == "3" and meta.get("modelspec.architecture")


def test_saver_omits_trigger_key_without_trigger(tmp_path, build_tiny_transformer):
    saver = _driver().get_saver()
    out = tmp_path / "h3.safetensors"
    saver.save({"unet": _peft_model(build_tiny_transformer), "config": {"save_precision": "fp32"}}, out, metadata={})
    _, meta = _read_artifact(out)
    assert "ss_training_comment" not in meta and "modelspec.trigger_phrase" not in meta, meta


def test_saver_imports_lora_metadata_helper():
    """The ss_* map and the trigger keys are IMPORTED from the shared helper,
    never re-typed in the family (the ltx2 lesson: a copied map drifts)."""
    tree = ast.parse(_SAVER_PY.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "app.engine.utils.lora_metadata"
        for alias in node.names
    }
    assert {"trigger_metadata", "kohya_config_metadata"} <= imported, imported
    local_ss = sorted(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("ss_")
    )
    assert not local_ss, f"saver.py re-types Kohya keys locally: {local_ss}"


def test_checkpoint_manager_saves_through_the_family_saver(tmp_path, build_tiny_transformer):
    """The production path: the manager built the way `pipeline_optimization.py`
    builds it, one save, the file read back."""
    from app.engine.components.checkpoints import CheckpointManager

    manager = CheckpointManager(output_dir=str(tmp_path / "out"), saver_impl=_driver().get_saver())
    manager.save_checkpoint(
        step=1,
        components={"unet": _peft_model(build_tiny_transformer)},
        config={"lora_name": "t", "save_precision": "fp32"},
    )
    assert manager.last_lora_path, "no artifact written"
    tensors, _ = _read_artifact(manager.last_lora_path)
    _assert_original_layout(tensors)
