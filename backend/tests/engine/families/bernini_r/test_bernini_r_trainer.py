"""Bernini-R trainer contract tests.

Pins the three training-recipe guarantees BR3 adds on top of the BR2 driver:

* **Timestep sampling** — the upstream ``NoiseScheduler`` SD3 ``mode`` weighting
  followed by the per-task shift-warp, producing RAW ``[0, 1000]`` timesteps.
  Pinned two ways: a DETERMINISTIC fixed-seed transcription pin (catches any
  drift in the ``cos**2`` term / shift-warp / ``*1000`` scale) and a statistical
  distribution check over ~50k draws.
* **Loss masking** — the model's prediction for the CONDITION tokens is sliced
  off inside the packed forward and can never reach the loss. Proven by mutating
  the condition-region of the transformer's own ``proj_out`` output and asserting
  the returned (target-only) prediction is byte-unchanged, then mutating the
  target region and asserting it DOES change.
* **Condition cleanliness** — the latents handed to the driver's condition path
  are bit-identical to ``batch['control_latents']`` (no noise, no timestep warp).

The timestep tests build a bare ``BerniniRTrainer`` via ``object.__new__`` (the
wan22 wiring-test precedent) so no real weights / loader are touched; the
loss-mask + cleanliness tests drive the REAL ``BerniniRDriver`` over a tiny stock
``WanTransformer3DModel``.
"""

from __future__ import annotations

import math

import structlog
import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer


# ── Bare trainer + tiny model helpers ────────────────────────────────────────


def _bare_trainer(config: dict | None = None) -> BerniniRTrainer:
    """A ``BerniniRTrainer`` shell with only ``device`` + ``config`` set.

    Mirrors ``test_wan22_sample_timesteps_wiring``'s ``object.__new__`` stub:
    ``sample_timesteps`` reads only ``self.device`` and ``self.config``.
    """
    t = object.__new__(BerniniRTrainer)
    t.device = torch.device("cpu")
    t.config = config if config is not None else {}
    return t


class _Defn:
    architecture_params = {"mode": "t2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


def _tiny_model(seed: int = 0) -> WanTransformer3DModel:
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        out_channels=16,
        text_dim=16,
        freq_dim=64,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )
    return model.to(torch.float32).eval()


def _make_driver(model: WanTransformer3DModel) -> BerniniRDriver:
    drv = BerniniRDriver(_Defn(), torch.device("cpu"))
    drv.assign_components({"unet": model})
    return drv


def _tokens(latent: torch.Tensor, model: WanTransformer3DModel) -> int:
    """Number of patch tokens a stream contributes (F/pt * H/ph * W/pw)."""
    p_t, p_h, p_w = model.config.patch_size
    return (
        (latent.shape[2] // p_t) * (latent.shape[3] // p_h) * (latent.shape[4] // p_w)
    )


def _reference_timesteps(
    raw: torch.Tensor, mode_scale: float, shift: float
) -> torch.Tensor:
    """The upstream ``NoiseScheduler`` transform, written longhand (verbatim)."""
    u = 1.0 - raw - mode_scale * (torch.cos(math.pi * raw / 2.0) ** 2 - 1.0 + raw)
    sigmas = shift * u / (1.0 + (shift - 1.0) * u)
    return sigmas * 1000.0


# ── Timestep sampling ─────────────────────────────────────────────────────────


class TestSampleTimesteps:
    def test_family_defaults_are_upstream_constants(self):
        """v2v shift = 5.0 (upstream ``shift_config``); mode_scale = 1.29 (SD3)."""
        assert BerniniRTrainer.DEFAULT_TIMESTEP_SHIFT == 5.0
        assert BerniniRTrainer.DEFAULT_MODE_SCALE == 1.29

    def test_deterministic_formula_pin(self):
        """A fixed-seed draw matches the longhand transform EXACTLY.

        This is the make-or-break transcription pin: it fails if the ``cos**2``
        term, its sign, the shift-warp, or the ``*1000`` scale drift at all.
        """
        n = 4096
        trainer = _bare_trainer({})  # empty config → family defaults

        torch.manual_seed(1234)
        got = trainer.sample_timesteps(n)

        torch.manual_seed(1234)
        raw = torch.rand(n, device=trainer.device)
        expected = _reference_timesteps(raw, mode_scale=1.29, shift=5.0)

        assert got.shape == (n,)
        assert torch.allclose(got, expected, atol=1e-6, rtol=0.0)

    def test_config_overrides_shift_and_mode_scale(self):
        """Explicit config values feed the same formula (r2v shift 4.0 etc.)."""
        n = 2048
        cfg = {"timestep_shift": 4.0, "mode_scale": 1.10}
        trainer = _bare_trainer(cfg)

        torch.manual_seed(7)
        got = trainer.sample_timesteps(n)

        torch.manual_seed(7)
        raw = torch.rand(n, device=trainer.device)
        expected = _reference_timesteps(raw, mode_scale=1.10, shift=4.0)
        assert torch.allclose(got, expected, atol=1e-6, rtol=0.0)

    def test_distribution_bounds_and_shape(self):
        """~50k draws stay in [0, 1000] and match an independent formula sample."""
        n = 50_000
        trainer = _bare_trainer({})

        torch.manual_seed(99)
        ts = trainer.sample_timesteps(n)
        assert ts.shape == (n,)
        assert float(ts.min()) >= 0.0
        assert float(ts.max()) <= 1000.0

        # Independent reference draw through the exact transform — quantiles of
        # the two large samples must agree (both are the SAME distribution).
        raw = torch.rand(n)
        ref = _reference_timesteps(raw, mode_scale=1.29, shift=5.0)
        qs = torch.linspace(0.05, 0.95, 19)
        got_q = torch.quantile(ts, qs)
        ref_q = torch.quantile(ref, qs)
        assert torch.allclose(got_q, ref_q, atol=15.0)

    def test_shift_biases_toward_high_noise(self):
        """shift = 5.0 (v2v) pushes mass to higher t than shift = 1.0 (identity)."""
        n = 40_000
        torch.manual_seed(3)
        hi = _bare_trainer({"timestep_shift": 5.0}).sample_timesteps(n)
        torch.manual_seed(3)
        lo = _bare_trainer({"timestep_shift": 1.0}).sample_timesteps(n)
        assert float(hi.mean()) > float(lo.mean())


# ── Loss masking: condition-token predictions can never reach the loss ─────────


class _MutatingProjOut(torch.nn.Module):
    """Wrap ``proj_out`` and add ``delta`` to a token-slice of its OUTPUT.

    ``proj_out`` runs over the FULL packed ``[cond..., target]`` sequence just
    before the target-tail slice. Perturbing the condition rows here is exactly a
    perturbation of the model's condition-token prediction — which the packed
    forward must discard.
    """

    def __init__(self, inner, token_slice, delta):
        super().__init__()
        self.inner = inner
        self.token_slice = token_slice
        self.delta = delta

    def forward(self, x):
        out = self.inner(x)
        out = out.clone()
        out[:, self.token_slice, :] = out[:, self.token_slice, :] + self.delta
        return out


class TestLossMasking:
    def _run(self, model, target, text, batch):
        drv = _make_driver(model)
        with torch.no_grad():
            return drv.forward_pass(target, torch.tensor([500.0]), text, batch)

    def test_condition_prediction_cannot_leak_into_loss(self):
        target = torch.randn(1, 16, 1, 8, 8)
        control = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)
        batch = {BerniniRDriver.BATCH_CONTROL_LATENTS: [control]}

        base_model = _tiny_model()
        cond_total = _tokens(control, base_model)  # rows [0:cond_total] = condition
        pred_base = self._run(base_model, target, text, batch)

        # Fixed velocity target — the loss the trainer's _compute_step_loss forms.
        velocity_target = torch.randn_like(pred_base)
        loss_base = F.mse_loss(pred_base.float(), velocity_target.float())

        # (a) Mutate the CONDITION rows of proj_out's output → sliced off.
        m_cond = _tiny_model()
        m_cond.proj_out = _MutatingProjOut(
            m_cond.proj_out, slice(0, cond_total), delta=1000.0
        )
        pred_cond = self._run(m_cond, target, text, batch)
        loss_cond = F.mse_loss(pred_cond.float(), velocity_target.float())

        assert torch.equal(pred_base, pred_cond), (
            "condition-token prediction leaked into the target output"
        )
        assert torch.equal(loss_base, loss_cond), "loss changed on condition mutation"

        # (b) Mutate the TARGET rows → the loss MUST move (sanity: the harness
        # actually perturbs something reachable).
        m_tgt = _tiny_model()
        total = cond_total + _tokens(target, m_tgt)
        m_tgt.proj_out = _MutatingProjOut(
            m_tgt.proj_out, slice(cond_total, total), delta=1000.0
        )
        pred_tgt = self._run(m_tgt, target, text, batch)
        loss_tgt = F.mse_loss(pred_tgt.float(), velocity_target.float())

        assert not torch.equal(pred_base, pred_tgt)
        assert not torch.equal(loss_base, loss_tgt)


# ── Condition cleanliness: control latents reach the driver untouched ──────────


class _RecordingLogger:
    def __init__(self):
        self.warnings: list[tuple[str, dict]] = []
        self.infos: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, event, **kw):
        self.infos.append((event, kw))


class _FakeUMT5Driver:
    """Minimal driver exposing a resident TE + a tensor-returning encode_text."""

    def __init__(self):
        self.text_encoder = object()  # truthy → warm path is active
        self.encoded: list[str] = []

    def encode_text(self, chunk, dtype):
        self.encoded.extend(chunk)
        return torch.zeros(len(chunk), 3, 4)  # [B, L, D] raw tensor


def _warm_trainer(config: dict) -> BerniniRTrainer:
    """A BerniniRTrainer shell wired for the text-cache warm path only."""
    t = object.__new__(BerniniRTrainer)
    t.device = torch.device("cpu")
    t.config = config
    t.text_cache = {}
    t.driver = _FakeUMT5Driver()
    t.logger = _RecordingLogger()
    # Stub the base-mixin collaborators so no disk / real model is touched.
    t._resolve_te_cache_dirs = lambda: []  # te1_dir = "" → memory-only warm
    t._resolve_loading_dtype = lambda: torch.float32
    t._build_caption_hints = lambda: {}
    t._sample_prompt_texts = lambda: ["a cat"]
    return t


class TestPreCacheAlwaysWarmsEmptyNegative:
    """F5 — Bernini's CFG#5 always encodes the "" negative; the warm set must
    include it even when the user configured a non-empty sample_negative_prompt
    (else every preview's uncond pass hits the offloaded encoder → RuntimeError)."""

    def test_configured_negative_still_warms_empty_string(self):
        t = _warm_trainer(
            {
                "cache_text_embeddings": True,
                "sample_negative_prompt": "worst quality, blurry",
                "sample_prompts": [{"prompt": "a cat"}],
            }
        )
        t._pre_cache_text_embeddings()

        # The frozen "" negative is warmed …
        assert "" in t.text_cache
        # … alongside the configured one (warmed by the base under its own key) …
        assert "worst quality, blurry" in t.text_cache
        # … and a warn fired that the configured negative is ignored.
        assert any("negative" in ev for ev, _ in t.logger.warnings)

    def test_warn_is_once(self):
        cfg = {
            "cache_text_embeddings": True,
            "sample_negative_prompt": "blurry",
            "sample_prompts": [{"prompt": "a cat"}],
        }
        t = _warm_trainer(cfg)
        t._pre_cache_text_embeddings()
        t._pre_cache_text_embeddings()
        neg_warns = [ev for ev, _ in t.logger.warnings if "negative" in ev]
        assert len(neg_warns) == 1

    def test_no_configured_negative_still_has_empty_and_no_warn(self):
        t = _warm_trainer(
            {
                "cache_text_embeddings": True,
                "sample_prompts": [{"prompt": "a cat"}],
            }
        )
        t._pre_cache_text_embeddings()
        assert "" in t.text_cache  # base already warms it; override is a no-op
        assert not any("negative" in ev for ev, _ in t.logger.warnings)


class TestConditionCleanliness:
    def test_control_latents_passed_bit_identical(self, monkeypatch):
        import app.engine.models.families.bernini_r.driver as drv_mod

        captured: dict = {}

        def _spy(
            model,
            cond_latents,
            cond_source_ids,
            target_latent,
            timestep,
            encoder_hidden_states,
            return_dict=False,
            **kw,
        ):
            captured["cond"] = [c.clone() for c in cond_latents]
            return (torch.zeros_like(target_latent),)

        monkeypatch.setattr(drv_mod, "bernini_packed_forward", _spy)

        drv = _make_driver(_tiny_model())
        target = torch.randn(1, 16, 1, 8, 8)
        control = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)
        batch = {drv.BATCH_CONTROL_LATENTS: [control]}

        drv.forward_pass(target, torch.tensor([734.0]), text, batch)

        assert len(captured["cond"]) == 1
        # No noise, no warp — the condition stream is the clean control latent.
        assert torch.equal(captured["cond"][0], control)


# ── Router p_high under the REAL band distribution (Task W3.T3) ──────────────


class _DualDefn:
    """Minimal 14B dual-expert definition stand-in (no weights, no YAML)."""

    architecture_params = {
        "mode": "t2v",
        "te.max_length": 512,
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
        "moe.boundary_ratio": 0.875,
        "scheduler.num_train_timesteps": 1000,
    }
    lora_targetable_modules: list[str] = []


def _dual_trainer(mode_scale: float = 1.29, shift: float = 5.0) -> BerniniRTrainer:
    """A bare ``BerniniRTrainer`` wired with a REAL dual driver + a router built
    via the real ``_build_router`` — exercises the ``p_high`` estimate exactly
    as production setup does, with no weights / loader touched (mirrors
    ``_bare_trainer``/``_band_trainer`` above)."""
    t = object.__new__(BerniniRTrainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {
        # The REAL bernini_r_14b.yaml definition pins this — without it the
        # reused router's un-fixed default estimate falls back to
        # "logit_normal" (its own generic default), not even the unwarped
        # "mode" formula the brief compares against. Must match production.
        "timestep_sampling": "mode",
        "mode_scale": mode_scale,
        "timestep_shift": shift,
        "expert_switch_interval": 1,
        "expert_swap_mode": "resident",
        "expert_mode": "both",
        "seed": 0,
    }
    t.driver = BerniniRDriver(_DualDefn(), t.device)
    t.expert_mode = "both"
    t._build_router()
    return t


class TestRouterPHighUnderRealBandDistribution:
    def test_router_p_high_uses_bernini_shift_warp(self):
        """The reused wan22 ExpertRouter's step-selection ``p_high`` must
        reflect BERNINI's REAL per-step distribution (SD3 mode + shift-5
        warp) — not the generic ``TimestepSampler`` 'mode' formula (no shift
        warp at all), which estimates ``p_high ≈ 0.16`` against the real
        ``≈ 0.29`` and under-trains the high expert roughly 2x. Analytic
        check (see the module-level MC verification): P(t >= 875) ≈ 0.29
        under mode+shift5; ≈ 0.16 unwarped.
        """
        t = _dual_trainer(mode_scale=1.29, shift=5.0)
        router = t.expert_router
        assert 0.25 <= router.p_high <= 0.34, router.p_high

    def test_router_receives_bernini_own_sampler_not_generic(self):
        """Directly observe that the router's estimate is driven by
        `_mode_shift_timesteps`, not the generic TimestepSampler 'mode'
        formula — a different (config-driven) mode_scale/shift must move
        `p_high` even though `config['timestep_sampling']` never changes."""
        low_shift = _dual_trainer(mode_scale=1.29, shift=1.0).expert_router.p_high
        high_shift = _dual_trainer(mode_scale=1.29, shift=5.0).expert_router.p_high
        # A bigger shift pushes more mass toward high noise -> bigger p_high.
        assert high_shift > low_shift + 0.05, (low_shift, high_shift)

    def test_pinned_single_expert_router_unaffected(self):
        """Single-expert (`high`/`low`) runs build a PINNED router that skips
        the Monte-Carlo estimate entirely — passing `timestep_draw` must stay
        harmless (never invoked) for them."""
        t = object.__new__(BerniniRTrainer)
        t.logger = structlog.get_logger("test")
        t.device = torch.device("cpu")
        t.config = {
            "mode_scale": 1.29,
            "timestep_shift": 5.0,
            "expert_switch_interval": 1,
            "expert_swap_mode": "resident",
            "expert_mode": "high",
            "seed": 0,
        }
        t.driver = BerniniRDriver(_DualDefn(), t.device)
        t.expert_mode = "high"
        router = t._build_router()
        assert router.pinned_expert == "high"
        assert router.p_high == 1.0
