"""Bernini-R v2v in-training sampler contract tests.

Pins the three make-or-break invariants of the preview denoise loop:

* **Frozen condition tokens** — the condition latent is VAE-encoded ONCE before
  the loop and reused by reference every step; the packed condition-token slice
  is bit-identical across all scheduler steps (and every CFG pass), while the
  target-token slice moves as UniPC steps the target.
* **CFG variant #5 (v2v)** — ``pred = eps_uncond + omega * (eps_cond -
  eps_uncond)`` where the CONDITION VIDEO rides in BOTH branches and only the
  TEXT swaps (empty-string negative). Pinned to upstream
  ``GEN_Wanx22.sample()`` ``guidance_mode == "v2v"``; NOT the chroma/lumina2/
  nucleus/ace combine.
* **No-control fallback** — with no control video the sampler runs the
  degenerate stock-t2v packed path and never raises (previews stay non-fatal).
"""

from __future__ import annotations

import torch
from diffusers import WanTransformer3DModel

from app.engine.models.families.bernini_r.sampler import BerniniRSampler
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer


class _Defn:
    """Minimal definition stand-in (flow_shift pin only)."""

    architecture_params = {"scheduler.flow_shift": 5.0, "mode": "t2v"}


def _tiny_model(seed: int = 0) -> WanTransformer3DModel:
    """Tiny stock 16-channel WanTransformer3DModel (fp32, CPU), text_dim=16."""
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


class _Driver:
    def __init__(self, model, vae=None):
        self._m = model
        self.vae = vae

    def get_primary_model(self):
        return self._m


class _Pipeline:
    """Minimal trainer stand-in the sampler binds to."""

    def __init__(self, model, vae=None, emb=None):
        self.config: dict = {}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _Driver(model, vae)
        self.definition = _Defn()
        self._emb = emb if emb is not None else torch.randn(1, 5, 16)

    # encode_prompt() (inherited) routes the empty-negative through here.
    def encode_text(self, caps, dtype):
        return self._emb


def _make_sampler(model, vae=None, emb=None) -> BerniniRSampler:
    return BerniniRSampler(_Pipeline(model, vae, emb))


class _RecordingBlock(torch.nn.Module):
    """Wrap a transformer block to capture the packed ``hidden_states`` it sees."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[torch.Tensor] = []

    def forward(self, hidden_states, *args, **kwargs):
        self.seen.append(hidden_states.detach().clone())
        return self.inner(hidden_states, *args, **kwargs)


# ── Frozen condition tokens ──────────────────────────────────────────────────


class TestFrozenConditionTokens:
    def test_condition_slice_is_frozen_across_steps(self, monkeypatch):
        """The packed condition-token slice is bit-identical across every step
        (and CFG pass); the target-token slice moves as UniPC steps the target.
        """
        model = _tiny_model()
        rec = _RecordingBlock(model.blocks[0])
        model.blocks[0] = rec

        emb = torch.randn(1, 5, 16)
        sampler = _make_sampler(model, emb=emb)
        sampler._active_prompt_cfg = {"control_images": ["dummy.mp4"]}

        # One condition stream, VAE-encoded ONCE (stub the encode → fixed latent).
        cond = torch.randn(1, 16, 1, 8, 8)
        monkeypatch.setattr(sampler, "_encode_control_video", lambda path, target: cond)

        noise = torch.randn(1, 16, 1, 8, 8)
        sampler.denoise(noise, emb, num_steps=3, guidance_scale=4.0, seed=0)

        # cond latent [1,16,1,8,8] patch (1,2,2) → 1*4*4 = 16 condition tokens,
        # concatenated BEFORE the 16 target tokens.
        cond_total = 16
        recs = rec.seen
        assert len(recs) == 6, "3 steps × (cond + uncond) CFG passes"

        first_cond = recs[0][:, :cond_total]
        for h in recs:
            assert torch.equal(h[:, :cond_total], first_cond), (
                "condition tokens must be frozen across steps/CFG passes"
            )

        # The target-token slice must actually move between the first and last
        # step (else the frozen assertion would be vacuous).
        assert not torch.equal(recs[0][:, cond_total:], recs[-1][:, cond_total:]), (
            "target tokens must change as the scheduler steps"
        )


# ── CFG variant #5 (v2v) ─────────────────────────────────────────────────────


class TestCfgVariant5:
    def test_v2v_formula_and_both_passes_carry_condition_video(self, monkeypatch):
        """pred = eps_uncond + omega*(eps_cond - eps_uncond); BOTH passes carry
        the condition video, only the text embedding swaps.
        """
        import app.engine.models.families.bernini_r.sampler as smod

        sampler = _make_sampler(_tiny_model())

        cond = torch.randn(1, 16, 1, 8, 8)
        target = torch.randn(1, 16, 1, 8, 8)
        ts = torch.tensor([500.0])
        text_cond = torch.randn(1, 5, 16)
        text_uncond = torch.randn(1, 5, 16)
        eps_cond = torch.randn(1, 16, 1, 8, 8)
        eps_uncond = torch.randn(1, 16, 1, 8, 8)

        seen: list[dict] = []

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
            is_cond = encoder_hidden_states is text_cond
            seen.append({"cond": list(cond_latents), "is_cond": is_cond})
            return (eps_cond if is_cond else eps_uncond,)

        monkeypatch.setattr(smod, "bernini_packed_forward", _spy)

        out = sampler._cfg_velocity(
            None,
            [cond],
            [1.0],
            target,
            ts,
            text_cond,
            text_uncond,
            4.0,
            torch.bfloat16,
            "cpu",
        )

        expected = eps_uncond + 4.0 * (eps_cond - eps_uncond)
        assert torch.equal(out, expected)

        # Exactly one cond-text pass and one uncond-text pass, BOTH carrying the
        # condition video (v2v keeps the source video in the uncond branch).
        assert len(seen) == 2
        assert sum(s["is_cond"] for s in seen) == 1
        assert all(
            len(s["cond"]) == 1 and torch.equal(s["cond"][0], cond) for s in seen
        )

    def test_cfg_off_is_single_conditional_velocity(self, monkeypatch):
        """guidance_scale <= 1 (text_uncond None) → the single conditional pass."""
        import app.engine.models.families.bernini_r.sampler as smod

        sampler = _make_sampler(_tiny_model())
        target = torch.randn(1, 16, 1, 8, 8)
        text_cond = torch.randn(1, 5, 16)
        eps_cond = torch.randn(1, 16, 1, 8, 8)

        calls = {"n": 0}

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
            calls["n"] += 1
            return (eps_cond,)

        monkeypatch.setattr(smod, "bernini_packed_forward", _spy)

        out = sampler._cfg_velocity(
            None,
            [],
            [],
            target,
            torch.tensor([500.0]),
            text_cond,
            None,
            1.0,
            torch.bfloat16,
            "cpu",
        )
        assert calls["n"] == 1  # no uncond pass
        assert torch.equal(out, eps_cond.to(torch.float32))


# ── No-control fallback ──────────────────────────────────────────────────────


class TestNoControlFallback:
    def test_no_control_runs_degenerate_t2v_without_raising(self):
        """No control video ⇒ the degenerate stock-t2v packed path; a preview
        never crashes for want of a control input.
        """
        model = _tiny_model()
        emb = torch.randn(1, 5, 16)
        sampler = _make_sampler(model, emb=emb)
        sampler._active_prompt_cfg = {}  # no control_images

        noise = torch.randn(1, 16, 1, 8, 8)
        out = sampler.denoise(noise, emb, num_steps=2, guidance_scale=4.0, seed=0)

        assert out.shape == noise.shape


# ── Cross-round control-latent memo ──────────────────────────────────────────


class TestControlLatentMemo:
    """A run's preview config is fixed, so the clean control latent is
    bit-identical across sampling rounds — the decode + fp32 Wan-VAE encode
    must run ONCE per (path, target shape, fps), not once per round."""

    @staticmethod
    def _counting_encode(calls: dict):
        def _fake(path, target):
            calls["n"] += 1
            return torch.randn(1, 16, 1, 8, 8)

        return _fake

    def test_encode_runs_once_across_rounds(self, monkeypatch):
        sampler = _make_sampler(_tiny_model())
        sampler._active_prompt_cfg = {"control_images": ["clip.mp4"]}
        calls = {"n": 0}
        monkeypatch.setattr(
            sampler, "_encode_control_video", self._counting_encode(calls)
        )

        target = torch.randn(1, 16, 1, 8, 8)
        round1, _ = sampler._build_condition_streams(target)
        round2, _ = sampler._build_condition_streams(target)

        assert calls["n"] == 1, "same path/shape/fps must be served from the memo"
        assert torch.equal(round1[0], round2[0])

    def test_changed_target_shape_reencodes(self, monkeypatch):
        """A changed preview geometry must NOT be served a stale latent."""
        sampler = _make_sampler(_tiny_model())
        sampler._active_prompt_cfg = {"control_images": ["clip.mp4"]}
        calls = {"n": 0}
        monkeypatch.setattr(
            sampler, "_encode_control_video", self._counting_encode(calls)
        )

        sampler._build_condition_streams(torch.randn(1, 16, 1, 8, 8))
        sampler._build_condition_streams(torch.randn(1, 16, 2, 8, 8))

        assert calls["n"] == 2


# ── _create_sampler wiring (F7 — house convention) ────────────────────────


class TestCreateSamplerWiring:
    """The trainer builds a BerniniRSampler only when sampling is configured
    (``sample_every_n_steps > 0``), mirroring wan21/wan22/chroma/boogu. The base
    sampler __init__ reads only ``pipeline.config`` + ``pipeline.device``."""

    @staticmethod
    def _bare(config: dict) -> BerniniRTrainer:
        t = BerniniRTrainer.__new__(BerniniRTrainer)
        t.config = config
        t.device = torch.device("cpu")
        return t

    def test_create_sampler_returns_bernini_sampler_when_enabled(self):
        t = self._bare({"sample_every_n_steps": 50})
        assert isinstance(t._create_sampler(), BerniniRSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        t = self._bare({"sample_every_n_steps": 0})
        assert t._create_sampler() is None
