"""Bernini-R driver contract tests.

Pins the driver seam BR3/BR4 build on:
* ``prepare_latents`` lifts a 4D still to 5D (inherited Wan behaviour).
* ``forward_pass`` returns the velocity for the TARGET tokens only, shaped
  ``[B, 16, F, H, W]`` — condition streams do not widen the output.
* the transformer is conditioned on the RAW ``[0, 1000]`` timestep (the
  pure-noise gotcha), shared by all tokens.
* condition streams get ``source_id = slot + 1`` (target implicitly 0).
* the LoRA target set equals wan21's (Bernini adds zero new weight modules).
"""

from __future__ import annotations

import torch
from diffusers import WanTransformer3DModel

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.wan_shared.driver_base import WAN_T2V_LORA_TARGETS


class _Defn:
    """Minimal definition stand-in (no weights, no YAML)."""

    architecture_params = {"mode": "t2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


def _tiny_model(seed: int = 0) -> WanTransformer3DModel:
    """Tiny stock 16-channel WanTransformer3DModel (fp32, CPU)."""
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


class _RecordingCondEmbedder(torch.nn.Module):
    """Wrap a condition_embedder to capture the timestep it is called with."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen_timesteps: list[torch.Tensor] = []

    def forward(self, timestep, *args, **kwargs):
        self.seen_timesteps.append(timestep.detach().clone().float())
        return self.inner(timestep, *args, **kwargs)


# ── prepare_latents 5D lift ──────────────────────────────────────────────────


class TestPrepareLatents:
    def test_lifts_4d_still_to_5d(self):
        drv = _make_driver(_tiny_model())
        out = drv.prepare_latents(torch.randn(1, 16, 8, 8))
        assert out.shape == (1, 16, 1, 8, 8)

    def test_leaves_5d_unchanged(self):
        drv = _make_driver(_tiny_model())
        lat = torch.randn(1, 16, 3, 8, 8)
        assert drv.prepare_latents(lat).shape == lat.shape


# ── forward_pass output shape ────────────────────────────────────────────────


class TestForwardPassShape:
    def test_v2v_returns_target_shape(self):
        """One condition stream + noisy target → velocity shaped like the target
        (16 channels, target frame/space), NOT widened by the condition stream.
        """
        model = _tiny_model()
        drv = _make_driver(model)
        target = torch.randn(1, 16, 1, 8, 8)
        control = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)
        batch = {drv.BATCH_CONTROL_LATENTS: [control]}

        with torch.no_grad():
            out = drv.forward_pass(target, torch.tensor([500.0]), text, batch)

        assert out.shape == target.shape

    def test_no_control_degenerate_returns_target_shape(self):
        """No condition latents → stock-t2v degenerate path, still target-shaped."""
        model = _tiny_model()
        drv = _make_driver(model)
        target = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)

        with torch.no_grad():
            out = drv.forward_pass(target, torch.tensor([500.0]), text, {})

        assert out.shape == target.shape


# ── raw [0,1000] timestep pin ────────────────────────────────────────────────


class TestRawTimestep:
    def test_forward_feeds_raw_1000_scale_timestep(self):
        """The transformer's condition embedder must see the RAW [0,1000]
        timestep, not ``timesteps / 1000`` (frozen embedder → t≈0 → pure noise).
        """
        model = _tiny_model()
        rec = _RecordingCondEmbedder(model.condition_embedder)
        model.condition_embedder = rec
        drv = _make_driver(model)

        timesteps = torch.tensor([734.0])
        target = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)

        with torch.no_grad():
            drv.forward_pass(target, timesteps, text, {})

        seen = rec.seen_timesteps[0]
        assert torch.allclose(seen, timesteps, atol=1e-4), (
            f"Bernini-R forward_pass must feed the raw [0,1000] timestep; "
            f"got {seen.tolist()} for input {timesteps.tolist()}."
        )


# ── source_id assignment + LoRA targets ──────────────────────────────────────


class TestConditioningContract:
    def test_condition_streams_get_source_id_slot_plus_one(self, monkeypatch):
        """Ordered condition streams map to source_id = slot + 1; target = 0."""
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
            captured["cond_source_ids"] = list(cond_source_ids)
            captured["n_cond"] = len(cond_latents)
            return (torch.zeros_like(target_latent),)

        monkeypatch.setattr(drv_mod, "bernini_packed_forward", _spy)

        drv = _make_driver(_tiny_model())
        target = torch.randn(1, 16, 1, 8, 8)
        text = torch.randn(1, 5, 16)
        batch = {
            drv.BATCH_CONTROL_LATENTS: [
                torch.randn(1, 16, 1, 8, 8),
                torch.randn(1, 16, 1, 8, 8),
            ]
        }
        drv.forward_pass(target, torch.tensor([500.0]), text, batch)

        assert captured["n_cond"] == 2
        assert captured["cond_source_ids"] == [1.0, 2.0]

    def test_lora_targets_equal_wan21(self):
        """Bernini adds ZERO new weight modules → wan21's T2V target set verbatim."""
        drv = BerniniRDriver(_Defn(), torch.device("cpu"))
        assert drv.get_lora_targets() == WAN_T2V_LORA_TARGETS


# ── get_saver() mode contract (W5.T5 — self.saver single source of truth) ────
#
# Trainer._setup_family() no longer constructs its own saver: driver.get_saver()
# is the checkpoint path's ONLY consumer (pipeline_optimization.py). Before this
# fix, get_saver() hardcoded mode="t2v" regardless of the driver's actual mode —
# silently correct only because every shipped bernini_r definition happens to
# set mode: t2v today. These tests pin get_saver() to self.mode directly so a
# future i2v/other-mode definition can't silently regress the export label.


class _I2VDefn:
    """Single-expert definition stand-in with a non-t2v mode."""

    architecture_params = {"mode": "i2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


class _DualI2VDefn:
    """Dual-expert (14B MoE) definition stand-in with a non-t2v mode."""

    architecture_params = {
        "mode": "i2v",
        "dual_expert": True,
        "te.max_length": 512,
    }
    lora_targetable_modules: list[str] = []


class TestGetSaverModeMatchesDriverMode:
    def test_single_expert_get_saver_mode_matches_driver_mode(self):
        drv = BerniniRDriver(_Defn(), torch.device("cpu"))  # mode: t2v
        assert drv.mode == "t2v"
        assert drv.get_saver().mode == drv.mode

    def test_single_expert_get_saver_mode_matches_driver_mode_i2v(self):
        drv = BerniniRDriver(_I2VDefn(), torch.device("cpu"))
        assert drv.mode == "i2v"
        assert drv.get_saver().mode == drv.mode == "i2v"

    def test_dual_expert_get_saver_mode_matches_driver_mode_i2v(self):
        drv = BerniniRDriver(_DualI2VDefn(), torch.device("cpu"))
        assert drv.is_dual
        assert drv.mode == "i2v"
        assert drv.get_saver().mode == drv.mode == "i2v"
