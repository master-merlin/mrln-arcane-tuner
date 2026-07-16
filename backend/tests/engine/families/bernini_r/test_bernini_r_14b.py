"""Bernini-R 14B dual-expert (MoE) contract tests (Task BR6).

Pins the 14B-only behaviour layered on top of the 1.3B single-expert family:

* **Definition** — the 14B YAML loads with the Wan2.2-A14B geometry, the
  ``dual_expert`` / ``switch_dit_boundary`` MoE flags, the v2v pins, and resolves
  a ``dual_expert`` + ``is_video`` capability (the 1.3B stays single-expert).
* **Expert routing (HIGH↔transformer assignment)** — ``expert_for_timestep``
  routes ``t >= 875`` to the HIGH expert (``transformer``) and ``t < 875`` to the
  LOW expert (``transformer_2``); a mutation on each side of the boundary reaches
  the RIGHT expert module (recon §3).
* **Range-split band sampling** — the HIGH expert's timesteps land entirely in
  ``[875, 1000]`` and the LOW expert's in ``[0, 875)``, each matching BERNINI's
  SD3-mode+shift formula RESTRICTED to the band (redraw/rejection, deterministic
  under a fixed seed). This is the deliberate divergence from wan22.
* **Sampler boundary switch** — a descending UniPC trajectory hits the HIGH
  expert while ``t >= 875`` and the LOW expert below it (stub transformers,
  counted calls).
* **Dual saver** — two ComfyUI files (``_high_noise`` / ``_low_noise``), wan22
  naming, ``bernini-r`` provenance label, wan-canonical ``diffusion_model.*`` keys.
* **1.3B untouched** — the 1.3B definition is NOT dual-expert.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from diffusers import WanTransformer3DModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import safe_open

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.bernini_r.saver import BerniniRDualSaver
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.expert_router import HIGH, LOW
from app.engine.models.registry import ModelRegistry

MODEL_ID = "bernini-r-14b"


# ── Fixtures / stubs ──────────────────────────────────────────────────────────


class _DualDefn:
    """Minimal dual-expert definition stand-in (no weights, no YAML)."""

    architecture_params = {
        "mode": "t2v",
        "te.max_length": 512,
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
        "moe.boundary_ratio": 0.875,
        "scheduler.num_train_timesteps": 1000,
    }
    lora_targetable_modules: list[str] = []


def _dual_driver() -> BerniniRDriver:
    return BerniniRDriver(_DualDefn(), torch.device("cpu"))


def _reference_band(
    n: int, mode_scale: float, shift: float, lo: float, hi: float, high: bool
) -> torch.Tensor:
    """Independent redraw of the mode+shift formula truncated to a band."""

    def _u(raw: torch.Tensor) -> torch.Tensor:
        return 1.0 - raw - mode_scale * (torch.cos(math.pi * raw / 2.0) ** 2 - 1.0 + raw)

    out = torch.empty(n)
    filled = torch.zeros(n, dtype=torch.bool)
    while not bool(filled.all()):
        need = int((~filled).sum())
        raw = torch.rand(need)
        uu = _u(raw)
        ts = (shift * uu / (1.0 + (shift - 1.0) * uu)) * 1000.0
        ok = (ts >= lo) if high else (ts < hi)
        idx = torch.nonzero(~filled, as_tuple=False).flatten()
        out[idx[ok]] = ts[ok]
        filled[idx[ok]] = True
    return out


# ── Definition ────────────────────────────────────────────────────────────────


@pytest.fixture()
def registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    r = ModelRegistry()
    r.initialize()
    yield r
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


class TestDefinition:
    def test_loads_and_is_edit(self, registry):
        defn = registry.get_definition(MODEL_ID)
        assert defn is not None, f"{MODEL_ID} did not load"
        assert defn.family == "bernini_r"
        assert defn.control_inputs == 1

    def test_dual_expert_moe_flags(self, registry):
        arch = registry.get_definition(MODEL_ID).architecture_params
        assert arch["dual_expert"] is True
        assert arch["skip_transformer_2"] is False
        assert arch["switch_dit_boundary"] == 0.875
        assert arch["moe.boundary_ratio"] == 0.875

    def test_14b_geometry_not_1_3b(self, registry):
        arch = registry.get_definition(MODEL_ID).architecture_params
        assert arch["transformer.num_layers"] == 40
        assert arch["transformer.num_attention_heads"] == 40
        assert arch["transformer.attention_head_dim"] == 128
        assert arch["transformer.hidden_size"] == 5120
        assert arch["transformer.ffn_dim"] == 13824
        assert arch["transformer.in_channels"] == 16
        assert arch["transformer.out_channels"] == 16
        assert arch["transformer.patch_size"] == [1, 2, 2]
        # second expert declared
        assert arch["transformer_2._class_name"] == "WanTransformer3DModel"

    def test_v2v_pins_match_family(self, registry):
        d = registry.get_definition(MODEL_ID)
        assert d.defaults["mode_scale"] == 1.29
        assert d.defaults["timestep_shift"] == 5.0
        assert d.defaults["num_inference_steps"] == 40
        assert d.defaults["guidance_scale"] == 4.0
        assert d.defaults["num_frames"] == 81
        assert d.architecture_params["scheduler.flow_shift"] == 5.0
        assert "scheduler.flow_shift" in d.enrich_pinned_keys

    def test_capability_dual_expert_and_video(self, registry):
        caps = resolve_capabilities(registry.get_definition(MODEL_ID))["capabilities"]
        assert caps["dual_expert"] is True
        assert caps["is_video"] is True
        assert caps["is_edit"] is True
        assert caps["supports_train_te"] is False

    def test_1_3b_is_not_dual_expert(self, registry):
        caps = resolve_capabilities(registry.get_definition("bernini-r-1.3b"))[
            "capabilities"
        ]
        assert caps.get("dual_expert", False) is False


# ── Expert routing (HIGH↔transformer assignment) ──────────────────────────────


class TestExpertRouting:
    def test_boundary_timestep_derived(self):
        drv = _dual_driver()
        assert drv.is_dual is True
        assert drv.boundary == 0.875
        assert drv.boundary_timestep == 875.0

    def test_high_expert_is_transformer_low_is_transformer_2(self):
        """t >= 875 → HIGH (transformer); t < 875 → LOW (transformer_2). An
        inversion here silently trains each LoRA on the wrong band (recon §3)."""
        drv = _dual_driver()
        assert drv.expert_for_timestep(torch.tensor([900.0])) == HIGH
        assert drv.expert_for_timestep(torch.tensor([874.9])) == LOW
        # Exactly on the boundary is HIGH (inclusive t >= boundary).
        assert drv.expert_for_timestep(torch.tensor([875.0])) == HIGH
        assert drv.expert_for_timestep(0.0) == LOW
        assert drv.expert_for_timestep(1000.0) == HIGH

    def test_mutation_each_side_reaches_right_module(self):
        drv = _dual_driver()
        high_stub, low_stub = nn.Linear(2, 2), nn.Linear(2, 2)
        drv.transformer_high = high_stub
        drv.transformer_low = low_stub
        # A timestep on each side of the boundary reaches the RIGHT module.
        assert drv.transformer_for_timestep(torch.tensor([950.0])) is high_stub
        assert drv.transformer_for_timestep(torch.tensor([100.0])) is low_stub

    def test_assign_components_wires_both_experts(self):
        drv = _dual_driver()
        high, low = nn.Linear(2, 2), nn.Linear(2, 2)
        drv.configure_expert_mode("both")
        drv.assign_components({"unet": high, "unet_low": low, "vae": None})
        assert drv.transformer_high is high
        assert drv.transformer_low is low
        assert drv.active_expert == HIGH
        assert drv.get_primary_model() is high


# ── Range-split band sampling ─────────────────────────────────────────────────


def _band_trainer() -> BerniniRTrainer:
    import structlog

    t = object.__new__(BerniniRTrainer)
    t.device = torch.device("cpu")
    t.config = {}
    t.driver = _dual_driver()
    t.logger = structlog.get_logger("test")
    return t


class TestBandSampling:
    def test_high_band_all_in_range(self):
        t = _band_trainer()
        torch.manual_seed(7)
        ts = t._sample_band(HIGH, 50_000)
        assert ts.shape == (50_000,)
        assert float(ts.min()) >= 875.0, float(ts.min())
        assert float(ts.max()) <= 1000.0

    def test_low_band_all_in_range(self):
        t = _band_trainer()
        torch.manual_seed(7)
        ts = t._sample_band(LOW, 50_000)
        assert float(ts.min()) >= 0.0
        assert float(ts.max()) < 875.0, float(ts.max())

    def test_high_band_distribution_matches_truncated_formula(self):
        """Within-band shape == mode+shift restricted to [875,1000] (rejection,
        not rescale)."""
        t = _band_trainer()
        torch.manual_seed(11)
        got = t._sample_band(HIGH, 50_000)
        ref = _reference_band(50_000, 1.29, 5.0, 875.0, 1000.0, high=True)
        qs = torch.linspace(0.05, 0.95, 19)
        assert torch.allclose(
            torch.quantile(got, qs), torch.quantile(ref, qs), atol=8.0
        )

    def test_deterministic_fixed_seed(self):
        t = _band_trainer()
        torch.manual_seed(1234)
        a = t._sample_band(HIGH, 4096)
        torch.manual_seed(1234)
        b = t._sample_band(HIGH, 4096)
        assert torch.equal(a, b)

    def test_dual_sample_timesteps_routes_to_active_band(self):
        """With a router attached, ``sample_timesteps`` band-samples the ACTIVE
        expert (the range-split divergence); no router → full-range."""
        t = _band_trainer()

        class _R:
            def choose_expert(self, s):
                return HIGH

        t.driver.router = _R()
        t.driver._set_active(HIGH)
        torch.manual_seed(3)
        hi = t.sample_timesteps(20_000)
        assert float(hi.min()) >= 875.0

        t.driver._set_active(LOW)
        torch.manual_seed(3)
        lo = t.sample_timesteps(20_000)
        assert float(lo.max()) < 875.0

    def test_single_expert_full_range_when_no_router(self):
        """No driver (bare trainer) → full-range formula (1.3B byte-identical)."""
        import structlog

        t = object.__new__(BerniniRTrainer)
        t.device = torch.device("cpu")
        t.config = {}
        t.logger = structlog.get_logger("t")
        torch.manual_seed(42)
        ts = t.sample_timesteps(20_000)
        # Full range spans both sides of the boundary.
        assert float(ts.min()) < 875.0 and float(ts.max()) > 875.0


# ── Sampler boundary switch ───────────────────────────────────────────────────


def _tiny_wan(seed: int = 0) -> WanTransformer3DModel:
    torch.manual_seed(seed)
    return (
        WanTransformer3DModel(
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
        .to(torch.float32)
        .eval()
    )


class _DualSamplerDriver:
    def __init__(self, high, low):
        self.is_dual = True
        self.transformer_high = high
        self.transformer_low = low
        self.boundary = 0.875
        self.boundary_timestep = 875.0
        self.vae = None

    def get_primary_model(self):
        return self.transformer_high

    def expert_for_timestep(self, t):
        tv = float(t.reshape(-1)[0].item()) if isinstance(t, torch.Tensor) else float(t)
        return HIGH if tv >= self.boundary_timestep else LOW

    def transformer_for_timestep(self, t):
        return (
            self.transformer_high
            if self.expert_for_timestep(t) == HIGH
            else self.transformer_low
        )


class _DualPipeline:
    class _Defn:
        architecture_params = {"scheduler.flow_shift": 5.0, "mode": "t2v"}

    def __init__(self, high, low, emb):
        self.config: dict = {}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _DualSamplerDriver(high, low)
        self.definition = self._Defn()
        self._emb = emb

    def encode_text(self, caps, dtype):
        return self._emb


class TestSamplerBoundarySwitch:
    def test_trajectory_switches_experts_at_boundary(self, monkeypatch):
        import app.engine.models.families.bernini_r.sampler as smod
        from app.engine.models.families.bernini_r.sampler import BerniniRSampler

        high, low = _tiny_wan(1), _tiny_wan(2)
        emb = torch.randn(1, 5, 16)
        sampler = BerniniRSampler(_DualPipeline(high, low, emb))
        sampler._active_prompt_cfg = {}  # no control → degenerate t2v

        calls: list[tuple[str, float]] = []

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
            who = "high" if model is high else "low" if model is low else "?"
            calls.append((who, float(timestep.reshape(-1)[0].item())))
            return (torch.zeros_like(target_latent),)

        monkeypatch.setattr(smod, "bernini_packed_forward", _spy)

        noise = torch.randn(1, 16, 1, 8, 8)
        # CFG off (gs<=1) → one forward per step.
        sampler.denoise(noise, emb, num_steps=12, guidance_scale=1.0, seed=0)

        assert calls, "no forwards recorded"
        highs = [t for who, t in calls if who == "high"]
        lows = [t for who, t in calls if who == "low"]
        assert highs and lows, f"expected BOTH experts across the boundary: {calls}"
        # Correct side: every high call had t>=875; every low call t<875.
        assert all(t >= 875.0 for t in highs), highs
        assert all(t < 875.0 for t in lows), lows


# ── Dual saver ────────────────────────────────────────────────────────────────


class _Attn(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])


class _FFN(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        proj_mod = nn.Module()
        proj_mod.proj = nn.Linear(dim, dim * 2)
        self.net = nn.ModuleList([proj_mod, nn.Identity(), nn.Linear(dim * 2, dim)])


class _Block(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.attn1 = _Attn(dim)
        self.attn2 = _Attn(dim)
        self.ffn = _FFN(dim)


class _FakeWan(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


def _peft_wrap(model):
    targets = [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=8, target_modules=targets))


class TestDualSaver:
    def test_writes_two_expert_files_bernini_labelled(self, tmp_path):
        high = _peft_wrap(_FakeWan(2))
        low = _peft_wrap(_FakeWan(2))
        out = tmp_path / "bernini_r_14b_lora.safetensors"

        BerniniRDualSaver(mode="t2v").save(
            {"unet_high": high, "unet_low": low, "config": {"save_precision": "bf16"}},
            out,
            metadata={},
        )

        high_path = tmp_path / "bernini_r_14b_lora_high_noise.safetensors"
        low_path = tmp_path / "bernini_r_14b_lora_low_noise.safetensors"
        assert high_path.exists() and low_path.exists()
        assert not out.exists()  # only the two expert files

        for path, expert in ((high_path, "high"), (low_path, "low")):
            with safe_open(str(path), framework="pt") as f:
                keys = list(f.keys())
                meta = f.metadata()
            # wan-canonical ComfyUI keys (byte-equal to wan22's export set).
            assert all(k.startswith("diffusion_model.blocks.") for k in keys), keys
            assert any(".self_attn.q.lora_down.weight" in k for k in keys)
            # Bernini provenance label, per-expert.
            assert meta.get("modelspec.architecture") == f"bernini-r-t2v-{expert}"
