"""PR6 — FLUX.1 Kontext contract tests.

These pin the two silent-failure classes the program memory warns about:
- timestep scale (the model gets [0,1] exactly once, never an extra ×1000)
- control latents are CLEAN (never noised) and tagged with offset position ids
plus the loss/output slicing to target tokens and the sampler precision regime.

A stub transformer captures forward kwargs so no real FLUX weights are needed.
"""

from __future__ import annotations

import types

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.flux1.family import Flux1Family
from app.engine.models.families.flux1.trainer import Flux1Trainer
from app.engine.models.families.flux1.trainer_kontext import Flux1KontextTrainer
from app.engine.models.families.flux1.utils import pack_latents
from app.engine.models.registry import registry


# ── Family dispatch + definition ─────────────────────────────────────────


class TestDispatch:
    def test_standard_definition_uses_base_trainer(self):
        defn = ModelDefinition(id="flux1-dev", family="flux1", name="Dev")
        assert Flux1Family(defn, {}).get_trainer_class() is Flux1Trainer

    def test_edit_definition_uses_kontext_trainer(self):
        defn = ModelDefinition(id="flux1-kontext-dev", family="flux1",
                               name="Kontext", control_inputs=1)
        assert Flux1Family(defn, {}).get_trainer_class() is Flux1KontextTrainer

    def test_kontext_definition_loads_with_control_inputs(self):
        registry.initialize()
        defn = registry.get_definition("flux1-kontext-dev")
        assert defn is not None
        assert defn.control_inputs == 1
        assert defn.family == "flux1"


# ── Forward-pass contract (stub transformer) ─────────────────────────────


class _StubTransformer:
    """Records forward kwargs; echoes hidden_states so slicing is observable."""

    def __init__(self):
        self.captured: dict = {}
        self.config = types.SimpleNamespace(pooled_projection_dim=768)

    def __call__(self, **kwargs):
        self.captured = kwargs
        # Return a tuple (return_dict=False) shaped like the full sequence so
        # the trainer's target-slice is exercised.
        return (kwargs["hidden_states"].clone(),)


def _make_kontext_trainer(stub):
    t = object.__new__(Flux1KontextTrainer)
    t.device = torch.device("cpu")
    t.autocast_dtype = torch.float32
    t.transformer = stub
    t.use_guidance_embed = True
    t.config = {"guidance_scale": 1.0}
    t._clip_pooled = None
    return t


def _packed_target(b=1, c=16, h=4, w=4):
    latents = torch.randn(b, c, h, w)
    packed, ids = pack_latents(latents)
    return packed, ids


class TestKontextForwardContract:
    def _run(self, control_h=4, control_w=4):
        stub = _StubTransformer()
        trainer = _make_kontext_trainer(stub)
        packed_target, target_ids = _packed_target(h=4, w=4)
        trainer._current_img_ids = target_ids
        control_latent = torch.randn(1, 16, control_h, control_w)
        batch = {"control_latents": [control_latent]}
        ts = torch.tensor([500.0])  # [0,1000] training convention
        out = trainer.forward_pass(packed_target, ts, torch.randn(1, 7, 4096), batch)
        return stub, trainer, packed_target, control_latent, out

    def test_timestep_divided_by_1000_once(self):
        stub, *_ = self._run()
        assert torch.allclose(stub.captured["timestep"], torch.tensor([0.5]))

    def test_control_latents_are_clean_not_noised(self):
        stub, _, packed_target, control_latent, _ = self._run()
        hidden = stub.captured["hidden_states"]
        target_len = packed_target.shape[1]
        expected_control, _ = pack_latents(control_latent)
        # The control tail must be bit-identical to the packed clean control.
        assert torch.equal(hidden[:, target_len:], expected_control)
        # And the head must be the (noisy) target, untouched.
        assert torch.equal(hidden[:, :target_len], packed_target)

    def test_control_position_ids_offset(self):
        stub, _, packed_target, _, _ = self._run()
        ids = stub.captured["img_ids"]
        target_len = packed_target.shape[1]
        # Target tokens keep coord-0 == 0; control tokens get coord-0 == 1.
        assert torch.all(ids[:target_len, 0] == 0)
        assert torch.all(ids[target_len:, 0] == 1)

    def test_output_sliced_to_target_tokens(self):
        _, _, packed_target, _, out = self._run()
        assert out.shape == packed_target.shape  # [B, L_target, 64]

    def test_no_controls_falls_back_to_base_forward(self):
        stub = _StubTransformer()
        trainer = _make_kontext_trainer(stub)
        packed_target, target_ids = _packed_target()
        trainer._current_img_ids = target_ids
        out = trainer.forward_pass(
            packed_target, torch.tensor([500.0]), torch.randn(1, 7, 4096),
            {"control_latents": []},
        )
        # Base forward feeds only the target tokens.
        assert stub.captured["hidden_states"].shape == packed_target.shape
        assert out.shape == packed_target.shape


# ── Sampler precision + slicing contract ─────────────────────────────────


class _SamplerStubTransformer:
    def __init__(self):
        self.config = types.SimpleNamespace(pooled_projection_dim=768)
        self.timesteps: list[float] = []
        self._params = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._params)

    def __call__(self, **kwargs):
        self.timesteps.append(float(kwargs["timestep"].reshape(-1)[0]))
        # Zero velocity → latents unchanged, so we can assert exact dtype/shape.
        return (torch.zeros_like(kwargs["hidden_states"]),)


class _StubVAE:
    def __init__(self):
        self.config = types.SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)
        self._params = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._params)

    def encode(self, x):
        b, _, h, w = x.shape
        latent = torch.randn(b, 16, h // 8, w // 8)
        dist = types.SimpleNamespace(mode=lambda: latent, sample=lambda: latent)
        return types.SimpleNamespace(latent_dist=dist)


def _make_kontext_sampler(tmp_path, control_image: bool):
    from app.engine.models.families.flux1.sampler_kontext import Flux1KontextSampler
    import structlog

    s = object.__new__(Flux1KontextSampler)
    s.device = torch.device("cpu")
    s.logger = structlog.get_logger("test")
    pipe = types.SimpleNamespace(
        transformer=_SamplerStubTransformer(),
        vae=_StubVAE(),
        _clip_pooled=None,
        use_guidance_embed=True,
        use_amp=False,
        components={},
        driver=None,
    )
    s.pipeline = pipe
    s.config = {}
    cfg = {}
    if control_image:
        from PIL import Image
        p = tmp_path / "ctrl.png"
        Image.new("RGB", (64, 64), "blue").save(p)
        cfg["control_images"] = [str(p)]
    s._active_prompt_cfg = cfg
    return s


class TestKontextSamplerContract:
    def test_trajectory_in_unit_interval_not_x1000(self, tmp_path):
        s = _make_kontext_sampler(tmp_path, control_image=True)
        noise = torch.randn(1, 16, 8, 8)
        prompt_emb = torch.randn(1, 7, 4096)
        s.denoise(noise, prompt_emb, num_steps=4, guidance_scale=1.0, seed=0)
        # Every timestep fed to the transformer is a raw flow time in [0,1].
        assert s.pipeline.transformer.timesteps
        assert all(0.0 <= t <= 1.0 for t in s.pipeline.transformer.timesteps)
        assert max(s.pipeline.transformer.timesteps) <= 1.0  # never ×1000

    def test_output_shape_is_target_only(self, tmp_path):
        s = _make_kontext_sampler(tmp_path, control_image=True)
        noise = torch.randn(1, 16, 8, 8)
        latents, lh, lw = s.denoise(
            noise, torch.randn(1, 7, 4096), num_steps=2, guidance_scale=1.0, seed=0,
        )
        # Packed target length = (8/2)*(8/2) = 16 tokens, 64 channels.
        assert latents.shape == (1, 16, 64)
        assert (lh, lw) == (8, 8)

    def test_no_control_image_falls_back(self, tmp_path):
        s = _make_kontext_sampler(tmp_path, control_image=False)
        noise = torch.randn(1, 16, 8, 8)
        latents, lh, lw = s.denoise(
            noise, torch.randn(1, 7, 4096), num_steps=2, guidance_scale=1.0, seed=0,
        )
        assert latents.shape == (1, 16, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
