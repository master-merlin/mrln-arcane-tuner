"""PR8 — Qwen-Image-Edit contract tests.

Pins the same silent-failure classes as Kontext, adapted to Qwen's patchified
sequence + ``img_shapes`` layout:
- timestep scale (model gets [0,1] exactly once, never an extra ×1000)
- control latents CLEAN (never noised), concatenated after the target patches
- output sliced to the target tokens (loss on target only)
- img_shapes carries one (F,H,W) per image (target + control)
- composite TE cache key: same caption + different control → distinct key
- sampler trajectory stays in flow time + slices to the target tokens

Stub transformer/VAE stand in for the 20B model so no weights are needed.
"""

from __future__ import annotations

import types

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.qwen_image.family import QwenImageFamily
from app.engine.models.families.qwen_image.trainer import QwenImageTrainer
from app.engine.models.families.qwen_image.trainer_edit import (
    QwenImageEditTrainer,
    composite_te_key,
    control_files_hash,
    patchify,
    unpatchify,
)
from app.engine.models.registry import registry


# ── Family dispatch + definition ─────────────────────────────────────────


class TestDispatch:
    def test_standard_definition_uses_base_trainer(self):
        defn = ModelDefinition(id="qwen-image-2512", family="qwen_image", name="Q")
        assert QwenImageFamily(defn, {}).get_trainer_class() is QwenImageTrainer

    def test_edit_definition_uses_edit_trainer(self):
        defn = ModelDefinition(id="qwen-image-edit-2509", family="qwen_image",
                               name="Edit", control_inputs=1)
        assert QwenImageFamily(defn, {}).get_trainer_class() is QwenImageEditTrainer

    def test_edit_definition_loads_with_control_inputs(self):
        registry.initialize()
        defn = registry.get_definition("qwen-image-edit-2509")
        assert defn is not None
        assert defn.control_inputs == 1
        assert defn.family == "qwen_image"


# ── Pure patchify / key helpers ──────────────────────────────────────────


class TestHelpers:
    def test_patchify_roundtrip_is_identity(self):
        latent = torch.randn(1, 16, 8, 8)
        x, pH, pW = patchify(latent, 2)
        assert x.shape == (1, 16, 64)  # (8/2)*(8/2)=16 tokens, 16*2*2=64 ch
        back = unpatchify(x, pH, pW, 16, 2)
        assert torch.equal(back, latent)

    def test_composite_key_differs_per_control(self):
        a = composite_te_key("make it rain", "aaaaaaaaaaaaaaaa")
        b = composite_te_key("make it rain", "bbbbbbbbbbbbbbbb")
        assert a != b
        # Same caption + same control → identical key.
        assert a == composite_te_key("make it rain", "aaaaaaaaaaaaaaaa")

    def test_control_files_hash_differs_per_file(self, tmp_path):
        p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
        p1.write_bytes(b"AAAA")
        p2.write_bytes(b"BBBB")
        assert control_files_hash([str(p1)]) != control_files_hash([str(p2)])
        # Deterministic + memoized.
        assert control_files_hash([str(p1)]) == control_files_hash([str(p1)])


# ── Forward-pass contract (stub transformer) ─────────────────────────────


class _StubModel:
    def __init__(self):
        self.captured: dict = {}
        self.config = types.SimpleNamespace(patch_size=2, out_channels=16)

    def __call__(self, **kwargs):
        self.captured = kwargs
        return (kwargs["hidden_states"].clone(),)


def _make_edit_trainer(stub):
    t = object.__new__(QwenImageEditTrainer)
    t.model = stub
    return t


class TestQwenEditForwardContract:
    def _run(self, control_h=8, control_w=8):
        stub = _StubModel()
        trainer = _make_edit_trainer(stub)
        noisy = torch.randn(1, 16, 8, 8)
        control = torch.randn(1, 16, control_h, control_w)
        enc = (torch.randn(1, 5, 3584), torch.ones(1, 5, dtype=torch.long))
        ts = torch.tensor([500.0])
        out = trainer.forward_pass(noisy, ts, enc, {"control_latents": [control]})
        return stub, noisy, control, out

    def test_timestep_divided_by_1000_once(self):
        stub, *_ = self._run()
        assert torch.allclose(stub.captured["timestep"], torch.tensor([0.5]))

    def test_control_patches_are_clean(self):
        stub, noisy, control, _ = self._run()
        hidden = stub.captured["hidden_states"]
        target_tokens = patchify(noisy, 2)[0].shape[1]
        expected_control, _, _ = patchify(control, 2)
        assert torch.equal(hidden[:, target_tokens:], expected_control)
        assert torch.equal(hidden[:, :target_tokens], patchify(noisy, 2)[0])

    def test_img_shapes_lists_target_and_control(self):
        stub, *_ = self._run()
        # One [target, control] shape-list per batch element.
        assert stub.captured["img_shapes"] == [[(1, 4, 4), (1, 4, 4)]]

    def test_output_sliced_and_unpatchified_to_target(self):
        _, noisy, _, out = self._run()
        # Echo stub → unpatchify(target patches) reconstructs the input exactly.
        assert out.shape == noisy.shape
        assert torch.equal(out, noisy)

    def test_no_controls_falls_back_to_base(self):
        stub = _StubModel()
        trainer = _make_edit_trainer(stub)
        noisy = torch.randn(1, 16, 8, 8)
        enc = (torch.randn(1, 5, 3584), torch.ones(1, 5, dtype=torch.long))
        out = trainer.forward_pass(noisy, torch.tensor([500.0]), enc,
                                   {"control_latents": []})
        # Base forward feeds only the target patches (no control tail).
        assert stub.captured["hidden_states"].shape == patchify(noisy, 2)[0].shape
        assert out.shape == noisy.shape


# ── Composite TE cache keying (text-only fallback path) ──────────────────


class TestEditEncodeTextComposite:
    def _trainer(self):
        t = object.__new__(QwenImageEditTrainer)
        t.text_cache = {}
        t._ctrl_hash_memo = {}
        t.device = torch.device("cpu")
        return t

    def test_same_caption_different_control_distinct_cache_entries(self, tmp_path, monkeypatch):
        t = self._trainer()
        c1, c2 = tmp_path / "c1.png", tmp_path / "c2.png"
        c1.write_bytes(b"ONE")
        c2.write_bytes(b"TWO")

        calls: list[str] = []

        def fake_encode(caption, control_path, dtype):
            calls.append(control_path)
            return torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.long)

        monkeypatch.setattr(t, "_encode_text_with_control", fake_encode)

        batch1 = {"control_paths": [[str(c1)]]}
        batch2 = {"control_paths": [[str(c2)]]}
        t.encode_text(["make it snow"], torch.float32, batch1)
        t.encode_text(["make it snow"], torch.float32, batch2)

        # Two distinct control images → two cache entries → two encodes.
        assert len(t.text_cache) == 2
        assert len(calls) == 2

    def test_same_caption_same_control_reuses_entry(self, tmp_path, monkeypatch):
        t = self._trainer()
        c1 = tmp_path / "c1.png"
        c1.write_bytes(b"ONE")
        calls: list[str] = []
        monkeypatch.setattr(
            t, "_encode_text_with_control",
            lambda cap, p, dt: (calls.append(p),
                                (torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.long)))[1],
        )
        batch = {"control_paths": [[str(c1)]]}
        t.encode_text(["make it snow"], torch.float32, batch)
        t.encode_text(["make it snow"], torch.float32, batch)
        assert len(t.text_cache) == 1
        assert len(calls) == 1  # second call is a cache hit

    def test_no_control_paths_falls_back_to_text_only(self, monkeypatch):
        t = self._trainer()
        called = {"n": 0}

        def fake_super(captions, dtype, batch=None):
            called["n"] += 1
            return torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.long)

        # Patch the bound super().encode_text via the base class.
        monkeypatch.setattr(QwenImageTrainer, "encode_text",
                            lambda self, c, d, batch=None: fake_super(c, d, batch))
        t.encode_text(["plain caption"], torch.float32, None)
        assert called["n"] == 1


# ── Sampler precision + slicing contract ─────────────────────────────────


class _SamplerStubTransformer:
    def __init__(self):
        self.config = types.SimpleNamespace(in_channels=64)
        self.timesteps: list[float] = []
        self._params = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._params)

    def to(self, *args, **kwargs):
        return self

    def __call__(self, **kwargs):
        self.timesteps.append(float(kwargs["timestep"].reshape(-1)[0]))
        return (torch.zeros_like(kwargs["hidden_states"]),)


class _StubVAE:
    def __init__(self):
        self.config = types.SimpleNamespace(
            latents_mean=[0.0] * 16, latents_std=[1.0] * 16, z_dim=16,
        )
        self._params = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._params)

    def encode(self, x):
        b, _, _, h, w = x.shape
        latent = torch.randn(b, 16, 1, h // 8, w // 8)
        dist = types.SimpleNamespace(mode=lambda: latent, sample=lambda: latent)
        return types.SimpleNamespace(latent_dist=dist)


def _make_edit_sampler(tmp_path, control_image: bool):
    from app.engine.models.families.qwen_image.sampler_edit import QwenImageEditSampler

    s = object.__new__(QwenImageEditSampler)
    s.device = torch.device("cpu")
    s.logger = structlog_get()
    s._scheduler = None
    pipe = types.SimpleNamespace(
        transformer=_SamplerStubTransformer(),
        vae=_StubVAE(),
        definition=types.SimpleNamespace(architecture_params={}),
    )
    s.pipeline = pipe
    s.config = {}
    # Geometry normally set by _create_initial_noise: for 64px @ vae_sf 8,
    # lat = 2*(64//16) = 8, giving (8/2)*(8/2) = 16 packed tokens.
    s._sample_height = 64
    s._sample_width = 64
    s._lat_h = 8
    s._lat_w = 8
    s._vae_sf = 8
    cfg: dict = {}
    if control_image:
        from PIL import Image
        p = tmp_path / "ctrl.png"
        Image.new("RGB", (64, 64), "blue").save(p)
        cfg["control_images"] = [str(p)]
    s._active_prompt_cfg = cfg
    return s


def structlog_get():
    import structlog
    return structlog.get_logger("test")


class TestQwenEditSamplerContract:
    def _noise(self):
        # Packed target [1, (8/2)*(8/2)=16 tokens, 16*4=64 ch].
        return torch.randn(1, 16, 64)

    def _prompt(self):
        return {"embeds": torch.randn(1, 5, 3584), "mask": torch.ones(1, 5, dtype=torch.long)}

    def test_trajectory_in_flow_time_not_x1000(self, tmp_path):
        s = _make_edit_sampler(tmp_path, control_image=True)
        s.denoise(self._noise(), self._prompt(), num_steps=3, guidance_scale=1.0, seed=0)
        assert s.pipeline.transformer.timesteps
        assert all(0.0 <= t <= 1.0 for t in s.pipeline.transformer.timesteps)

    def test_output_is_target_only(self, tmp_path):
        s = _make_edit_sampler(tmp_path, control_image=True)
        out = s.denoise(self._noise(), self._prompt(), num_steps=2,
                        guidance_scale=1.0, seed=0)
        # Unpacked back to [B, C, 1, lat_h, lat_w] for the target only.
        assert out["latents"].shape == (1, 16, 1, 8, 8)

    def test_no_control_image_falls_back(self, tmp_path):
        s = _make_edit_sampler(tmp_path, control_image=False)
        out = s.denoise(self._noise(), self._prompt(), num_steps=2,
                        guidance_scale=1.0, seed=0)
        assert out["latents"].shape == (1, 16, 1, 8, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
