"""Tests for Krea2Sampler — TDD steps 1–4.

Tests cover:
  1. encode_prompt returns 4-D embeds + mask
  2. denoise shape/CFG + distilled-mu branch (guidance_scale>0 and ==0)
  3. decode_latents + trainer._create_sampler wiring
  4. Precision-contract: no autocast collapse, multi-step run stays non-degenerate
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import torch


# ── Shared tiny transformer config (2-block, minimal dims) ──────────────────

_TINY_CFG = dict(
    in_channels=64,
    num_layers=2,
    attention_head_dim=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=256,
    timestep_embed_dim=256,
    text_hidden_dim=128,
    num_text_layers=12,
    text_num_attention_heads=4,
    text_num_key_value_heads=4,
    text_intermediate_size=128,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(32, 48, 48),   # sum=128 == attention_head_dim
    rope_theta=1000.0,
    norm_eps=1e-5,
)

# Tiny spatial dims: 8×8 px → lat_h=1, lat_w=1, img_seq=1, patch latent dim=64
_W, _H = 8, 8   # pixel dims used in tests
_VAE_SF = 8     # vae scale factor
_PATCH = 2      # transformer patch_size
# lat = H//vae_sf = 8//8 = 1
_LAT_H = _H // _VAE_SF   # = 1
_LAT_W = _W // _VAE_SF   # = 1
_IMG_SEQ = (_LAT_H // _PATCH) * (_LAT_W // _PATCH)   # = 0? No: pack_latents in pipeline is 2-stride

# Actually for our sampler, the latent grid is NOT further packed via qwen-style.
# Krea2 latents [B,C,H,W] → pack_latents [B, (H/p)*(W/p), C*p*p]
# With H=W=1, p=2: (1//2)*(1//2) = 0 → we need at least 2×2 latents.
# Use 16×16 px → lat_h=2, lat_w=2, img_seq = (2//2)*(2//2) = 1
_W2, _H2 = 16, 16
_LAT_H2 = _H2 // _VAE_SF   # = 2
_LAT_W2 = _W2 // _VAE_SF   # = 2
# img_seq = (lat_h // patch) * (lat_w // patch) = 1*1 = 1
_TXT_SEQ = 7
_NUM_TEXT_LAYERS = 12
_TEXT_DIM = 128   # tiny


def _build_tiny_model():
    from diffusers import Krea2Transformer2DModel
    return Krea2Transformer2DModel.from_config(_TINY_CFG).eval()


def _build_mock_pipeline(model, raw=True):
    """Build a mock Krea2Trainer-like pipeline with tiny model wired up.

    Uses a real Krea2Driver with the tiny model so driver.forward_pass
    produces real tensor output (not MagicMock chains).
    """
    from app.engine.models.families.krea2.driver import Krea2Driver

    # Build definition mock for driver
    drv_defn = MagicMock()
    drv_defn.family = "krea2"
    drv_defn.id = "krea2-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = {
        "te.text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    }

    driver = Krea2Driver(drv_defn, torch.device("cpu"))
    driver.assign_components({
        "unet": model,
        "vae": None,
        "text_encoder": None,
        "tokenizer": None,
    })

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline.vae = _build_mock_vae()
    pipeline.driver = driver

    # encode_text returns (4-D emb, mask) for single caption
    def _encode_text(prompts, dtype=None):
        B = len(prompts)
        emb = torch.randn(B, _TXT_SEQ, _NUM_TEXT_LAYERS, _TEXT_DIM)
        mask = torch.ones(B, _TXT_SEQ, dtype=torch.long)
        return emb, mask

    pipeline.encode_text = _encode_text

    # definition with is_distilled from raw flag
    defn = MagicMock()
    defn.architecture_params = {}
    defn.defaults = {"is_distilled": not raw}
    pipeline.definition = defn

    pipeline.config = {
        "sample_every_n_steps": 50,
        "sample_negative_prompt": "",
    }
    pipeline._block_swap_managers = None
    return pipeline


def _build_mock_vae():
    """Build a mock VAE that performs a trivial decode."""
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config = MagicMock()
    # 16 latent channels, 8 scale factor
    vae.config.latents_mean = [0.0] * 16
    vae.config.latents_std = [1.0] * 16
    vae.config.z_dim = 16

    def _decode(latents, return_dict=False):
        # latents is [B, C, 1, H, W] or [B, C, H, W]; return [B, 3, H*vae_sf, W*vae_sf]
        B = latents.shape[0]
        # For our test: H=W=2 → decoded H=16, W=16
        out = torch.zeros(B, 3, 1, _H2, _W2)
        if return_dict:
            result = MagicMock()
            result.sample = out
            return result
        return (out,)

    vae.decode = _decode
    return vae


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — encode_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodePrompt:

    def test_encode_prompt_returns_4d_embeds(self):
        """encode_prompt must return dict with 4-D embeds and a mask."""
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=True)

        sampler = Krea2Sampler(pipeline)
        result = sampler.encode_prompt("a test prompt")

        assert isinstance(result, dict), "encode_prompt must return a dict"
        assert "embeds" in result, "result must have 'embeds' key"
        assert "mask" in result, "result must have 'mask' key"
        embeds = result["embeds"]
        assert embeds.ndim == 4, (
            f"embeds must be 4-D (B, seq, num_layers, dim), got {embeds.ndim}-D"
        )
        # Batch dim=1 for single prompt
        assert embeds.shape[0] == 1, f"batch dim should be 1, got {embeds.shape[0]}"
        mask = result["mask"]
        assert mask.ndim == 2, f"mask must be 2-D (B, seq), got {mask.ndim}-D"

    def test_encode_prompt_delegates_to_pipeline(self):
        """encode_prompt must call pipeline.encode_text exactly once."""
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=True)

        calls = []

        def _spy_encode(prompts, dtype=None):
            calls.append(prompts)
            B = len(prompts)
            emb = torch.randn(B, _TXT_SEQ, _NUM_TEXT_LAYERS, _TEXT_DIM)
            mask = torch.ones(B, _TXT_SEQ, dtype=torch.long)
            return emb, mask

        pipeline.encode_text = _spy_encode

        sampler = Krea2Sampler(pipeline)
        sampler.encode_prompt("hello")

        assert len(calls) == 1, f"encode_text called {len(calls)} times, expected 1"
        assert calls[0] == ["hello"], f"wrong prompts passed: {calls[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — denoise: shape + CFG + mu branch
# ─────────────────────────────────────────────────────────────────────────────

class TestDenoise:

    def _build_sampler(self, raw=True):
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=raw)
        sampler = Krea2Sampler(pipeline)
        return sampler, model

    def test_denoise_shape_and_finite_cfg(self):
        """2-step denoise with CFG produces correct shape, finite, non-degenerate."""
        sampler, model = self._build_sampler(raw=True)
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W2, _H2, gen)

        prompt_emb = sampler.encode_prompt("a test image")
        result = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=4.5,
            seed=42,
        )

        # Result must be a dict with "latents"
        assert isinstance(result, dict), "denoise must return a dict"
        assert "latents" in result, "result must have 'latents' key"
        latents = result["latents"]

        # Shape: [B, C, 1, lat_h, lat_w] from unpack step — or [B, C, H, W]
        # Our sampler returns raw [B, C, H, W] latents
        assert latents.shape[0] == 1, f"batch dim wrong: {latents.shape}"
        assert latents.isfinite().all(), "denoise output contains NaN or inf"
        assert latents.float().std() > 0, "denoise output is degenerate (zero std)"

    def test_denoise_no_cfg_turbo(self):
        """Turbo (guidance_scale=0) runs single forward per step (no uncond)."""
        sampler, model = self._build_sampler(raw=False)

        forward_calls = []
        original_forward = model.forward

        def _spy_forward(*args, **kwargs):
            forward_calls.append(1)
            return original_forward(*args, **kwargs)

        model.forward = _spy_forward

        gen = torch.Generator().manual_seed(0)
        noise = sampler._create_initial_noise(_W2, _H2, gen)
        prompt_emb = sampler.encode_prompt("test")

        result = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=0.0,
            seed=0,
        )

        latents = result["latents"]
        assert latents.isfinite().all(), "turbo output has NaN"

        # With guidance_scale=0: exactly num_steps=2 forward calls (one per step)
        assert forward_calls == [1, 1], (
            f"turbo should call forward once per step (2 total), got {len(forward_calls)}"
        )

    def test_distilled_mu_is_fixed_115(self):
        """When is_distilled=True, mu must be exactly 1.15 (not resolution-derived)."""
        from app.engine.models.families.krea2.sampler import _DISTILLED_MU

        assert abs(_DISTILLED_MU - 1.15) < 1e-9, f"_DISTILLED_MU must be 1.15, got {_DISTILLED_MU}"

        sampler, _ = self._build_sampler(raw=False)   # raw=False → is_distilled=True
        # The mu used for distilled must be 1.15 — we check via the _compute_mu helper
        mu = sampler._compute_mu(image_seq_len=4)  # arbitrary seq_len
        assert abs(mu - 1.15) < 1e-9, (
            f"distilled mu should be 1.15, got {mu}"
        )

    def test_raw_mu_is_resolution_derived(self):
        """When is_distilled=False, mu is resolution-derived (not fixed)."""
        sampler, _ = self._build_sampler(raw=True)   # is_distilled=False
        mu_small = sampler._compute_mu(image_seq_len=256)
        mu_large = sampler._compute_mu(image_seq_len=6400)
        # Must differ by a meaningful amount — not fixed
        assert abs(mu_large - mu_small) > 0.01, (
            f"raw mu should vary with seq_len, but got small={mu_small}, large={mu_large}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — decode_latents + trainer._create_sampler wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeAndTrainerWiring:

    def test_decode_latents_returns_pil(self):
        """decode_latents must return a PIL Image."""
        from PIL import Image
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=True)
        sampler = Krea2Sampler(pipeline)

        # Build a latents bundle matching the expected format (after denoise)
        # [B, C, 1, lat_h, lat_w]
        latents_bundle = {
            "latents": torch.randn(1, 16, 1, _LAT_H2, _LAT_W2),
            "height": _H2,
            "width": _W2,
        }

        result = sampler.decode_latents(latents_bundle)
        assert isinstance(result, Image.Image), (
            f"decode_latents must return PIL Image, got {type(result)}"
        )
        assert result.mode == "RGB", f"expected RGB, got {result.mode}"
        assert result.size == (_W2, _H2), f"expected {(_W2, _H2)}, got {result.size}"

    def test_create_sampler_returns_krea2sampler_when_enabled(self):
        """_create_sampler must return Krea2Sampler when sample_every_n_steps > 0."""
        from app.engine.models.families.krea2.trainer import Krea2Trainer
        from app.engine.models.families.krea2.sampler import Krea2Sampler

        trainer = MagicMock(spec=Krea2Trainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = Krea2Trainer._create_sampler(trainer)
        assert isinstance(sampler, Krea2Sampler), (
            f"expected Krea2Sampler, got {type(sampler)}"
        )

    def test_create_sampler_returns_none_when_disabled(self):
        """_create_sampler must return None when sample_every_n_steps == 0."""
        from app.engine.models.families.krea2.trainer import Krea2Trainer

        trainer = MagicMock(spec=Krea2Trainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = Krea2Trainer._create_sampler(trainer)
        assert sampler is None, f"expected None, got {sampler}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Precision-contract: no autocast collapse
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecisionContract:

    def test_no_autocast_in_denoise_source(self):
        """Denoise source code must NOT use torch.autocast as a context manager."""
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        source = inspect.getsource(Krea2Sampler.denoise)
        # Strip comment lines before checking — comments are docs, not code.
        non_comment_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        ]
        non_comment_source = "\n".join(non_comment_lines)
        # The invariant: no with torch.autocast wrapping the DiT forward
        assert "torch.autocast" not in non_comment_source, (
            "denoise must NOT use torch.autocast (autocast-collapse gotcha)"
        )

    def test_multistep_run_stays_non_degenerate(self):
        """4-step tiny-model run must produce finite, non-degenerate latents."""
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=True)
        sampler = Krea2Sampler(pipeline)

        gen = torch.Generator().manual_seed(7)
        noise = sampler._create_initial_noise(_W2, _H2, gen)
        prompt_emb = sampler.encode_prompt("a precision test")

        result = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=4,
            guidance_scale=4.5,
            seed=7,
        )

        latents = result["latents"]
        assert latents.isfinite().all(), "4-step denoise output contains NaN or inf"
        assert latents.float().std() > 0, "4-step denoise output is degenerate"

    def test_cfg_double_forward_per_step(self):
        """With guidance_scale>0, transformer is called 2× per step (cond+uncond)."""
        from app.engine.models.families.krea2.sampler import Krea2Sampler
        model = _build_tiny_model()
        pipeline = _build_mock_pipeline(model, raw=True)
        sampler = Krea2Sampler(pipeline)

        forward_calls = []
        original_forward = model.forward

        def _spy_forward(*args, **kwargs):
            forward_calls.append(1)
            return original_forward(*args, **kwargs)

        model.forward = _spy_forward

        gen = torch.Generator().manual_seed(1)
        noise = sampler._create_initial_noise(_W2, _H2, gen)
        prompt_emb = sampler.encode_prompt("test cfg")

        sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=4.5,
            seed=1,
        )

        # 2 steps × 2 forwards per step (cond + uncond) = 4
        assert len(forward_calls) == 4, (
            f"CFG should call forward 2× per step (4 total for 2 steps), "
            f"got {len(forward_calls)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — TE warm: sample + negative prompts pre-cached during precache
# ─────────────────────────────────────────────────────────────────────────────

class TestTEWarmPrecache:
    """Verify _pre_cache_text_embeddings warms sample + negative prompts."""

    def _build_trainer_stub(self, config: dict):
        """Build a minimal Krea2Trainer-shaped stub for precache tests.

        Uses MagicMock(spec=Krea2Trainer) so real method implementations
        are callable via Krea2Trainer.<method>(trainer, ...).  Methods that
        must use the real implementation are patched via side_effect.
        """
        from app.engine.models.families.krea2.trainer import Krea2Trainer

        trainer = MagicMock(spec=Krea2Trainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.text_cache = {}
        trainer.logger = MagicMock()

        # Stub _build_caption_hints to return a minimal set (one dataset caption)
        trainer._build_caption_hints.return_value = {"a dataset caption": "hint0"}

        # Stub _resolve_te_cache_dirs to return no dirs (no disk cache)
        trainer._resolve_te_cache_dirs.return_value = []

        # Stub _resolve_loading_dtype
        trainer._resolve_loading_dtype.return_value = torch.float32

        # Stub _log_writer to None (no-op status calls)
        trainer._log_writer = None

        # text_encoder must be non-None so the method proceeds
        trainer.text_encoder = MagicMock()

        # _encode_text_direct returns (emb [B, L, 12, 2560], mask [B, L])
        # Use tiny dims in tests: L=4, 12 layers, dim=8
        def _fake_encode(captions, dtype):
            B = len(captions)
            emb = torch.zeros(B, 4, 12, 8)
            mask = torch.zeros(B, 4, dtype=torch.long)
            return emb, mask

        trainer._encode_text_direct = _fake_encode

        # _sample_prompt_texts must use the real implementation so wildcards
        # are expanded correctly and the cache key matches the sampler's request.
        trainer._sample_prompt_texts.side_effect = (
            lambda: Krea2Trainer._sample_prompt_texts(trainer)
        )

        return trainer

    def test_krea2_warms_sample_prompts_into_cache(self):
        """_pre_cache_text_embeddings must add expanded sample + negative to cache.

        This is the primary VRAM fix: before this, the 8GB Qwen3-VL TE reloaded
        from CPU on the first sampling call (cache miss). After the fix, both the
        expanded sample prompt and the negative prompt are keys in self.text_cache
        after precache completes.
        """
        from app.engine.models.families.krea2.trainer import Krea2Trainer

        config = {
            "cache_text_embeddings": True,
            "sample_prompts": [{"prompt": "a red car"}],
            "sample_negative_prompt": "blurry",
        }
        trainer = self._build_trainer_stub(config)

        # Run the real implementation
        Krea2Trainer._pre_cache_text_embeddings(trainer)

        # Both the expanded sample prompt AND the negative must be cached
        assert "a red car" in trainer.text_cache, (
            "expanded sample prompt 'a red car' must be in text_cache after precache"
        )
        assert "blurry" in trainer.text_cache, (
            "negative prompt 'blurry' must be in text_cache after precache"
        )

    def test_krea2_warms_negative_only_when_sample_prompts_present(self):
        """Negative prompt is NOT pre-cached when there are no sample prompts."""
        from app.engine.models.families.krea2.trainer import Krea2Trainer

        config = {
            "cache_text_embeddings": True,
            "sample_prompts": [],
            "sample_negative_prompt": "blurry",
        }
        trainer = self._build_trainer_stub(config)
        Krea2Trainer._pre_cache_text_embeddings(trainer)

        # With no sample prompts, negative should NOT be cached (no sampler will run)
        assert "blurry" not in trainer.text_cache, (
            "negative prompt must NOT be cached when there are no sample prompts"
        )

    def test_krea2_sample_prompt_texts_expands_wildcards(self):
        """_sample_prompt_texts must expand [triggerword] via expand_prompt_wildcards.

        expand_prompt_wildcards replaces [triggerword] with config["global_triggerword"].
        """
        from app.engine.models.families.krea2.trainer import Krea2Trainer

        trainer = MagicMock(spec=Krea2Trainer)
        trainer.config = {
            "sample_prompts": [{"prompt": "[triggerword] in space"}],
            "global_triggerword": "astronaut",
        }

        texts = Krea2Trainer._sample_prompt_texts(trainer)

        assert len(texts) == 1
        # expand_prompt_wildcards replaces [triggerword] with global_triggerword
        assert texts[0] == "astronaut in space", (
            f"wildcard not expanded: got {texts[0]!r}"
        )
