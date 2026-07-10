"""boogu_image trainer tests (Task 5) — override trio + VLM TE cache.

Mirrors the krea2 trainer-test style: real trainer + real driver + a tiny
REAL vendored transformer, no GPU, no network, no downloads. Leaf
processor/text-encoder objects are hand-rolled stand-ins (never mocking the
trainer->driver seam itself — the historical bug class this whole suite
exists to prevent, see ``test_trainer_seam_contract.py``'s module docstring).

Covers the binding handoffs from task-5-brief.md:

1. Convention delegation (LOAD-BEARING): trainer-level ``add_noise`` /
   ``compute_target`` / ``sample_timesteps`` MUST delegate to the driver's
   inverted convention, or the family silently trains a pure-noise LoRA.
2. ``encode_text`` tuple contract (the krea2 C1/C2 pattern).
3. ``progress`` pass-through for the radc curriculum mode.
4. The rest of the house override trio: ``_update_primary_model`` also
   syncs ``driver.model``; ``transformer`` property never goes stale.
5. The Boogu VLM ``encode_text`` path: chat template + system prompt +
   last-layer tap, attention-mask semantics, disk-cache key template
   identity.
6. End-to-end tiny CPU training step: real PEFT-wrapped tiny transformer,
   loss.backward() reaches LoRA params, never leaks into frozen base
   weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.boogu_image.driver import BooguImageDriver
from app.engine.models.families.boogu_image.trainer import (
    BooguImageTrainer,
    _disk_cache_key,
)
from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
)
from app.engine.strategies.noise_interpolation import NoiseInterpolation

# Same divisibility-respecting tiny config as test_boogu_image_driver.py.
TINY_AXES_DIM_ROPE = (2, 2, 4)
TINY_AXES_LENS = (64, 64, 64)
TINY_HIDDEN_SIZE = 16
TINY_IN_CHANNELS = 4
TINY_INSTRUCTION_FEAT_DIM = 8


def _tiny_transformer_config(**overrides) -> dict:
    cfg = dict(
        patch_size=2,
        in_channels=TINY_IN_CHANNELS,
        out_channels=None,
        hidden_size=TINY_HIDDEN_SIZE,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=8,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        axes_dim_rope=TINY_AXES_DIM_ROPE,
        axes_lens=TINY_AXES_LENS,
        instruction_feature_configs=dict(
            instruction_feat_dim=TINY_INSTRUCTION_FEAT_DIM,
            reduce_type="mean",
            num_instruction_feat_layers=1,
        ),
        prompt_tuning_configs=dict(use_prompt_tuning=False),
        timestep_scale=1000.0,
    )
    cfg.update(overrides)
    return cfg


def _tiny_transformer(**overrides) -> BooguImageTransformer2DModel:
    model = BooguImageTransformer2DModel(**_tiny_transformer_config(**overrides))
    model.eval()
    return model


def _definition(lora_targets: list[str] | None = ()) -> MagicMock:
    d = MagicMock(spec=ModelDefinition)
    d.family = "boogu_image"
    d.id = "boogu-image-test"
    d.lora_targetable_modules = list(lora_targets) if lora_targets else lora_targets
    d.architecture_params = {}
    return d


def _fake_text(batch: int, length: int = 3, dim: int = TINY_INSTRUCTION_FEAT_DIM):
    embeds = torch.randn(batch, length, dim)
    mask = torch.ones(batch, length, dtype=torch.bool)
    return (embeds, mask)


def _trainer_shell() -> BooguImageTrainer:
    """A real ``BooguImageTrainer`` + real ``BooguImageDriver``, no heavy
    ``__init__`` (mirrors ``test_trainer_seam_contract.py``'s ``_make_trainer``)."""
    t = object.__new__(BooguImageTrainer)
    t.device = torch.device("cpu")
    t.definition = _definition()
    t.driver = BooguImageDriver(t.definition, t.device)
    t.config = {"timestep_sampling": "uniform"}
    t.logger = MagicMock()
    return t


# ── Contract 1: convention delegation is LOAD-BEARING ──────────────────────


class TestConventionDelegationLoadBearing:
    def test_inverted_trainer_override_gives_zero_loss(self):
        """A perfect-velocity oracle model + the trainer's OWN (inverted)
        add_noise/compute_target/forward_pass must round-trip to ~0 loss."""
        trainer = _trainer_shell()
        torch.manual_seed(0)
        B = 1
        x0 = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        true_velocity = x0 - noise  # t-independent by construction

        def oracle(**kwargs):
            return [true_velocity[i] for i in range(B)]

        oracle.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        trainer.driver.model = oracle

        t = torch.tensor([0.37])
        text = _fake_text(B)

        noisy = trainer.add_noise(x0, noise, t)
        target = trainer.compute_target(x0, noise, t)
        pred = trainer.forward_pass(noisy, t, text, {})
        loss = F.mse_loss(pred, target)

        assert loss.item() < 1e-8

    def test_standard_mixin_path_is_not_zero_loss(self):
        """The SAME oracle model through the un-overridden
        ``PipelineBaseMixin`` (standard-convention) path must NOT land near
        zero — proves the trainer-level override is load-bearing, not
        cosmetic."""
        trainer = _trainer_shell()
        torch.manual_seed(0)
        B = 1
        x0 = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        true_velocity = x0 - noise

        def oracle(**kwargs):
            return [true_velocity[i] for i in range(B)]

        oracle.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        trainer.driver.model = oracle
        trainer.noise_interpolation = NoiseInterpolation("linear")

        t = torch.tensor([0.37])
        text = _fake_text(B)

        # Real, un-overridden PipelineBaseMixin methods — the bug class this
        # whole test class exists to pin against (a family that forgot to
        # override would hit exactly this path with Boogu's raw [0,1) t).
        noisy_std = PipelineBaseMixin.add_noise(trainer, x0, noise, t)
        target_std = PipelineBaseMixin.compute_target(trainer, x0, noise, t)
        pred_std = trainer.forward_pass(noisy_std, t, text, {})
        loss_std = F.mse_loss(pred_std, target_std)

        assert loss_std.item() > 0.5

    def test_sample_timesteps_delegates_to_driver_raw_0_1(self):
        trainer = _trainer_shell()
        trainer.max_train_steps = 100
        trainer.global_step = 0
        t = trainer.sample_timesteps(8)
        assert t.shape == (8,)
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)

    def test_sample_timesteps_forwards_progress_to_driver(self):
        """Contract 3: progress must reach TimestepSampler.sample, not
        silently degrade to 0 — spy on the driver call."""
        trainer = _trainer_shell()
        trainer.max_train_steps = 200
        trainer.global_step = 50  # progress == 0.25

        captured = {}
        real_sample_timesteps = trainer.driver.sample_timesteps

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_sample_timesteps(*args, **kwargs)

        trainer.driver.sample_timesteps = spy
        trainer.sample_timesteps(4)

        assert captured.get("progress") == pytest.approx(0.25)

    def test_add_noise_and_compute_target_delegate_to_driver(self):
        trainer = _trainer_shell()
        x0 = torch.ones(1, 1, 2, 2)
        noise = torch.zeros(1, 1, 2, 2)
        t = torch.tensor([0.5])

        assert torch.allclose(trainer.add_noise(x0, noise, t), trainer.driver.add_noise(x0, noise, t))
        assert torch.allclose(
            trainer.compute_target(x0, noise, t), trainer.driver.compute_target(x0, noise, t),
        )
        # Sanity: inverted convention values, not the standard-convention ones.
        assert torch.allclose(trainer.add_noise(x0, noise, t), torch.full_like(x0, 0.5))
        assert torch.allclose(trainer.compute_target(x0, noise, t), x0 - noise)


# ── Contract 2: encode_text tuple contract (krea2 C1/C2) ───────────────────


class TestEncodeTextTupleContract:
    def test_encode_text_returns_2_tuple_not_text_encoder_output(self):
        trainer = _trainer_shell()
        trainer.text_cache = {"a caption": (torch.randn(3, TINY_INSTRUCTION_FEAT_DIM), torch.ones(3, dtype=torch.long))}
        trainer.text_encoder = None  # pre-cached, never touched

        out = trainer.encode_text(["a caption"], torch.float32)

        assert isinstance(out, tuple) and len(out) == 2
        emb, mask = out
        assert not isinstance(out, TextEncoderOutput)
        assert emb.shape == (1, 3, TINY_INSTRUCTION_FEAT_DIM)
        assert mask.shape == (1, 3)

    def test_encode_to_forward_real_seam(self):
        """encode_text's result must be directly consumable by forward_pass
        (the exact krea2 C1/C2 regression) — real driver.encode_text via a
        hand-rolled leaf processor/text-encoder, real forward_pass."""
        trainer = _trainer_shell()
        trainer.config = {"cache_text_embeddings": False}

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeTextEncoder(nn.Module):
            """Deliberately has NO `.last_hidden_state` on its output — mirrors
            the real Qwen3VLCausalLMOutputWithPast (see driver.py docstring
            "Task 5 update" note 1). If the driver mistakenly relied on
            `.last_hidden_state`, this fake would raise AttributeError."""

            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.proj = nn.Linear(1, dim)

            def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True):
                assert output_hidden_states is True
                B, L = input_ids.shape
                hs = tuple(torch.randn(B, L, self.dim) for _ in range(3))
                return _FakeOutput(hidden_states=hs)

        class _FakeProcessor:
            def apply_chat_template(self, prompts, **kwargs):
                assert kwargs["tokenize"] is True
                assert kwargs["return_dict"] is True
                assert kwargs["max_length"] == 256
                B = len(prompts)
                L = 5
                return {
                    "input_ids": torch.arange(B * L).reshape(B, L),
                    "attention_mask": torch.ones(B, L, dtype=torch.long),
                }

        trainer.driver.text_encoder = _FakeTextEncoder(TINY_IN_CHANNELS * 2)
        trainer.driver.processor = _FakeProcessor()
        trainer.driver.model = _tiny_transformer()
        trainer.text_encoder = trainer.driver.text_encoder

        text_emb = trainer.encode_text(["a boogu test caption"], torch.float32)
        assert isinstance(text_emb, tuple) and len(text_emb) == 2
        emb, mask = text_emb
        assert emb.ndim == 3  # [B, L, D]

        noisy = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        with torch.no_grad():
            pred = trainer.forward_pass(noisy, torch.rand(1), text_emb, {})
        assert pred.shape == noisy.shape
        assert torch.isfinite(pred).all()


# ── Contract 5: Boogu VLM encode_text path (system prompt / mask / cache) ──


class TestBooguVlmEncodeTextPath:
    def _wire(self, dim=TINY_INSTRUCTION_FEAT_DIM, seq_len=6):
        captured = {}

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(1, 1)

            def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True):
                captured["output_hidden_states"] = output_hidden_states
                B, L = input_ids.shape
                hs = tuple(torch.full((B, L, dim), float(i)) for i in range(3))
                return _FakeOutput(hidden_states=hs)

        class _FakeProcessor:
            def apply_chat_template(self, prompts, **kwargs):
                captured["prompts"] = prompts
                captured["kwargs"] = kwargs
                B = len(prompts)
                return {
                    "input_ids": torch.arange(B * seq_len).reshape(B, seq_len),
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
                }

        drv = BooguImageDriver(_definition(), torch.device("cpu"))
        drv.text_encoder = _FakeTextEncoder()
        drv.processor = _FakeProcessor()
        return drv, captured

    def test_system_prompt_is_verbatim_upstream_t2i_prompt(self):
        drv, captured = self._wire()
        drv.encode_text(["draw a cat"], torch.float32)

        messages = captured["prompts"][0]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert (
            system_msg["content"][0]["text"]
            == "You are a helpful assistant that generates high-quality images "
            "based on user instructions. The instructions are as follows."
        )
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"][0]["text"] == "draw a cat"

    def test_taps_last_hidden_state_layer_via_output_hidden_states(self):
        """num_instruction_feature_layers==1 (always, per driver.py note 2) ->
        the LAST entry of the hidden_states tuple, obtained via
        output_hidden_states=True (never a `.last_hidden_state` attribute)."""
        drv, captured = self._wire()
        out = drv.encode_text(["x"], torch.float32)

        assert captured["output_hidden_states"] is True
        # Our fake tags layer i with value i -- last layer (index 2) -> all 2.0.
        assert torch.allclose(out.embeddings.float(), torch.full_like(out.embeddings.float(), 2.0))

    def test_attention_mask_is_processor_mask_no_fixed_crop(self):
        """No krea2-style fixed-token crop -- returned length matches
        whatever the processor produced."""
        drv, captured = self._wire(seq_len=11)
        out = drv.encode_text(["a caption of some length"], torch.float32)

        assert out.embeddings.shape[1] == 11
        assert out.attention_mask.shape == (1, 11)
        assert torch.equal(out.attention_mask, torch.ones(1, 11, dtype=torch.long))

    def test_embeddings_cast_to_requested_dtype(self):
        drv, _ = self._wire()
        out = drv.encode_text(["x"], torch.float64)
        assert out.embeddings.dtype == torch.float64

    def test_max_sequence_length_256_passed_to_processor(self):
        drv, captured = self._wire()
        drv.encode_text(["x"], torch.float32)
        assert captured["kwargs"]["max_length"] == 256
        assert captured["kwargs"]["padding"] == "longest"
        assert captured["kwargs"]["truncation"] is False

    def test_encode_text_without_assign_components_raises(self):
        drv = BooguImageDriver(_definition(), torch.device("cpu"))
        with pytest.raises(RuntimeError, match="assign_components"):
            drv.encode_text(["x"], torch.float32)


class TestDiskCacheKeyTemplateIdentity:
    def test_disk_cache_key_bakes_in_template_identity(self):
        assert _disk_cache_key("a caption") != "a caption"
        assert "a caption" in _disk_cache_key("a caption")

    def test_same_caption_different_template_yields_different_hash(self):
        """Pins the collision-safety property required by task-5-brief.md:
        a future template/system-prompt version bump must never silently
        reuse a stale disk-cached embedding for the same caption text."""
        from app.engine.components.text_embeddings import TextEmbeddingCache

        caption = "a red bicycle"
        key_v1 = _disk_cache_key(caption)
        key_v2_hypothetical = "boogu_image/t2i_system_prompt/v2::" + caption

        fname_v1 = TextEmbeddingCache.caption_to_filename(key_v1)
        fname_v2 = TextEmbeddingCache.caption_to_filename(key_v2_hypothetical)
        fname_raw = TextEmbeddingCache.caption_to_filename(caption)

        assert fname_v1 != fname_v2
        assert fname_v1 != fname_raw


# ── Contract 4: _update_primary_model / transformer property ───────────────


class TestOverrideTrio:
    def test_update_primary_model_syncs_driver_and_alias(self):
        trainer = _trainer_shell()
        loaded = nn.Linear(2, 2)
        wrapped = nn.Linear(2, 2)
        trainer.driver.model = loaded
        trainer.model = loaded
        trainer.components = {"unet": loaded}

        trainer._update_primary_model(wrapped)

        assert trainer.driver.model is wrapped
        assert trainer.model is wrapped
        assert trainer.components["unet"] is wrapped
        assert trainer.transformer is wrapped

    def test_transformer_property_delegates_to_driver_model(self):
        trainer = _trainer_shell()
        m = nn.Linear(2, 2)
        trainer.driver.model = m
        assert trainer.transformer is m

    def test_transformer_property_none_when_no_driver(self):
        trainer = object.__new__(BooguImageTrainer)
        trainer.driver = None
        assert trainer.transformer is None


# ── Contract 6: end-to-end tiny CPU training step ───────────────────────────


def _attn_ff_suffixes(container) -> list[str]:
    def _is_modulation_linear(name: str) -> bool:
        parts = name.split(".")
        return len(parts) >= 2 and parts[-1] == "linear" and "norm" in parts[-2]

    return sorted(
        name
        for name, mod in container.named_modules()
        if isinstance(mod, torch.nn.Linear) and name and not _is_modulation_linear(name)
    )


def _tiny_expanded_targets(model: BooguImageTransformer2DModel) -> list[str]:
    targets: list[str] = []
    for suf in _attn_ff_suffixes(model.double_stream_layers[0]):
        targets.append(f"double_stream_layers.0.{suf}")
    for suf in _attn_ff_suffixes(model.single_stream_layers[0]):
        targets.append(f"single_stream_layers.0.{suf}")
    for container_name in ("noise_refiner", "ref_image_refiner", "context_refiner"):
        for suf in _attn_ff_suffixes(model.single_stream_layers[0]):
            targets.append(f"{container_name}.0.{suf}")
    return targets


class TestEndToEndTrainingStep:
    def test_backward_reaches_lora_params_never_leaks_into_base_weights(self):
        from peft import LoraConfig, get_peft_model

        model = _tiny_transformer()
        # AdaLN-Zero DiT design: the modulation MLP that produces each
        # block's residual gate is zero-initialized by construction, so at
        # a pure random init EVERY attn/ff gradient (LoRA or not) is
        # exactly zero on step 0 -- a well-known DiT init property, not a
        # trainer/driver bug. Re-randomize to exercise a non-degenerate step.
        torch.manual_seed(0)
        for p in model.parameters():
            nn.init.normal_(p, std=0.02)
        model.train()

        targets = _tiny_expanded_targets(model)
        peft_model = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=4, target_modules=targets),
        )
        peft_model.train()

        definition = _definition(lora_targets=targets)
        driver = BooguImageDriver(definition, torch.device("cpu"))
        driver.assign_components({
            "unet": peft_model, "vae": None, "text_encoder": None,
            "processor": None, "scheduler": None,
        })

        trainer = object.__new__(BooguImageTrainer)
        trainer.device = torch.device("cpu")
        trainer.definition = definition
        trainer.driver = driver
        trainer.model = peft_model
        trainer.config = {"timestep_sampling": "uniform"}

        B = 2
        x0 = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        text = _fake_text(B)

        timesteps = trainer.driver.sample_timesteps(B, torch.device("cpu"), trainer.config)
        noisy = trainer.add_noise(x0, noise, timesteps)
        target = trainer.compute_target(x0, noise, timesteps)
        pred = trainer.forward_pass(noisy, timesteps, text, {})
        loss = F.mse_loss(pred, target)
        loss.backward()

        lora_nonzero = 0
        base_leak: list[str] = []
        for name, p in peft_model.named_parameters():
            has_grad_signal = p.grad is not None and p.grad.abs().sum().item() > 0
            if "lora_A" in name or "lora_B" in name:
                if has_grad_signal:
                    lora_nonzero += 1
            elif has_grad_signal:
                base_leak.append(name)

        assert lora_nonzero > 0, "no LoRA param received a gradient -- graph is disconnected"
        assert not base_leak, f"gradient leaked into frozen base weights: {base_leak}"

    def test_frozen_base_weights_never_require_grad_after_peft_wrap(self):
        from peft import LoraConfig, get_peft_model

        model = _tiny_transformer()
        targets = _tiny_expanded_targets(model)
        peft_model = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=2, target_modules=targets),
        )

        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad
            else:
                assert not p.requires_grad, f"base weight trainable: {name}"
