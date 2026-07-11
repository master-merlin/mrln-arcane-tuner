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
            axes_dim_rope=TINY_AXES_DIM_ROPE,
            axes_lens=TINY_AXES_LENS,
        )
        trainer.driver.model = oracle

        t = torch.tensor([0.37])
        text = _fake_text(B)

        noisy = trainer.add_noise(x0, noise, t)
        target = trainer.compute_target(x0, noise, t)
        pred = trainer.forward_pass(noisy, t, text, {})
        loss = F.mse_loss(pred, target)

        assert loss.item() < 1e-8

    def test_base_mixin_path_autodelegates_to_driver_convention(self):
        """W5-1 structural cure: the SAME oracle model through the un-overridden
        ``PipelineBaseMixin`` base defaults now ALSO lands near ~0 loss — because
        the base ``add_noise``/``compute_target`` auto-delegate to
        ``BooguImageDriver`` (which meaningfully overrides both) instead of the
        old standard-convention ``NoiseInterpolation('linear')`` + ``noise -
        latents`` path.

        BEFORE W5-1 this same call landed at HIGH loss (Boogu's raw ``[0,1)`` t
        run through the wrong standard convention) — that was the dead-dispatch
        trap this class pinned. The trainer-level override (exercised by
        :meth:`test_inverted_trainer_override_gives_zero_loss`) is now
        redundant-but-harmless: even a family that ``forgot`` to override would
        get the driver's convention via auto-delegation, so this wrong path is
        structurally unreachable."""
        trainer = _trainer_shell()
        torch.manual_seed(0)
        B = 1
        x0 = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(B, TINY_IN_CHANNELS, 4, 4)
        true_velocity = x0 - noise

        def oracle(**kwargs):
            return [true_velocity[i] for i in range(B)]

        oracle.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE,
            axes_lens=TINY_AXES_LENS,
        )
        trainer.driver.model = oracle
        trainer.noise_interpolation = NoiseInterpolation("linear")

        t = torch.tensor([0.37])
        text = _fake_text(B)

        # Un-overridden PipelineBaseMixin base defaults — post-W5-1 these
        # auto-delegate to the driver's Boogu convention, so the previously-wrong
        # standard-convention path is no longer reachable here.
        noisy_std = PipelineBaseMixin.add_noise(trainer, x0, noise, t)
        target_std = PipelineBaseMixin.compute_target(trainer, x0, noise, t)
        pred_std = trainer.forward_pass(noisy_std, t, text, {})
        loss_std = F.mse_loss(pred_std, target_std)

        # Matches the driver-convention round-trip (test_inverted_...) exactly.
        assert loss_std.item() < 1e-8

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

        assert torch.allclose(
            trainer.add_noise(x0, noise, t), trainer.driver.add_noise(x0, noise, t)
        )
        assert torch.allclose(
            trainer.compute_target(x0, noise, t),
            trainer.driver.compute_target(x0, noise, t),
        )
        # Sanity: inverted convention values, not the standard-convention ones.
        assert torch.allclose(trainer.add_noise(x0, noise, t), torch.full_like(x0, 0.5))
        assert torch.allclose(trainer.compute_target(x0, noise, t), x0 - noise)


# ── Contract 2: encode_text tuple contract (krea2 C1/C2) ───────────────────


class TestEncodeTextTupleContract:
    def test_encode_text_returns_2_tuple_not_text_encoder_output(self):
        trainer = _trainer_shell()
        trainer.text_cache = {
            "a caption": (
                torch.randn(3, TINY_INSTRUCTION_FEAT_DIM),
                torch.ones(3, dtype=torch.long),
            )
        }
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

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ):
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
    def _wire(self, dim=TINY_INSTRUCTION_FEAT_DIM, seq_len=6, architecture_params=None):
        captured = {}

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(1, 1)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ):
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

        definition = _definition()
        if architecture_params is not None:
            definition.architecture_params = architecture_params
        drv = BooguImageDriver(definition, torch.device("cpu"))
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
        assert torch.allclose(
            out.embeddings.float(), torch.full_like(out.embeddings.float(), 2.0)
        )

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

    def test_max_sequence_length_read_from_definition_architecture_params(self):
        """Task 5 review minor: ``te.max_sequence_length`` must come from
        the definition (``architecture_params``), not a hardcoded module
        constant — a definition shipping a different VLM context window
        must be honored without a code change."""
        drv, captured = self._wire(
            architecture_params={"te.max_sequence_length": 128},
        )
        drv.encode_text(["x"], torch.float32)
        assert captured["kwargs"]["max_length"] == 128

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
        key_current = _disk_cache_key(caption)
        key_other_version = "boogu_image/some_other_template_version::" + caption

        fname_current = TextEmbeddingCache.caption_to_filename(key_current)
        fname_other = TextEmbeddingCache.caption_to_filename(key_other_version)
        fname_raw = TextEmbeddingCache.caption_to_filename(caption)

        assert fname_current != fname_other
        assert fname_current != fname_raw

    def test_template_id_bumped_past_v1(self):
        """Fix wave 1 (Finding 2): the v1 template (T2I prompt for ALL
        captions, including dropout) produced embeddings that are now WRONG
        for empty captions — the disk-cache key must have moved off the v1
        identity so no stale v1 embedding is ever silently reused."""
        from app.engine.components.text_embeddings import TextEmbeddingCache

        caption = "a red bicycle"
        key_now = _disk_cache_key(caption)
        key_v1_legacy = f"boogu_image/t2i_system_prompt/v1::{caption}"

        assert key_now != key_v1_legacy
        assert TextEmbeddingCache.caption_to_filename(
            key_now
        ) != TextEmbeddingCache.caption_to_filename(key_v1_legacy)

    def test_template_id_derived_from_prompt_texts(self):
        """The template id embeds a fingerprint HASHED FROM the actual
        system-prompt strings — editing either prompt text changes every
        disk-cache key automatically, so a future prompt tweak can never
        silently forget the version bump. Sourced via the driver's PUBLIC
        ``te_template_fingerprint()`` helper (reviewer minor #3) rather than
        the trainer reaching into the driver's private
        ``_SYSTEM_PROMPT_*`` constants to recompute it itself."""
        from app.engine.models.families.boogu_image.driver import (
            te_template_fingerprint,
        )
        from app.engine.models.families.boogu_image.trainer import _TE_TEMPLATE_ID

        assert te_template_fingerprint() in _TE_TEMPLATE_ID


# ── Fix wave 1, Finding 2: caption-dropout DROP system prompt ───────────────


class TestDropoutSystemPromptSelection:
    """Upstream adjudication (pipeline_boogu.py:235 + :1596-1598): EVERY
    empty-instruction/no-image encode — including the plain-T2I CFG negative
    (``encode_instruction`` defaults ``negative_instruction=""`` at
    :2491-2494; ``system_prompt_follows_task_type`` defaults ``False`` at
    :2291/:2699) — uses ``SYSTEM_PROMPT_DROP`` (= ``SYSTEM_PROMPT_4_TI2I_
    UNIFIED``, a DIFFERENT text from the T2I prompt). The base checkpoint's
    learned unconditional anchor therefore lives under the DROP prompt; our
    caption-dropout ``""`` encodes must match it or CFG semantics drift."""

    _DROP_TEXT = (
        "Describe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency "
        "with the original input where appropriate."
    )
    _T2I_TEXT = (
        "You are a helpful assistant that generates high-quality images "
        "based on user instructions. The instructions are as follows."
    )

    def _wire(self):
        captured = {}

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(1, 1)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ):
                B, L = input_ids.shape
                hs = tuple(
                    torch.randn(B, L, TINY_INSTRUCTION_FEAT_DIM) for _ in range(2)
                )
                return _FakeOutput(hidden_states=hs)

        class _FakeProcessor:
            def apply_chat_template(self, prompts, **kwargs):
                captured["prompts"] = prompts
                B, L = len(prompts), 4
                return {
                    "input_ids": torch.arange(B * L).reshape(B, L),
                    "attention_mask": torch.ones(B, L, dtype=torch.long),
                }

        drv = BooguImageDriver(_definition(), torch.device("cpu"))
        drv.text_encoder = _FakeTextEncoder()
        drv.processor = _FakeProcessor()
        return drv, captured

    def _system_text(self, messages) -> str:
        return next(m for m in messages if m["role"] == "system")["content"][0]["text"]

    def test_empty_caption_uses_verbatim_drop_prompt(self):
        drv, captured = self._wire()
        drv.encode_text([""], torch.float32)
        assert self._system_text(captured["prompts"][0]) == self._DROP_TEXT

    def test_whitespace_only_caption_uses_drop_prompt(self):
        """Mirrors upstream's ``len(instruction.strip()) == 0`` test
        (pipeline_boogu.py:1597) — whitespace-only counts as empty."""
        drv, captured = self._wire()
        drv.encode_text(["   \n\t "], torch.float32)
        assert self._system_text(captured["prompts"][0]) == self._DROP_TEXT

    def test_non_empty_caption_still_uses_t2i_prompt(self):
        drv, captured = self._wire()
        drv.encode_text(["draw a cat"], torch.float32)
        assert self._system_text(captured["prompts"][0]) == self._T2I_TEXT

    def test_mixed_batch_selects_per_caption(self):
        drv, captured = self._wire()
        drv.encode_text(["draw a cat", ""], torch.float32)
        assert self._system_text(captured["prompts"][0]) == self._T2I_TEXT
        assert self._system_text(captured["prompts"][1]) == self._DROP_TEXT


# ── Fix wave 1, Finding 1: ragged-length TE cache entries ───────────────────


class TestRaggedLengthCachePath:
    """Boogu is the FIRST variable-length family on this cache pattern
    (``padding="longest"``, no fixed crop — krea2 crops to 34 tokens,
    longcat pads to a fixed 512). Cache entries have per-caption lengths,
    so the reassembly ``torch.stack`` must pad to the batch max
    (mask-aware) or any real batch whose captions tokenize to different
    lengths crashes with RuntimeError at step 1."""

    def _wire_trainer(self, dim=TINY_INSTRUCTION_FEAT_DIM):
        """Real trainer + real driver + a fake processor whose tokenized
        length DEPENDS ON the caption (word count + 2), so different
        captions genuinely produce different-length cache entries."""

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(1, 1)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ):
                B, L = input_ids.shape
                torch.manual_seed(int(input_ids.sum().item()) % 10_000)
                hs = tuple(torch.randn(B, L, dim) for _ in range(2))
                return _FakeOutput(hidden_states=hs)

        class _FakeProcessor:
            def apply_chat_template(self, prompts, **kwargs):
                B = len(prompts)
                lengths = []
                for messages in prompts:
                    user = next(m for m in messages if m["role"] == "user")
                    caption = user["content"][0]["text"]
                    lengths.append(len(caption.split()) + 2)
                L = max(lengths)
                mask = torch.zeros(B, L, dtype=torch.long)
                for i, li in enumerate(lengths):
                    mask[i, :li] = 1
                return {
                    "input_ids": torch.arange(B * L).reshape(B, L),
                    "attention_mask": mask,
                }

        trainer = _trainer_shell()
        trainer.config = {"cache_text_embeddings": True}
        trainer.text_cache = {}
        trainer.driver.text_encoder = _FakeTextEncoder()
        trainer.driver.processor = _FakeProcessor()
        trainer.text_encoder = trainer.driver.text_encoder
        return trainer

    def test_ragged_batch_through_real_cache_path_reaches_forward_pass(self):
        """The exact crash mode under review: cache miss -> per-caption
        entries of DIFFERENT lengths -> cache hit -> ``torch.stack`` over
        ragged ``[L_i, D]`` entries raises RuntimeError. Must instead pad
        to the batch max with mask=0 at padded positions and flow through
        the REAL forward_pass."""
        trainer = self._wire_trainer()
        short_cap = "cat"  # 1 word  -> L=3
        long_cap = "a much longer caption with many words"  # 7 words -> L=9

        # Cache-miss pass (per-caption single encodes populate the cache).
        emb, mask = trainer.encode_text([short_cap, long_cap], torch.float32)

        L_max = 9
        assert emb.shape == (2, L_max, TINY_INSTRUCTION_FEAT_DIM)
        assert mask.shape == (2, L_max)
        # Padded positions carry mask=0; real positions mask=1.
        assert mask[0, :3].bool().all() and not mask[0, 3:].bool().any()
        assert mask[1].bool().all()

        # Cache-HIT pass must produce the identical batch (pure reassembly).
        emb2, mask2 = trainer.encode_text([short_cap, long_cap], torch.float32)
        assert torch.allclose(emb, emb2)
        assert torch.equal(mask, mask2)

        # And the stacked batch must be consumable by the REAL forward_pass.
        trainer.driver.model = _tiny_transformer()
        noisy = torch.randn(2, TINY_IN_CHANNELS, 4, 4)
        with torch.no_grad():
            pred = trainer.forward_pass(noisy, torch.rand(2), (emb2, mask2), {})
        assert pred.shape == noisy.shape
        assert torch.isfinite(pred).all()

    def test_cache_entries_are_trimmed_to_true_length(self):
        """Entries must be stored TRIMMED to their mask length (the
        kandinsky5 precedent) so reassembly padding is well-defined and a
        given caption's cached entry is independent of whichever batch it
        happened to be first encoded in."""
        trainer = self._wire_trainer()
        trainer.encode_text(
            ["cat", "a much longer caption with many words"],
            torch.float32,
        )

        emb_short, mask_short = trainer.text_cache["cat"]
        assert emb_short.shape[0] == 3
        assert mask_short.shape[0] == 3
        assert mask_short.bool().all()

    def test_pre_cache_sub_batch_entries_are_trimmed_consistently(self):
        """``_pre_cache_text_embeddings`` encodes in sub-batches of 4 —
        each sub-batch pads to ITS OWN max, so un-trimmed entries would
        carry inconsistent cross-sub-batch padding. Entries must land in
        ``text_cache`` trimmed to true length, exactly like the lazy path."""
        trainer = self._wire_trainer()
        trainer.config = {
            "cache_text_embeddings": True,
            "te_quantization": "none",
            "sample_prompts": [],
        }
        trainer._log_writer = None
        trainer._resolve_te_cache_dirs = lambda: []
        caps = {
            "cat": "img_a",
            "a much longer caption with many words": "img_b",
        }
        trainer._build_caption_hints = lambda: caps
        trainer._resolve_loading_dtype = lambda: torch.float32

        trainer._pre_cache_text_embeddings()

        assert trainer.text_cache["cat"][0].shape[0] == 3
        assert (
            trainer.text_cache["a much longer caption with many words"][0].shape[0] == 9
        )

        # Reassembly across the two lengths must not crash and must pad.
        emb, mask = trainer.encode_text(list(caps), torch.float32)
        assert emb.shape == (2, 9, TINY_INSTRUCTION_FEAT_DIM)
        assert not mask[0, 3:].bool().any()


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
            model,
            LoraConfig(r=4, lora_alpha=4, target_modules=targets),
        )
        peft_model.train()

        definition = _definition(lora_targets=targets)
        driver = BooguImageDriver(definition, torch.device("cpu"))
        driver.assign_components(
            {
                "unet": peft_model,
                "vae": None,
                "text_encoder": None,
                "processor": None,
                "scheduler": None,
            }
        )

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

        timesteps = trainer.driver.sample_timesteps(
            B, torch.device("cpu"), trainer.config
        )
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

        assert lora_nonzero > 0, (
            "no LoRA param received a gradient -- graph is disconnected"
        )
        assert not base_leak, f"gradient leaked into frozen base weights: {base_leak}"

    def test_frozen_base_weights_never_require_grad_after_peft_wrap(self):
        from peft import LoraConfig, get_peft_model

        model = _tiny_transformer()
        targets = _tiny_expanded_targets(model)
        peft_model = get_peft_model(
            model,
            LoraConfig(r=2, lora_alpha=2, target_modules=targets),
        )

        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad
            else:
                assert not p.requires_grad, f"base weight trainable: {name}"
