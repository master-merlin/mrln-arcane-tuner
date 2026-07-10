"""BooguImageDriver — family-specific training behavior for Boogu-Image.

Task 4 scope (this file): the REAL driver — replaces Task 2's stubs with the
load-bearing correctness contracts (time convention, timestep scale, LoRA
targeting, list-tensor I/O adapter, RoPE wiring). ``encode_text`` and
``get_saver`` were ``NotImplementedError`` — Qwen3-VL text encoding lands
in Task 5 (trainer), the safetensors saver in a later task; both are outside
this task's charter (driver forward-pass math only).

## Task 5 update — ``encode_text`` implemented + ``sample_timesteps``
``progress`` forwarding added

``encode_text`` below is the Qwen3-VL VLM forward (chat-template + system
prompt + last-layer tap) — see ``task-5-report.md`` for the full evidence
trail. Two verified facts worth flagging here because they contradict what
a literal reading of the upstream pipeline suggests:

1. **No ``.last_hidden_state`` fast path.** Upstream's
   ``_get_instruction_feature_embeds`` (``pipeline_boogu.py:1467-1490``)
   tries ``self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state``
   first and falls back to ``output_hidden_states=True`` +
   ``hidden_states[-1]`` in an ``except Exception`` handler. Verified against
   the installed ``transformers==4.57.0``: ``Qwen3VLForConditionalGeneration``
   (the class both upstream AND this loader instantiate — not a vendored
   subclass) returns ``Qwen3VLCausalLMOutputWithPast``, which has NO
   ``last_hidden_state`` field (only the inner ``Qwen3VLModel``'s own output
   does). The "fast path" therefore ALWAYS raises for this model class —
   the except-branch is the one that genuinely executes upstream. This
   driver calls that path directly instead of reproducing the try/except.
2. **``num_instruction_feature_layers`` is always 1 in practice.** Upstream
   reads ``self.transformer.instruction_feature_configs.get(
   "num_instruction_feature_layers", 1)`` — but the vendored transformer's
   own config dict key is ``"num_instruction_feat_layers"`` (no "ure"), a
   name mismatch (verified: ``transformer_boogu.py:824-828``). The ``.get()``
   call therefore always misses and falls through to its default, ``1``.
   Hardcoded here rather than threaded through a live config lookup.

``sample_timesteps`` now accepts and forwards a ``progress: float = 0.0``
keyword to ``TimestepSampler.sample`` (previously always implicitly ``0.0``
via the sampler's own default) — the trainer computes real progress from
``global_step``/``max_train_steps`` and passes it through; without this the
``radc`` curriculum timestep mode silently never advances past its
step-0 center regardless of training progress.

## Time convention (DERIVED, not assumed — see task-4-report.md for the full
evidence trail with file:line citations)

Boogu-Image is INVERTED relative to the house default (most families:
``t=0`` clean, ``t=1`` noise). Verified from the vendored scheduler
(``vendor/schedulers/scheduling_flow_match_euler_discrete_time_shifting.py``)
plus the upstream ``pipeline_boogu.py`` clone:

- ``FlowMatchEulerDiscreteScheduler.__init__`` builds
  ``timesteps = linspace(0, 1, N+1)[:-1]`` — i.e. ``[0, ..., <1)``, walking
  UP toward 1, with a synthetic ``1.0`` appended in ``set_timesteps``
  (``self._timesteps = cat([timesteps, ones(1)])``).
- ``prepare_latents`` seeds the loop with ``randn_tensor(...)`` (pure
  Gaussian noise) BEFORE the ``for i, t in enumerate(timesteps)`` denoise
  loop — i.e. sampling STARTS at the first (≈0) timestep with pure noise.
- ``scheduler.step()`` is plain forward Euler:
  ``prev_sample = sample + (t_next - t) * model_output`` with
  ``t_next > t`` (walking UP), landing on the synthetic ``t=1.0`` on the
  final step, after which ``vae.decode(latents)`` runs directly (no
  un-scaling toward "less noisy" — the loop's ENDPOINT is the clean image).

Together: ``t=0`` ~ pure noise, ``t=1`` = clean data, i.e.
``x_t = (1-t)*noise + t*x0`` and the model predicts velocity
``x0 - noise`` (data minus noise — the SIGN is flipped from the house
default ``noise - latents``). This is encoded below in ``add_noise`` /
``compute_target``, pinned exactly by the perfect-velocity round-trip test
in ``test_boogu_image_driver.py`` (runs the REAL vendored scheduler loop
with an oracle transformer standing in for the real one).

## Timestep scale (contract 2)

The transformer's ``Lumina2CombinedTimestepCaptionEmbedding`` multiplies the
incoming timestep by ``timestep_scale`` (1000 on the real checkpoint,
``transformer.timestep_scale: 1000`` in both definition YAMLs) INSIDE its
``Timesteps`` sinusoidal embedding. Boogu's OWN scheduler already lives in
``[0, 1)`` (unlike stock diffusers' ``FlowMatchEulerDiscreteScheduler``,
which lives in ``[0, 1000]``) — so driver/trainer/sampler code passes the
RAW ``[0, 1)`` ``t`` everywhere. No ``/1000``, no ``*1000`` anywhere in this
file.

## Interface handoff to Task 5 (Trainer) — READ THIS BEFORE WIRING

``PipelineBaseMixin.add_noise`` / ``.compute_target`` / ``.sample_timesteps``
(``app/engine/core/pipeline/pipeline_base.py``) hardcode the STANDARD
(non-inverted) convention via ``NoiseInterpolation("linear")`` and
``noise - latents`` and do **NOT** delegate to ``IModelDriver``'s own
``add_noise`` / ``compute_target`` / ``sample_timesteps`` methods (those are
a separate, driver-level contract with a different signature —
``sample_timesteps`` here takes ``device``/``config`` explicitly instead of
reading ``self.device``/``self.config``). ``BooguImageTrainer`` (Task 5)
MUST override ``sample_timesteps`` / ``add_noise`` / ``compute_target`` at
the TRAINER level to delegate to (or reimplement identically) this driver's
versions — leaving the ``PipelineBaseMixin`` defaults un-overridden would
silently train a pure-noise LoRA (wrong direction AND wrong scale). This
mirrors the SDXL precedent (``families/sdxl/trainer.py`` overrides all
three directly for its own, different, convention).

Similarly, ``PipelineBaseMixin.encode_text`` passes the driver's raw
``TextEncoderOutput`` dataclass straight into ``forward_pass`` as
``text_embeddings`` — but THIS driver's ``forward_pass`` (like
krea2/prx/dreamlite) expects a plain
``(instruction_hidden_states, instruction_attention_mask)`` tuple. Task 5's
``BooguImageTrainer.encode_text`` MUST unwrap
``TextEncoderOutput(embeddings=..., attention_mask=...)`` into that tuple
before calling ``forward_pass`` (the krea2 "C1/C2 fix" pattern — see
``Krea2Trainer.encode_text`` / ``._encode_text_direct``).

## forward_pass() interface for Task 5

```
forward_pass(
    noisy_input: Tensor[B, 16, H, W],       # noised latents (post add_noise)
    timesteps: Tensor[B],                    # raw [0, 1) — same t as add_noise
    text_embeddings: tuple[Tensor[B, L<=256, 4096], Tensor[B, L] | None],
    batch: dict,                             # unused (control_inputs: 0 — no ref images)
) -> Tensor[B, 16, H, W]                      # velocity prediction, x0 - noise convention
```

``instruction_hidden_states`` = the VLM's ``last_hidden_state`` (Qwen3-VL
mllm, ``te.hidden_size: 4096``, ``te.max_sequence_length: 256``).
``instruction_attention_mask`` = the processor's attention mask, same
``[B, L]`` shape, bool or int (``.sum(dim=1)`` is used internally to derive
per-sample caption lengths — must not be ``None``).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver, IModelSaver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

_NOT_IMPLEMENTED_NOTE = (
    " — lands in a later task, see .agent/workdir/sdd-boogu/task-4-report.md"
)

# RoPE base frequency — hardcoded in the vendored transformer's own
# ``rope_embedder`` construction (transformer_boogu.py:859,
# ``BooguImageDoubleStreamRotaryPosEmbed(theta=10000, ...)``) and in the
# upstream pipeline's ``get_freqs_cis(..., theta=10000)`` call
# (pipeline_boogu.py:2899). Not a definition/config knob.
_ROPE_THETA = 10000

# Verbatim from upstream pipeline_boogu.py:232
# (``self.SYSTEM_PROMPT_4_T2I_UNIFIED``, aliased as ``SYSTEM_PROMPT_4_T2I``
# at :234) — the branch ``_apply_chat_template`` (pipeline_boogu.py:1600)
# selects for a NON-EMPTY instruction with no reference images, i.e. every
# real Boogu-Image training caption (``control_inputs: 0``, pure T2I; both
# shipped definitions).
_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images "
    "based on user instructions. The instructions are as follows."
)

# Verbatim from upstream pipeline_boogu.py:231
# (``self.SYSTEM_PROMPT_4_TI2I_UNIFIED``, aliased as ``SYSTEM_PROMPT_DROP``
# at :235). ``_apply_chat_template``'s adaptive branch
# (pipeline_boogu.py:1596-1598 — the default
# ``system_prompt_follows_task_type=False`` path, defaults at :2291/:2699)
# applies THIS prompt to EVERY empty-instruction/no-image encode, including
# the plain-T2I CFG NEGATIVE (``encode_instruction`` defaults
# ``negative_instruction=""`` at :2491-2494). The base checkpoint's learned
# unconditional anchor therefore lives under this DROP prompt —
# caption-dropout training ("" captions) must encode with it, NOT the T2I
# prompt, or the LoRA's CFG semantics drift from the base model (Task-5
# review Finding 2; see task-5-report.md "Fix wave 1").
_SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)


def _select_system_prompt(caption: str) -> str:
    """Mirror upstream ``_apply_chat_template``'s adaptive no-image branch
    (pipeline_boogu.py:1596-1600): empty/whitespace-only instruction ->
    DROP prompt, otherwise the T2I prompt."""
    if caption is None or len(caption.strip()) == 0:
        return _SYSTEM_PROMPT_DROP
    return _SYSTEM_PROMPT_T2I

# te.max_sequence_length, both definitions (base.yaml / turbo.yaml).
_MAX_SEQUENCE_LENGTH = 256

# See module docstring "Task 5 update" note 2 — always 1 in practice due to
# an upstream config-key name mismatch.
_NUM_INSTRUCTION_FEATURE_LAYERS = 1


class BooguImageDriver(IModelDriver):
    """Boogu-Image family driver (Base / Turbo share the same transformer
    geometry — only the checkpoint repo and native sample defaults differ)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.processor: Any = None
        self.scheduler: Any = None
        self._components: dict[str, Any] = {}

        # Lazily-built static RoPE lookup table (independent of batch shape —
        # see ``_build_freqs_cis``). Cached per-model since it never changes
        # for a given checkpoint.
        self._freqs_cis_cache: tuple[Any, list[torch.Tensor]] | None = None

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Boogu-Image components into driver state.

        ``scheduler`` is assigned here too (Task 3 reviewer handoff note) so
        it is available before ``init_scheduler()`` is explicitly called.
        """
        self._components = components
        self.model = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.processor = components.get("processor")
        self.scheduler = components.get("scheduler")
        self._freqs_cis_cache = None

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """Boogu-Image LoRA targets — the curated definition list, VERBATIM.

        No suffix re-expansion (kandinsky5 lesson: full-path entries must
        pass through untouched or PEFT wraps zero modules). No
        attention-only fallback: a definition without a curated list would
        silently leave the double-stream blocks' PROCESSOR-owned
        ``img_instruct_attn.processor.{img,instruct}_to_{q,k,v}`` /
        ``{img,instruct}_out`` projections un-adapted (they are NOT matched
        by any generic ``attn.to_*`` pattern — see
        ``BooguImageDoubleStreamTransformerBlock``, which ``del``s the
        module-level ``attn.to_q/to_k/to_v`` and re-homes them on the
        processor). A silent wrong fallback here is exactly the
        enrichment-crash class already fixed for dreamlite/kandinsky5/etc —
        fail loudly instead.
        """
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if not definition_targets:
            raise RuntimeError(
                "boogu_image: definition ships no curated "
                "'lora_targetable_modules'. There is no safe generic "
                "fallback for this architecture — the double-stream "
                "blocks' processor-owned joint-attention projections "
                "(img_instruct_attn.processor.*) are not matched by any "
                "attn.to_* pattern, so a naive fallback would silently "
                "leave them un-adapted. Ship a curated list in the "
                "definition YAML (see definitions/base.yaml)."
            )
        self.logger.info(
            "lora_targets_from_definition", count=len(definition_targets),
        )
        return list(definition_targets)

    def init_scheduler(self) -> Any:
        """Return the LOADER-provided vendored scheduler.

        Consumes ``self._components["scheduler"]`` (checkpoint-specific
        ``do_shift`` / ``dynamic_time_shift`` / ``time_shift_version`` /
        ``seq_len`` config, loaded via the vendored class's own
        ``ConfigMixin.from_pretrained`` in ``BooguImageLoader``) — does NOT
        construct a fresh default instance, which would silently drop the
        real checkpoint's shift config.
        """
        scheduler = self._components.get("scheduler")
        if scheduler is None:
            raise RuntimeError(
                "boogu_image: init_scheduler() called with no 'scheduler' "
                "component assigned — assign_components() must run first "
                "(the scheduler comes from the loader, not a fresh default)."
            )
        self.scheduler = scheduler
        return scheduler

    def resolve_loading_dtype(self) -> torch.dtype:
        """Boogu-Image loads in bf16 (mllm + transformer + VAE all bf16)."""
        return torch.bfloat16

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Qwen3-VL full-VLM text encoding — chat template + last-layer tap.

        Mirrors upstream ``_get_instruction_feature_embeds``'s
        ``use_prompt_tuning_embedding=False`` / ``num_instruction_feature_layers
        == 1`` path (``pipeline_boogu.py:1448-1498``): build one
        ``[system, user]`` chat-template message list per caption (system
        prompt selected per caption by :func:`_select_system_prompt` —
        T2I for real captions, DROP for empty/dropout ones; no images —
        ``control_inputs: 0``), tokenize via the
        processor's ``apply_chat_template`` (stock Qwen3-VL ChatML jinja,
        attention-mask based — NO fixed-token crop), forward through the mllm
        with ``output_hidden_states=True``, and tap the LAST decoder layer
        (``hidden_states[-1]``) — see module docstring "Task 5 update" note 1
        for why this driver calls that path directly instead of upstream's
        ``.last_hidden_state`` attribute access (which does not exist on
        this exact model class).

        ``use_prompt_tuning: false`` in both shipped definitions — no
        ``PromptEmbedding`` soft-token prepending (skipped entirely, matching
        the loader's docstring).

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype for the returned embeddings.

        Returns:
            ``TextEncoderOutput`` with:
            - ``embeddings``: ``[B, L<=256, 4096]`` (``te.hidden_size``)
            - ``attention_mask``: ``[B, L]`` (processor's own mask, int/bool)
        """
        if self.processor is None or self.text_encoder is None:
            raise RuntimeError(
                "boogu_image encode_text: 'processor'/'text_encoder' not "
                "assigned — assign_components() must run first."
            )

        # Per-caption system prompt (upstream's adaptive branch): non-empty
        # caption -> T2I prompt; empty/whitespace (caption dropout / CFG
        # unconditional) -> DROP prompt, matching the base checkpoint's
        # learned unconditional anchor (review Finding 2).
        prompts = [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": _select_system_prompt(caption)},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": caption}]},
            ]
            for caption in captions
        ]

        vlm_inputs = self.processor.apply_chat_template(
            prompts,
            padding="longest",
            max_length=_MAX_SEQUENCE_LENGTH,
            truncation=False,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        vlm_inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in vlm_inputs.items()
        }

        with torch.no_grad():
            outputs = self.text_encoder(
                **vlm_inputs, output_hidden_states=True, return_dict=True,
            )

        # See module docstring "Task 5 update" note 1: Qwen3VLForConditional
        # Generation's output has no `.last_hidden_state` — the last decoder
        # layer of the `hidden_states` tuple IS what upstream's own
        # except-fallback computes, and num_instruction_feature_layers==1
        # (note 2) means we only ever need that single last layer.
        assert _NUM_INSTRUCTION_FEATURE_LAYERS == 1
        hidden_states = outputs.hidden_states[-1]
        attention_mask = vlm_inputs["attention_mask"]

        return TextEncoderOutput(
            embeddings=hidden_states.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder (Qwen3-VL mllm) LoRA not supported for Boogu-Image."""
        return []

    # --- Phase 5: Training Loop Hooks ---

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Sample raw ``[0, 1)`` timesteps — NO scaling (contract 2).

        Delegates the actual distribution to the shared ``TimestepSampler``
        (``.sample``, not ``.sample_scaled`` — the latter multiplies by
        1000, which is wrong for Boogu's own ``[0, 1)``-native scheduler).

        ``progress`` (training progress in ``[0, 1]``, ``global_step /
        max_train_steps``) is forwarded to ``TimestepSampler.sample`` — it
        drives the ``radc`` curriculum mode's center shift. The TRAINER
        (Task 5) computes and passes this; the default ``0.0`` here only
        applies to direct/legacy callers that don't supply it (e.g. Task 4's
        driver-level tests).
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler

        mode = config.get("timestep_sampling", "logit_normal")
        return TimestepSampler.sample(
            mode, batch_size, device, config, latents=latents, progress=progress,
        )

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Boogu's INVERTED flow-match lerp: ``x_t = (1-t)*noise + t*x0``.

        ``t=0`` -> pure noise, ``t=1`` -> clean latents (derived from the
        vendored scheduler + upstream pipeline — see module docstring and
        task-4-report.md for the full evidence trail).
        """
        t = timesteps
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        t = t.to(dtype=latents.dtype, device=latents.device)
        return (1.0 - t) * noise + t * latents

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity target ``x0 - noise`` — the CORRECT SIGN for Boogu's
        inverted convention (opposite of the house default ``noise - latents``).

        ``d/dt[(1-t)*noise + t*x0] = x0 - noise``, matching ``add_noise``
        exactly — the pairing pinned by the perfect-velocity round-trip test.
        """
        return latents - noise

    def _build_freqs_cis(self, model: nn.Module) -> list[torch.Tensor]:
        """Build the static RoPE lookup table from the model's own rope
        config — matches ``pipeline_boogu.py``'s
        ``BooguImageRotaryPosEmbed.get_freqs_cis(axes_dim_rope, axes_lens,
        theta=10000)`` call made ONCE before the denoise loop (contract 5).

        Purely a function of ``(axes_dim_rope, axes_lens, theta)`` — NOT of
        batch/image shapes (the batch-shape-dependent position-id gather
        happens inside the model's own ``rope_embedder.forward()``). Cached
        since it is identical across every training step for a given model.
        """
        from app.engine.models.families.boogu_image.vendor.models.transformers.rope import (
            BooguImageDoubleStreamRotaryPosEmbed,
        )

        axes_dim_rope = tuple(model.config.axes_dim_rope)
        axes_lens = tuple(model.config.axes_lens)
        cache_key = (axes_dim_rope, axes_lens, _ROPE_THETA)

        if self._freqs_cis_cache is not None and self._freqs_cis_cache[0] == cache_key:
            return self._freqs_cis_cache[1]

        freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            axes_dim_rope, axes_lens, theta=_ROPE_THETA,
        )
        self._freqs_cis_cache = (cache_key, freqs_cis)
        return freqs_cis

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """BooguImageTransformer2DModel forward via the list-tensor adapter.

        1. Unpack ``text_embeddings`` -> ``(instruction_hidden_states,
           instruction_attention_mask)`` (see module docstring — Task 5's
           trainer must supply this tuple shape, not a raw
           ``TextEncoderOutput``).
        2. List-tensor I/O adapter (contract 4): explicitly split the batched
           ``[B, C, H, W]`` ``noisy_input`` into a python list of per-sample
           ``[C, H, W]`` tensors (the model's real, variable-resolution I/O
           contract), call the model, and re-stack the list output back into
           a ``[B, C, H, W]`` tensor for the loss. Our bucketing guarantees
           equal shapes per training batch today, so this always exercises
           the equal-resolution fast path in practice — but the adapter
           itself is resolution-agnostic (a genuinely mixed-resolution batch
           round-trips through the same list path unchanged; see
           ``test_forward_pass_list_adapter_preserves_per_sample_identity``
           for the explicit round-trip proof).
        3. Raw ``[0, 1)`` timestep, broadcast to ``[B]`` (contract 2 — no
           ``/1000``/``*1000``; the transformer's own ``timestep_scale``
           config multiplies internally).
        4. Static ``freqs_cis`` RoPE lookup table from the model's own config
           (contract 5).
        5. ``ref_image_hidden_states=None`` — both shipped definitions declare
           ``control_inputs: 0`` (pure T2I, no reference-image conditioning).

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]`` (post ``add_noise``).
            timesteps: Raw ``[0, 1)`` timesteps ``[B]`` (same ``t`` as
                ``add_noise``).
            text_embeddings: ``(instruction_hidden_states [B, L<=256, 4096],
                instruction_attention_mask [B, L] | None)`` tuple.
            batch: Full batch dict (unused — no ref-image conditioning yet).

        Returns:
            Velocity prediction ``[B, 16, H, W]`` (``x0 - noise`` convention).
        """
        if isinstance(text_embeddings, tuple):
            instruction_hidden_states, instruction_attention_mask = text_embeddings
        else:
            instruction_hidden_states = text_embeddings
            instruction_attention_mask = None

        if instruction_attention_mask is None:
            raise ValueError(
                "boogu_image forward_pass: instruction_attention_mask is "
                "required (the model's rope_embedder derives per-sample "
                "caption lengths from it via .sum(dim=1)) — Task 5's "
                "encode_text must supply the processor's attention mask, "
                "not None."
            )

        model = self.get_primary_model()

        # -- Contract 4: list-tensor I/O adapter --
        batch_size = noisy_input.shape[0]
        hidden_states_list = [noisy_input[i] for i in range(batch_size)]

        # -- Contract 2: raw [0, 1) timestep, no scaling --
        timestep = timesteps.reshape(batch_size).to(dtype=noisy_input.dtype)

        # -- Contract 5: static RoPE lookup table --
        freqs_cis = self._build_freqs_cis(model)

        output = model(
            hidden_states=hidden_states_list,
            timestep=timestep,
            instruction_hidden_states=instruction_hidden_states,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=None,
            return_dict=False,
        )

        if isinstance(output, list):
            return torch.stack(output, dim=0)
        return output

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> IModelSaver:
        raise NotImplementedError("boogu_image saver" + _NOT_IMPLEMENTED_NOTE)
