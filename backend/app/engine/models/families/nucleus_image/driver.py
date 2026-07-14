"""NucleusImageDriver — family-specific training behavior for Nucleus-Image.

Implements ``IModelDriver`` for NucleusAI's 17B sparse-MoE DiT (~2B active
per token; ``NucleusMoEImageTransformer2DModel``, diffusers 0.39, native —
no vendoring) with a frozen Qwen3-VL text encoder
(``transformers.Qwen3VLForConditionalGeneration``) and the
``AutoencoderKLQwenImage`` VAE (shared with ``qwen_image``). Apache-2.0.

All facts below verified against the INSTALLED package (not the HF blog,
which is vague/wrong about "native diffusers pipeline integration" being a
``trust_remote_code`` situation — it is not):
``venv/Lib/site-packages/diffusers/models/transformers/
transformer_nucleusmoe_image.py`` and
``venv/Lib/site-packages/diffusers/pipelines/nucleusmoe_image/
pipeline_nucleusmoe_image.py`` (diffusers 0.39.0, read in full, 2026-07-13),
cross-checked against the live HF repo ``NucleusAI/Nucleus-Image`` config
JSONs (``transformer/config.json``, ``text_encoder/config.json``,
``scheduler/scheduler_config.json``, ``vae/config.json``, HF tree API for
file sizes — fetched directly, no download needed).

Nucleus-Image specifics mirrored here:

1. **Chat-template prompt formatting, NO positive/negative asymmetry**
   (``pipeline_nucleusmoe_image.py`` lines 41, 178-185, 502-520):
   ``NucleusMoEImagePipeline._format_prompt`` wraps EVERY prompt — both the
   real caption AND the CFG negative/uncond prompt, via the SAME
   ``encode_prompt`` call site — in a system+user chat message pair, then
   ``self.processor.apply_chat_template(messages, tokenize=False,
   add_generation_prompt=True)``::

       messages = [
           {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
           {"role": "user", "content": [{"type": "text", "text": prompt}]},
       ]

   Verbatim system prompt (``pipeline_nucleusmoe_image.py`` line 41)::

       "You are an image generation assistant. Follow the user's prompt
        literally. Pay careful attention to spatial layout: objects
        described as on the left must appear on the left, on the right on
        the right. Match exact object counts and assign colors to the
        correct objects."

   UNLIKE ``lumina2`` (whose ``Lumina2Pipeline`` prefixes ONLY the positive
   prompt and encodes the negative prompt raw — see ``lumina2/driver.py``
   module docstring §1), Nucleus's ``__call__`` calls the identical
   ``self.encode_prompt(prompt=...)`` for BOTH ``prompt`` and
   ``negative_prompt`` (lines 502-520) — there is no ``system_prompt``
   parameter to disable it anywhere in ``encode_prompt``'s signature. This
   driver therefore has ONE ``encode_text`` method used for every caption,
   real or negative — no ``apply_system_prompt`` flag, no separate uncond
   cache needed (a caption and a negative prompt with identical text
   produce IDENTICAL embeddings here, so a shared cache is not just safe
   but correct).

2. **Prompt embedding tap** (``pipeline_nucleusmoe_image.py`` lines 174-176,
   215, 233-235): ``self.default_return_index = -8`` in ``__init__``;
   ``encode_prompt`` forwards through Qwen3-VL with ``use_cache=False``,
   ``output_hidden_states=True`` and takes ``outputs.hidden_states[-8]`` — a
   mid-network layer, NOT ``last_hidden_state``/``hidden_states[-1]``.
   Tokenization goes through the PROCESSOR object (not a bare tokenizer):
   ``self.processor(text=formatted, padding="longest", pad_to_multiple_of=8,
   max_length=max_sequence_length, truncation=True,
   return_attention_mask=True, return_tensors="pt")``.
   ``max_sequence_length`` DEFAULTS TO 1024
   (``self.default_max_sequence_length = 1024``, line 175) — NOT 512 as the
   ``__call__`` docstring claims (line 450, a stale/wrong upstream docstring
   — the actual runtime default, read directly from ``__init__``, is 1024).

3. **RAGGED (variable-length) padding — ``padding="longest"``, not
   ``"max_length"``**: unlike ``lumina2`` (fixed ``padding="max_length"``,
   trivially cacheable) this pipeline pads each ENCODE CALL to its own
   batch's longest prompt (rounded up to a multiple of 8), exactly the same
   situation ``qwen_image`` hit (see ``qwen_image/trainer.py``
   ``_trim_entry``/``_get_cached_text_embeddings`` docstrings, "W3-4"). This
   driver's ``encode_text`` is called once per single-caption
   (batch-size-1) cache-fill call by the trainer, so ``pad_to_multiple_of=8``
   still produces a per-caption length that varies caption-to-caption — the
   TRAINER (not this driver) is responsible for trimming each cached entry
   to its true (mask-derived) length and re-padding to the batch max at
   retrieval time, mirroring ``QwenImageTrainer._trim_entry`` verbatim
   (see ``trainer.py``).

4. **Timestep — normalized [0,1], NOT reversed; output NEGATED**
   (``pipeline_nucleusmoe_image.py`` lines 570-599) — the #1 silent-LoRA-
   killer risk for this family, and the inverse mistake from ``lumina2``'s
   (which reverses the timestep). Read verbatim::

       timestep = t.expand(latents.shape[0]).to(latents.dtype)
       noise_pred = self.transformer(
           hidden_states=latents,
           timestep=timestep / self.scheduler.config.num_train_timesteps,
           ...
       )[0]
       ...  # (CFG combine, if any)
       noise_pred = -noise_pred
       latents = self.scheduler.step(noise_pred, t, latents, ...)

   The transformer's OWN ``timestep`` input uses the STANDARD (non-reversed)
   convention: raw scheduler ``t`` in ``[0, 1000]`` divided by
   ``num_train_timesteps`` (1000) straight into ``[0, 1]`` — ``t=1000``
   (pure noise) maps to ``timestep=1.0``, ``t=0`` (clean) maps to
   ``timestep=0.0``. Do NOT reverse it (that would be copying ``lumina2``'s
   quirk onto a family that does not have it).

   The raw transformer output IS, however, negated before ``scheduler.step``
   — exactly the same negation shape as ``lumina2``'s (module docstring §3
   there), just WITHOUT the accompanying timestep reversal. Reasoning: this
   project's default ``add_noise`` is
   ``(1-t)*latents + t*noise`` (``NoiseInterpolation._linear``, t=timesteps/
   1000) and default ``compute_target`` is ``noise - latents`` (project-wide
   flow-matching convention, unmodified — see ``IModelDriver.compute_target``).
   Since the pipeline negates the raw model output right before feeding the
   scheduler (which itself expects ``noise - latents``-direction input to
   step FROM noise TOWARD the clean sample as sigma decreases), the
   transformer's OWN natural output convention must be
   ``-(noise - latents) = latents - noise``. This driver replicates BOTH
   halves in ``forward_pass``: feed the transformer
   ``timesteps / 1000.0`` (NOT reversed), and return ``-pred`` (NOT the raw
   model output). With that, no override of ``add_noise``,
   ``sample_timesteps``, or ``compute_target`` is needed — the base
   ``IModelDriver`` flow-match defaults already match once the negation
   happens inside this one hook. Pinned by
   ``backend/app/engine/tests/test_nucleus_image_family.py``.

5. **Packed (2x2-patchified) latents, symmetric channel count**
   (``pipeline_nucleusmoe_image.py`` lines 302-328, 522-523;
   ``transformer/config.json``: ``in_channels: 64, out_channels: 16,
   patch_size: 2``): the transformer consumes PACKED hidden states
   ``[B, (H/2)*(W/2), 64]`` (``16 * 2 * 2 = 64``, symmetric with the VAE's
   16-channel latents) and emits ``[B, (H/2)*(W/2), 64]`` which unpacks back
   to ``[B, 16, H, W]`` — identical patchify/unpatchify math to
   ``qwen_image``'s driver (`` num_channels_latents = in_channels // 4``).
   This driver reuses that exact pack/unpack shape math.

6. **Text tokens are KV-only** (confirmed by reading BOTH
   ``NucleusMoEAttnProcessor2_0.__call__`` AND the ``Attention`` constructor
   logic it depends on, ``diffusers/models/attention_processor.py`` lines
   256-276): each block's ``self.attn = Attention(..., added_kv_proj_dim=dim,
   context_pre_only=None, ...)``. Passing ``context_pre_only=None`` (a
   Python ``None``, distinct from ``True``/``False``) means the
   ``Attention.__init__`` branch that creates ``add_q_proj`` is SKIPPED
   entirely (``if self.context_pre_only is not None:`` at line 259 is
   ``False``) and ``to_add_out`` is never created either (line 273's
   ``self.context_pre_only is not None and not self.context_pre_only`` is
   also ``False``) — only ``add_k_proj``/``add_v_proj`` exist. The processor
   itself never references ``attn.add_q_proj``/``attn.to_add_out``: image
   queries attend to ``joint_key = cat([img_key, txt_key])`` /
   ``joint_value = cat([img_value, txt_value])``; there is no text query and
   no text-side attention output. Per-block, raw TE hidden states are first
   projected via the block's OWN ``self.encoder_proj = nn.Linear(
   joint_attention_dim, dim)`` (called ``context = self.encoder_proj(
   encoder_hidden_states)`` inside ``NucleusMoEImageTransformerBlock.forward``)
   before reaching ``add_k_proj``/``add_v_proj`` inside the attention module —
   this driver passes the SAME raw (pre-``txt_norm``, pre-``encoder_proj``)
   cached TE hidden states the pipeline uses as ``prompt_embeds``, letting
   the transformer's own internal ``txt_norm``/``encoder_proj``/
   ``add_k_proj``/``add_v_proj`` (all in the trainable/gradient graph) do the
   rest — the standard per-caption TE-cache seam (caching the plain
   ``hidden_states[return_index]`` tensor) is exactly sufficient, unmodified
   in kind; no new caching primitive needed.

7. **LoRA targets — CONTROLLER-PINNED scope, narrower than ai-toolkit's**
   (task brief decision #1, cites recon §4): **attention + shared/dense FFN
   ``nn.Linear`` modules ONLY.** The MoE router gate (``NucleusMoELayer.gate
   = nn.Linear(hidden_size * 2, num_experts, bias=False)``) is EXCLUDED by
   construction — none of this driver's target patterns match its module
   name (``transformer_blocks.N.img_mlp.gate``), pinned by
   ``test_lora_targets_never_match_router_gate`` in the family test module.
   The 64 routed experts (``SwiGLUExperts.gate_up_proj``/``down_proj``, raw
   ``nn.Parameter`` tensors, NOT ``nn.Linear`` modules — confirmed reading
   ``transformer_nucleusmoe_image.py`` lines 391-392) cannot be PEFT-wrapped
   by name-based targeting regardless and stay frozen either way.

   This is DELIBERATELY NARROWER than the superset ai-toolkit's own Nucleus
   loader would touch (recon §4: ai-toolkit's generic "wrap every
   ``nn.Linear`` under the transformer class" walker would ALSO pick up the
   per-block ``encoder_proj``, ``img_mod`` AdaLN modulation ``nn.Linear``,
   and the top-level ``img_in``/``proj_out``/``norm_out.linear`` — and
   leaves the router ``gate`` Linear ambiguous, since ai-toolkit sets no
   explicit ``ignore_if_contains`` for it). The controller's rationale:
   minimize LoRA-perturbation surface on a genuinely new (for this
   codebase) MoE architecture where the router's routing decision depends
   on the UNMODULATED post-attention hidden state (recon risk #2:
   ``NucleusMoELayer.forward``'s ``router_input = cat([timestep_emb,
   hidden_states_unmodulated])``) — training LESS of the surrounding
   architecture reduces the chance of shifting expert-choice routing
   dynamics mid-training, a failure mode with no prior precedent in this
   codebase (every earlier DiT family here is dense). Targets:

       attn.to_q, attn.to_k, attn.to_v, attn.to_out.0,
       attn.add_k_proj, attn.add_v_proj,
       img_mlp.net.0.proj, img_mlp.net.2,           # 3 dense blocks (idx 0-2)
       shared_expert.net.0.proj, shared_expert.net.2,  # 29 MoE blocks (idx 3-31)

   ``img_mlp.net.0.proj``/``img_mlp.net.2`` and ``shared_expert.net.0.proj``/
   ``shared_expert.net.2`` are UNAMBIGUOUS suffix patterns: dense blocks'
   ``img_mlp`` IS a ``FeedForward(activation_fn="swiglu")`` (``net.0`` =
   ``SwiGLU`` sub-module with a ``.proj`` Linear, ``net.2`` = the output
   Linear — confirmed ``diffusers/models/attention.py`` lines 1720-1731,
   1725-1731, and ``diffusers/models/activations.py`` lines 137-140); MoE
   blocks' ``img_mlp`` is a ``NucleusMoELayer`` (no ``.net`` attribute of its
   own) whose ``shared_expert`` is ITSELF a ``FeedForward(activation_fn=
   "swiglu")`` with the identical ``net.0.proj``/``net.2`` internal naming.
   Neither pattern's suffix ever collides with the other block type's
   module tree (a dense block has no ``shared_expert``; a MoE block's
   ``img_mlp`` has no bare ``net`` attribute), verified by live
   introspection in ``test_nucleus_image_family.py``.

8. **MoE forward — use the native class as-is, do not force the fallback**
   (task brief decision #2, recon caveat 3): ``SwiGLUExperts.forward``
   dispatches on ``self.use_grouped_mm`` — ``_run_experts_grouped_mm``
   (``torch.nn.functional.grouped_mm``, added in PyTorch 2.11, present in
   this repo's local venv — ``torch 2.12.1+cu130`` →
   ``hasattr(F, 'grouped_mm') is True``) when the checkpoint's shipped
   ``use_grouped_mm: true`` is honored, else the pure-Python
   ``_run_experts_for_loop`` (fully differentiable, correctness-safe,
   `.tolist()`-synced but no crash). This driver does NOT monkeypatch
   ``use_grouped_mm`` or force the loop fallback — the native class's own
   dispatch is used unmodified. DOCKER CAVEAT (recon caveat 3, un-verified
   by this task — GPU/docker UAT item): MRLN's Docker image bakes torch
   2.11.0 (the version ``grouped_mm`` first landed in publicly per recon),
   which SHOULD carry ``F.grouped_mm`` on the cu128 default variant, but has
   NOT been GPU-UAT'd for this specific op on the cu126/older-driver
   fallback path. If it is ever missing there, the model's own
   ``hasattr(F, "grouped_mm")``-gated fallback keeps training correct
   (just slower) — no code change needed on our side either way, but this
   must be checked once at GPU/docker UAT time, not assumed.

9. **LoRA saver / ComfyUI status** — see ``saver.py`` module docstring for
   the full evidence-cited decision (no ``NucleusMoEImageLoraLoaderMixin``
   exists in diffusers 0.39.0; ComfyUI has ZERO Nucleus support at all,
   verified against live ``comfy/lora.py``/``comfy/supported_models.py``).
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# Verbatim system prompt — ``NucleusMoEImagePipeline`` module-level constant
# ``DEFAULT_SYSTEM_PROMPT`` (pipeline_nucleusmoe_image.py line 41). Applied
# to EVERY prompt (positive AND negative — see module docstring §1).
NUCLEUS_SYSTEM_PROMPT = (
    "You are an image generation assistant. Follow the user's prompt "
    "literally. Pay careful attention to spatial layout: objects described "
    "as on the left must appear on the left, on the right on the right. "
    "Match exact object counts and assign colors to the correct objects."
)

# ``NucleusMoEImagePipeline.__init__`` (line 175) — default_max_sequence_length.
# The ``__call__`` docstring claims 512; the actual runtime default (this
# constant) is 1024. See module docstring §2.
_DEFAULT_MAX_SEQUENCE_LENGTH = 1024

# ``NucleusMoEImagePipeline.__init__`` (line 176) — default_return_index.
_DEFAULT_RETURN_INDEX = -8

# The MoE router gate's exact module BASENAME (``NucleusMoELayer.gate``) —
# used only by the pinning test to assert no LoRA target pattern ends with
# this literal (see module docstring §7).
ROUTER_GATE_MODULE_NAME = "gate"

# Fingerprint of the system-prompt string, hashed so any future edit to the
# prompt text changes the fingerprint automatically. Exposed publicly via
# :func:`te_template_fingerprint` so the trainer's disk-cache key template
# identity never needs to import this module-private constant directly
# (boogu_image/qwen_image/lumina2 precedent — a hardcoded prompt template
# baked into ``encode_text`` means a future prompt edit must invalidate
# stale on-disk embeddings, not silently reuse them).
_TE_TEMPLATE_FINGERPRINT = hashlib.sha256(
    NUCLEUS_SYSTEM_PROMPT.encode("utf-8"),
).hexdigest()[:16]


def te_template_fingerprint() -> str:
    """Public fingerprint of this driver's system-prompt template.

    Used by ``NucleusImageTrainer`` to version its disk-cache key template
    identity — any future edit to ``NUCLEUS_SYSTEM_PROMPT`` changes the
    fingerprint, and therefore every disk-cache filename, so a stale
    pre-edit embedding can never be silently reused.
    """
    return _TE_TEMPLATE_FINGERPRINT


class NucleusImageDriver(IModelDriver):
    """Nucleus-Image family driver.

    Handles:
    - Qwen3-VL chat-template prompt encoding via the full ``Qwen3VLProcessor``
      (``apply_chat_template`` + tokenization), ``hidden_states[-8]`` tap,
      ragged (``padding="longest"``) sequence lengths
    - Non-reversed [0,1] timestep into the transformer, NEGATED raw output
      (module docstring §4) — the family's #1 silent-LoRA-killer risk
    - Packed (2x2-patchified) flow-matching forward pass
    - Controller-pinned, MoE-router-excluding LoRA target scope
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None  # Qwen3VLProcessor (chat template + BPE)
        self._components: dict[str, Any] = {}

        # Architecture params
        arch = getattr(definition, "architecture_params", {}) or {}
        self.max_sequence_length = int(
            arch.get("te.max_sequence_length", _DEFAULT_MAX_SEQUENCE_LENGTH),
        )
        self.return_index = int(
            arch.get("te.hidden_state_tap_index", _DEFAULT_RETURN_INDEX),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Nucleus-Image components into driver state."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return the single Qwen3-VL text encoder."""
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """Nucleus-Image LoRA targets — attention + shared/dense FFN ONLY.

        See module docstring §7 for the full rationale (controller-pinned,
        deliberately narrower than ai-toolkit's superset, structurally
        excludes the MoE router gate).
        """
        definition_targets = getattr(
            self.definition,
            "lora_targetable_modules",
            None,
        )
        if definition_targets and len(definition_targets) > 0:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        self.logger.info("lora_targets_pattern_defaults")
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "shared_expert.net.0.proj",
            "shared_expert.net.2",
        ]

    def init_scheduler(self) -> Any:
        """Nucleus-Image uses flow matching — no external scheduler for training."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Nucleus-Image loads in bf16 (checkpoint shards are bf16 throughout —
        ``text_encoder/config.json``: ``"dtype": "bfloat16"``; transformer
        shard total bytes / param count is consistent with bf16)."""
        return torch.bfloat16

    def get_precision_spec(
        self, mixed_precision: str, *, is_adaptive_optimizer: bool = False,
    ) -> Any:
        """AMP is force-disabled for this family (GPU UAT crash, 2026-07-14).

        Under ``torch.autocast(bf16)`` LayerNorm executes in fp32, so each
        block's modulated hidden states reach the frozen (non-LoRA) MoE
        experts as fp32 — and ``torch._grouped_mm`` is NOT on autocast's
        cast-policy list, so it receives the raw fp32 activations against
        bf16 expert weights and raises "expected mat1 and mat2 to have the
        same dtype". The transformer manages its own precision islands
        (router scores in fp32 with explicit casts back), and the sampler
        already runs the native no-autocast bf16 regime on real weights, so
        the trainer matches it: bf16 inputs, no autocast, no GradScaler
        (ideogram4 precedent).
        """
        from app.engine.core.layer_manifest import PrecisionSpec  # noqa: PLC0415

        return PrecisionSpec(
            autocast_dtype=torch.bfloat16,
            use_amp=False,
            grad_scaler_enabled=False,
        )

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported — Qwen3-VL stays frozen."""
        return []

    def get_layer_manifest(self) -> Any:
        """Nucleus-Image layer manifest — single ``transformer_blocks`` group
        (32 blocks: first 3 dense, last 29 MoE per ``dense_moe_strategy``)."""
        from app.engine.core.layer_manifest import (  # noqa: PLC0415
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            group = getattr(model, "transformer_blocks", None)
            if group is not None:
                for i, block in enumerate(group):
                    blocks.append(
                        BlockInfo(
                            name=f"transformer_blocks.{i}",
                            block_type="moe" if getattr(block, "moe_enabled", False) else "dense",
                            param_count=sum(p.numel() for p in block.parameters()),
                            depth_index=i,
                        ),
                    )

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions replicating ``NucleusMoEImagePipeline._format_
        prompt``/``encode_prompt`` (see module docstring §1-3).

        Used identically for training captions AND the CFG negative/uncond
        prompt — the real pipeline applies the SAME chat-template wrapping
        to both (no positive/negative asymmetry, unlike ``lumina2``).

        Args:
            captions: Raw caption strings.
            dtype: Target dtype for the returned embeddings.

        Returns:
            ``TextEncoderOutput`` with ``hidden_states[return_index]``
            embeddings ``[B, L, 4096]`` (``L`` = this call's own
            ``padding="longest"`` length, ragged across different calls —
            see module docstring §3) and the raw processor attention mask
            ``[B, L]``.
        """
        messages = [
            [
                {"role": "system", "content": NUCLEUS_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": c}]},
            ]
            for c in captions
        ]
        formatted = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
            )
            for m in messages
        ]

        inputs = self.tokenizer(
            text=formatted,
            padding="longest",
            pad_to_multiple_of=8,
            max_length=self.max_sequence_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
        # hidden_states[return_index] (default -8) — NOT the last layer
        # (pipeline_nucleusmoe_image.py lines 233-235).
        prompt_embeds = outputs.hidden_states[self.return_index]

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """NucleusMoEImageTransformer2DModel forward — packed latents,
        NON-reversed [0,1] timestep, NEGATED raw output (module docstring
        §4-5, the family's #1 silent-LoRA-killer risk).

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]``.
            timesteps: Flow-matching timesteps in ``[0, 1000]`` scale (the
                SAME raw scale as every other family in this codebase — the
                ``/1000.0`` normalization below is entirely internal to this
                hook and matches the pipeline's own ``timestep /
                self.scheduler.config.num_train_timesteps``).
            text_embeddings: ``(embeddings, attention_mask)`` tuple (the
                trainer's ``encode_text`` contract) or a plain embeddings
                tensor (mask-less fallback).
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 16, H, W]`` in the standard
            ``noise - latents`` convention (already negated — see module
            docstring §4).
        """
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs, enc_mask = text_embeddings, None

        B, C, H, W = noisy_input.shape
        model = self.get_primary_model()
        patch_size = getattr(model.config, "patch_size", 2)

        # Patchify: [B, C, H, W] -> [B, (H/p)*(W/p), C*p*p]
        pH = H // patch_size
        pW = W // patch_size
        x = noisy_input.reshape(B, C, pH, patch_size, pW, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, pH * pW, C * patch_size * patch_size)

        # NON-reversed normalization: transformer's own timestep convention
        # matches the raw [0,1000]->[0,1] scale directly (module docstring
        # §4 — do NOT reverse, unlike lumina2).
        model_timestep = timesteps / 1000.0

        img_shapes = [(1, pH, pW)] * B

        output = model(
            hidden_states=x,
            img_shapes=img_shapes,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=enc_mask,
            timestep=model_timestep,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # Unpatchify: [B, (H/p)*(W/p), out_C*p*p] -> [B, out_C, H, W]
        out_channels = getattr(model.config, "out_channels", None) or C
        pred = pred.reshape(B, pH, pW, out_channels, patch_size, patch_size)
        pred = pred.permute(0, 3, 1, 4, 2, 5)
        pred = pred.reshape(B, out_channels, H, W)

        # NEGATION: the raw model output is expressed in the OPPOSITE sign
        # convention from this project's default `noise - latents` target
        # (pipeline_nucleusmoe_image.py line 599: `noise_pred = -noise_pred`,
        # applied right before `scheduler.step`). Module docstring §4.
        return -pred

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return the Nucleus-Image diffusers-canonical LoRA saver."""
        from app.engine.models.families.nucleus_image.saver import (  # noqa: PLC0415
            NucleusImageSaver,
        )

        return NucleusImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Nucleus-Image block topology: single transformer_blocks group."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                topology.append(
                    {
                        "name": "transformer_blocks",
                        "attr_path": "transformer_blocks",
                        "count": len(blocks),
                        "approx_vram_mb": 900,
                    },
                )
        return topology
