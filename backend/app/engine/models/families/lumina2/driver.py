"""Lumina2Driver — family-specific training behavior for Lumina-Image-2.0.

Implements ``IModelDriver`` for Alpha-VLLM's Lumina-Image-2.0: a 2.6B DiT
(``Lumina2Transformer2DModel``, diffusers 0.39) with a single frozen
Gemma-2-2B text encoder (``transformers.Gemma2Model``) and the FLUX.1-dev
16-channel VAE. Apache-2.0. Also the architectural base of the NetaYume /
Neta-Lumina anime-finetune lineage — a future definition-only add (same
transformer/loader/driver code, different checkpoint weights + maybe a
richer default sample prompt) should NOT need a new family.

All facts below verified against ``venv/Lib/site-packages/diffusers/
pipelines/lumina2/pipeline_lumina2.py`` and ``.../models/transformers/
transformer_lumina2.py`` (diffusers 0.39.0) plus the live HF repo
``Alpha-VLLM/Lumina-Image-2.0`` (config.json files, 2026-07-13, no download
needed).

Lumina2 specifics mirrored here:

1. **System-prompt contract** (``pipeline_lumina2.py`` lines 185, 285-288):
   ``Lumina2Pipeline.__init__`` sets a fixed ``self.system_prompt`` and
   ``encode_prompt`` prepends it to every POSITIVE prompt with a literal
   ``" <Prompt Start> "`` separator::

       prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]

   Verbatim system prompt (``pipeline_lumina2.py`` line 185)::

       "You are an assistant designed to generate superior images with the
        superior degree of image-text alignment based on textual prompts or
        user prompts."

   Critically, the NEGATIVE prompt used for CFG is encoded RAW — the
   prefixing above only ever touches the ``prompt`` list, never
   ``negative_prompt`` (``pipeline_lumina2.py`` lines 304-328: the
   ``negative_prompt`` branch calls ``_get_gemma_prompt_embeds`` directly on
   the unprefixed string). This driver replicates the asymmetry: the shared
   ``encode_text`` applies the system prompt by default (used for every real
   training caption and the sampler's positive prompt); the sampler calls it
   with ``apply_system_prompt=False`` for the negative/uncond prompt (see
   ``sampler.py``). Like the ovis_image "rich vs sparse prompt" lesson,
   TRAINING captions should be reasonably descriptive — Lumina2 was trained
   with the system prompt anchoring "superior image-text alignment", so a
   sparse caption under-utilizes that anchor the same way a sparse Ovis
   caption produced detail-free samples.

2. **Prompt embedding tap** (``pipeline_lumina2.py`` lines 219-222):
   ``_get_gemma_prompt_embeds`` forwards through Gemma-2 with
   ``output_hidden_states=True`` and takes ``hidden_states[-2]`` — the
   SECOND-TO-LAST hidden layer, NOT ``last_hidden_state`` and NOT a
   pooled/projected output. Padding is ``padding="max_length"``,
   ``max_length=256`` (default), ``truncation=True``; ``tokenizer.
   padding_side`` is forced to ``"right"`` in ``Lumina2Pipeline.__init__``
   (line 190). The RAW tokenizer attention mask is what both the T5-style
   encoder forward AND the transformer's ``encoder_attention_mask`` consume
   — no zeroing/foot-gun modification (unlike ovis_image/chroma).

3. **Timestep REVERSAL — the #1 silent-LoRA-killer risk for this family**
   (``pipeline_lumina2.py`` lines 720-724, 758):

       # reverse the timestep since Lumina uses t=0 as the noise and
       # t=1 as the image
       current_timestep = 1 - t / self.scheduler.config.num_train_timesteps

   The scheduler's own raw ``t`` (``FlowMatchEulerDiscreteScheduler``
   timesteps, ``[0, 1000]``) follows the same "t=1000 is pure noise, t=0 is
   clean" convention as every OTHER flow-match family in this codebase (and
   this project's shared ``TimestepSampler``/``NoiseInterpolation``
   defaults) — but the Lumina2 TRANSFORMER's OWN ``timestep`` input
   conditions on the OPPOSITE convention (0=noise, 1=image). The pipeline
   flips it right before every transformer call. Then, right before handing
   the prediction to the scheduler (line 758)::

       noise_pred = -noise_pred

   the raw model output is NEGATED. Reasoning: the network's natural
   "velocity" is expressed w.r.t. its OWN (reversed) time variable
   ``t' = 1 - sigma``, i.e. ``d(sample)/dt' = -(noise - latents) = latents -
   noise``; negating converts it back to the ``noise - latents`` convention
   the scheduler (and this project's default ``compute_target``) expects.

   This driver replicates BOTH halves in ``forward_pass``: feed the
   transformer ``1.0 - (timesteps / 1000.0)``, and return ``-pred`` (NOT the
   raw model output). With that, no override of ``add_noise``,
   ``sample_timesteps``, or ``compute_target`` is needed — the base
   ``IModelDriver`` flow-match defaults (``noise - latents`` velocity target,
   ``[0, 1000]``-scale raw timesteps into ``forward_pass``) already match
   once the reversal/negation happens inside this one hook. Pinned by
   ``backend/tests/engine/families/lumina2/test_lumina2_timestep_reversal.py``.

4. **True CFG, pipeline defaults pinned** (``pipeline_lumina2.py`` lines
   531-550, 618-621, 737-752): ``num_inference_steps=30``,
   ``guidance_scale=4.0``, ``cfg_trunc_ratio=1.0`` (never truncates —
   ``(i+1)/num_inference_steps > 1.0`` is never true, so CFG runs every
   step), ``cfg_normalization=True`` (rescales the combined velocity's
   ``dim=-1`` norm back to the conditional prediction's norm). This driver
   / the sampler PIN ``cfg_trunc_ratio``/``cfg_normalization`` at their
   pipeline defaults rather than surfacing them as user knobs (not worth the
   schema surface for a truncation feature that is a no-op at its own
   default).

5. **Scheduler**: ``scheduler/scheduler_config.json`` on the live repo is
   STATIC-shift (``use_dynamic_shifting: false``, ``shift: 6.0``) — NOT
   resolution-dependent dynamic shifting like flux1/ovis_image, despite the
   pipeline unconditionally computing + passing ``mu`` (the scheduler
   silently ignores it when not dynamic, chroma1-hd precedent). ComfyUI's
   own ``Lumina2`` supported-model entry independently confirms
   ``sampling_settings = {"shift": 6.0}`` — cross-checked, not just the HF
   config.
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

# Verbatim system prompt — ``Lumina2Pipeline.__init__`` (pipeline_lumina2.py
# line 185). Prepended to POSITIVE prompts only (see module docstring §1).
LUMINA2_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the "
    "superior degree of image-text alignment based on textual prompts or "
    "user prompts."
)

# Literal separator the pipeline inserts between the system prompt and the
# user prompt (``pipeline_lumina2.py`` line 288).
_PROMPT_START_SEP = " <Prompt Start> "

# ``Lumina2Pipeline.encode_prompt`` default / hard maximum is 512, but the
# ``__call__`` signature default (and what actually gets used unless a
# caller overrides it) is 256 (pipeline_lumina2.py line 550).
_DEFAULT_MAX_SEQUENCE_LENGTH = 256

# Fingerprint of the system-prompt string, hashed so any future edit to the
# prompt text changes the fingerprint automatically. Exposed publicly via
# :func:`te_template_fingerprint` so the trainer's disk-cache key template
# identity never needs to import this module-private constant directly
# (boogu_image precedent).
_TE_TEMPLATE_FINGERPRINT = hashlib.sha256(
    LUMINA2_SYSTEM_PROMPT.encode("utf-8"),
).hexdigest()[:16]


def te_template_fingerprint() -> str:
    """Public fingerprint of this driver's system-prompt template.

    Used by ``Lumina2Trainer`` to version its disk-cache key template
    identity — any future edit to ``LUMINA2_SYSTEM_PROMPT`` changes the
    fingerprint, and therefore every disk-cache filename, so a stale
    pre-edit embedding can never be silently reused.
    """
    return _TE_TEMPLATE_FINGERPRINT


class Lumina2Driver(IModelDriver):
    """Lumina-Image-2.0 family driver.

    Handles:
    - Gemma-2 prompt encoding replicating ``Lumina2Pipeline._get_gemma_
      prompt_embeds``/``encode_prompt`` (system-prompt prefix asymmetry,
      ``hidden_states[-2]`` tap)
    - The Lumina2 REVERSED flow-match timestep convention (module
      docstring §3) — the single most important correctness contract for
      this family
    - Un-packed (no 2×2 pack/unpack step — the transformer patchifies
      internally) flow-matching forward pass
    - Single-block-family LoRA targets across ``layers``/
      ``context_refiner``/``noise_refiner``
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        arch = getattr(definition, "architecture_params", {}) or {}
        self.max_sequence_length = int(
            arch.get("te.max_sequence_length", _DEFAULT_MAX_SEQUENCE_LENGTH),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Lumina2 components; force right-padding on the
        tokenizer (``Lumina2Pipeline.__init__`` line 190 — Gemma tokenizers
        default to left-padding, which would silently break the
        ``max_length`` alignment this driver relies on)."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "right"

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return the single Gemma-2 text encoder."""
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def release_text_encoders(self) -> None:
        """Null the attr get_text_encoders() reads."""
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """Lumina2 LoRA targets — attention + feed-forward Linear modules.

        Applies uniformly across ``layers.*`` (26, modulated), ``context_
        refiner.*`` (2, no modulation) and ``noise_refiner.*`` (2, modulated)
        — all three ModuleLists share the same ``Lumina2TransformerBlock``
        submodule names (verified: ``transformer_lumina2.py`` lines 393-444,
        AND live-instantiated + introspected, 2026-07-13):
        ``attn.to_q/to_k/to_v/to_out.0`` and ``feed_forward.linear_1/2/3``.
        ``attn.norm_q``/``attn.norm_k`` are ``RMSNorm`` (no weight matrix
        PEFT can decompose) and are intentionally excluded, as is the
        AdaLN modulation ``norm1.linear`` (consistent with this codebase's
        ovis_image/chroma precedent of NOT LoRA-targeting timestep/AdaLN
        modulation linears).
        """
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "feed_forward.linear_1",
            "feed_forward.linear_2",
            "feed_forward.linear_3",
        ]

    def init_scheduler(self) -> Any:
        """Lumina2 uses flow matching — no external scheduler for training."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Lumina2 loads in bf16 (``Lumina2Pipeline`` docstring example:
        ``torch_dtype=torch.bfloat16`` — the checkpoint's raw shards are
        fp32, but bf16 is the intended inference/training dtype)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported — Gemma-2 stays frozen."""
        return []

    def get_layer_manifest(self) -> Any:
        """Lumina2 layer manifest across all three block groups."""
        from app.engine.core.layer_manifest import (  # noqa: PLC0415
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            offset = 0
            for attr, block_type in (
                ("layers", "joint"),
                ("context_refiner", "context_refiner"),
                ("noise_refiner", "noise_refiner"),
            ):
                group = getattr(model, attr, None)
                if group is None:
                    continue
                for i, block in enumerate(group):
                    blocks.append(
                        BlockInfo(
                            name=f"{attr}.{i}",
                            block_type=block_type,
                            param_count=sum(p.numel() for p in block.parameters()),
                            depth_index=offset + i,
                        ),
                    )
                offset += len(group)

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
        *,
        apply_system_prompt: bool = True,
    ) -> TextEncoderOutput:
        """Encode captions replicating ``Lumina2Pipeline._get_gemma_prompt_
        embeds``/``encode_prompt`` (see module docstring §1-2).

        Args:
            captions: Raw caption strings.
            dtype: Target dtype for the returned embeddings.
            apply_system_prompt: When ``True`` (default — used for every
                real training caption and the sampler's POSITIVE prompt),
                prefixes each caption with ``LUMINA2_SYSTEM_PROMPT +
                " <Prompt Start> "``, exactly matching the pipeline's
                ``prompt`` branch. When ``False`` (the sampler's negative/
                uncond prompt), captions are encoded RAW — matching the
                pipeline's ``negative_prompt`` branch, which never sees the
                system-prompt prefix.

        Returns:
            ``TextEncoderOutput`` with ``hidden_states[-2]`` embeddings
            ``[B, max_sequence_length, 2304]`` and the RAW tokenizer
            attention mask ``[B, max_sequence_length]`` (no zeroing/
            foot-gun modification, unlike ovis_image/chroma).
        """
        if apply_system_prompt:
            texts = [
                f"{LUMINA2_SYSTEM_PROMPT}{_PROMPT_START_SEP}{c}" for c in captions
            ]
        else:
            texts = list(captions)

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Second-to-last hidden layer — NOT last_hidden_state (pipeline_
        # lumina2.py line 222).
        prompt_embeds = outputs.hidden_states[-2]

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
        """Lumina2Transformer2DModel forward — REVERSED timestep + negated
        output (module docstring §3, the #1 silent-LoRA-killer risk).

        Unlike flux1/ovis_image/chroma, Lumina2 does NOT pack latents 2×2
        before the transformer — ``Lumina2Transformer2DModel.forward``
        patchifies internally via its own ``rope_embedder`` (``transformer_
        lumina2.py`` lines 492-506). ``noisy_input`` is passed straight
        through as ``[B, C, H, W]``.

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]``.
            timesteps: Flow-matching timesteps in ``[0, 1000]`` scale (the
                SAME raw scale as every other family in this codebase —
                the reversal below is entirely internal to this hook).
            text_embeddings: ``(embeddings, attention_mask)`` tuple (the
                trainer's ``encode_text`` contract) or a plain embeddings
                tensor (mask-less fallback — an all-ones mask is
                synthesized so the transformer's ``rope_embedder`` doesn't
                crash on ``attention_mask.shape``).
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 16, H, W]`` in the standard
            ``noise - latents`` convention (already negated — see below).
        """
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs, enc_mask = text_embeddings, None

        if enc_mask is None:
            enc_mask = torch.ones(
                enc_hs.shape[0], enc_hs.shape[1],
                device=enc_hs.device, dtype=torch.long,
            )

        # REVERSAL: the transformer's own timestep convention is 0=noise,
        # 1=image — opposite the raw [0,1000] scale (pipeline_lumina2.py
        # lines 723-724: "reverse the timestep since Lumina uses t=0 as the
        # noise and t=1 as the image").
        model_timestep = 1.0 - (timesteps / 1000.0)

        model = self.get_primary_model()
        output = model(
            hidden_states=noisy_input,
            timestep=model_timestep,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_mask,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # NEGATION: the raw model output is expressed w.r.t. the reversed
        # timestep (points latents->noise); negate to land in the standard
        # noise-latents convention this project's default compute_target()
        # uses (pipeline_lumina2.py line 758: `noise_pred = -noise_pred`).
        return -pred

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return the Lumina2 diffusers-canonical LoRA saver."""
        from app.engine.models.families.lumina2.saver import (  # noqa: PLC0415
            Lumina2Saver,
        )

        return Lumina2Saver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Lumina2 block topology: layers + context_refiner + noise_refiner."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            for attr, name, vram in (
                ("layers", "layers", 340),
                ("context_refiner", "context_refiner", 340),
                ("noise_refiner", "noise_refiner", 340),
            ):
                group = getattr(model, attr, None)
                if group is not None:
                    topology.append(
                        {
                            "name": name,
                            "attr_path": attr,
                            "count": len(group),
                            "approx_vram_mb": vram,
                        },
                    )
        return topology
