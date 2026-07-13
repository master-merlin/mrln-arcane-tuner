"""OmniGen2Driver — family-specific training behavior for OmniGen2.

Implements ``IModelDriver`` for VectorSpaceLab's OmniGen2 (arXiv:2506.18871,
Apache-2.0): a ~4B DiT (vendored ``OmniGen2Transformer2DModel`` — diffusers
0.39 ships NO OmniGen2 classes, see ``vendor/__init__.py``) conditioned by
Qwen2.5-VL-3B-Instruct used purely as a TEXT encoder, with the FLUX.1-dev
16-channel VAE. Edit-first: the single shipped definition is
``control_inputs: 1`` (instruction edit on paired Target/Control datasets);
pure T2I works whenever a batch/prompt carries no control image.

All upstream citations below refer to the pinned revision in
``vendor/REVISION`` (``omnigen2/pipelines/omnigen2/pipeline_omnigen2.py`` =
"pipeline", ``train.py``/``omnigen2/transport/*`` = "train"), plus the live
HF checkpoint ``OmniGen2/OmniGen2`` config files (fetched 2026-07-13).

## 1. Conditioning channels (recon finding — differs from boogu_image!)

Reference/control images enter through EXACTLY ONE channel: clean VAE
latents fed to the transformer as ``ref_image_hidden_states``.

- Inference: ``prepare_image`` -> ``encode_vae`` (pipeline ~L217-272:
  ``vae.encode(...).latent_dist.sample()`` then ``- shift_factor`` then
  ``* scaling_factor``) -> ``predict(..., ref_image_hidden_states=
  ref_latents)`` (~L667/686/763-764).
- Training: train.py L523-538 VAE-encodes ``input_images`` identically and
  passes ``ref_image_hidden_states=input_latents`` (L547).
- The MLLM (``Qwen2_5_VLForConditionalGeneration``) is invoked in exactly
  one place, ``_get_qwen2_prompt_embeds`` (~L274-339), with ``text_input_ids
  + attention_mask`` ONLY — no ``pixel_values``, no vision tower. Training
  (train.py L515-519) likewise calls the text encoder on ids+mask only.

Consequence: unlike boogu_image (whose VL encoder DOES attend the control
pixels, hence composite (caption, control) TE-cache keys), OmniGen2's text
embeddings are CONTROL-INDEPENDENT — the plain per-caption TE cache is
correct and composite keys are deliberately NOT used here.

## 2. Text encoding contract

``encode_text`` mirrors ``_apply_chat_template`` + ``_get_qwen2_prompt_
embeds`` (pipeline ~L274-350):

- Chat template: ``[{"role": "system", "content": OMNIGEN2_SYSTEM_PROMPT},
  {"role": "user", "content": caption}]`` serialized via
  ``tokenizer.apply_chat_template(..., tokenize=False,
  add_generation_prompt=False)`` (~L341-350).
- Tokenize ``padding="longest"``, ``truncation=True`` with the definition's
  ``te.max_sequence_length`` (~L305-311; ``encode_prompt``'s own default is
  256).
- Forward the mllm with ``output_hidden_states=True`` and tap
  ``hidden_states[-1]`` (~L324-328) — the last decoder layer (equivalent to
  train.py's ``TextEncoder(...).last_hidden_state`` tap, L515-519).
- The CFG NEGATIVE prompt goes through the SAME chat template (pipeline
  L413-418 applies ``_apply_chat_template`` to ``negative_prompt`` too,
  default ``""``) — NO lumina2-style raw/prefixed asymmetry, so a single
  encode path (and a single cache) serves both.

## 3. Time convention — INVERTED (the boogu_image convention, verbatim)

Derived from the vendored scheduler + transport (cited in
``vendor/schedulers/scheduling_flow_match_euler_discrete.py``'s header and
train transport ``path.py::ICPlan``): ``t=0`` pure noise, ``t=1`` clean
image, ``x_t = t*x0 + (1-t)*noise`` (``compute_mu_t``: ``alpha_t=t``,
``sigma_t=1-t``), velocity target ``d/dt x_t = x0 - noise``
(``compute_ut``: ``d_alpha=1, d_sigma=-1``), forward-Euler ``prev = sample
+ (t_next - t) * v``. The transformer consumes the RAW ``[0, 1)`` ``t``
(pipeline ``predict`` L758: ``t.expand(...)`` with scheduler-native
``[0, 1)`` timesteps) — its ``timestep_scale: 1000.0`` config
(checkpoint ``transformer/config.json``) multiplies INSIDE
``Lumina2CombinedTimestepCaptionEmbedding``'s ``Timesteps`` proj. No
``/1000`` or ``*1000`` anywhere in family code (flow-match timestep-scale
gotcha).

## 4. Training-time timestep shift (train.py L322-334 + transport L104-172)

Upstream's LoRA config (``options/ft_lora.yml``): ``snr_type: lognorm``,
``do_shift: true``, ``dynamic_time_shift: true`` (version default "v1").
That pipeline is algebraically: draw ``u ~ N(0,1)``, ``t_raw = sigmoid(u)``
(lognorm), then ``time_shift(mu, 1.0, t)`` with ``mu = lin(256->0.5,
4096->1.15)(patch_tokens)`` (transport L139-172, ``get_lin_function``
defaults L180-185). Because ``time_shift`` wraps its sigmoid shift in
``t = 1 - t`` on both sides (L163-172 — "we adopt the reverse"), the whole
draw collapses to ``sigma = sigmoid(N(0,1) + mu)`` in standard sigma-space
— EXACTLY this codebase's ``flux_shift`` TimestepSampler mode with
``base=0.5``/``max=1.15`` over patchified (p=2) sequence length.
``sample_timesteps`` therefore defaults to ``flux_shift`` (seeding
``flux_shift_base=0.5``, ``flux_shift_max=1.15``,
``flux_shift_patchify_factor=2`` unless the user overrides) and converts to
the native reversed clock as ``t = 1 - sigma`` — the identity that holds
for EVERY house mode, so a user-picked mode keeps its documented semantics.

## 5. Guidance mapping (documented for the sampler)

Pipeline ``__call__`` defaults: ``text_guidance_scale=4.0``,
``image_guidance_scale=1.0``, ``cfg_range=(0, 1)`` (L485-487); with no
input images ``image_guidance_scale`` is FORCED to 1 (L562-563). Our single
``guidance_scale`` knob maps to ``text_guidance_scale``;
``image_guidance_scale`` comes from the definition's
``defaults.image_guidance_scale`` (edit default 2.0 — upstream
``example_edit.sh``: ``--text_guidance_scale 5.0 --image_guidance_scale
2.0``). See ``sampler.py``/``sampler_edit.py`` for the 2-pass/3-pass
combine (pipeline L672-723).
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver, IModelSaver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# RoPE base frequency — hardcoded in the vendored transformer's own
# rope_embedder construction (transformer_omnigen2.py: ``theta=10000``) and
# in the pipeline's ``get_freqs_cis(..., theta=10000)`` call (~L578-582).
_ROPE_THETA = 10000

# Verbatim from the pipeline's ``_apply_chat_template`` (~L341-350). Applied
# to EVERY encode — positive captions, dropout ("") captions AND the CFG
# negative prompt (pipeline L392 + L413-418 template both branches
# identically). No boogu-style per-caption prompt selection.
OMNIGEN2_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality images based "
    "on user instructions."
)

# Fallback when a definition ships no ``te.max_sequence_length`` (the
# shipped definition does — 256, the ``encode_prompt`` default, pipeline
# ~L363).
_DEFAULT_MAX_SEQUENCE_LENGTH = 256

# Upstream training-shift constants (transport ``get_lin_function`` defaults
# L180-185: x1=256, y1=0.5, x2=4096, y2=1.15). Seeded as flux_shift config
# defaults in sample_timesteps (module docstring §4).
_SHIFT_BASE = 0.5
_SHIFT_MAX = 1.15

# Fingerprint of the system-prompt string, hashed so any future edit to the
# prompt text changes the fingerprint automatically (lumina2/boogu_image
# precedent — consumed by the trainer's disk-cache key template identity).
_TE_TEMPLATE_FINGERPRINT = hashlib.sha256(
    OMNIGEN2_SYSTEM_PROMPT.encode("utf-8"),
).hexdigest()[:16]


def te_template_fingerprint() -> str:
    """Public fingerprint of this driver's chat-template system prompt.

    Used by ``OmniGen2Trainer`` to version its disk-cache key template
    identity — any edit to ``OMNIGEN2_SYSTEM_PROMPT`` changes every
    disk-cache filename, so a stale pre-edit embedding can never be
    silently reused.
    """
    return _TE_TEMPLATE_FINGERPRINT


class OmniGen2Driver(IModelDriver):
    """OmniGen2 family driver (edit-first; T2I via the no-control fallback)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.processor: Any = None
        self.scheduler: Any = None
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", None) or {}
        self.max_sequence_length = int(
            arch.get("te.max_sequence_length", _DEFAULT_MAX_SEQUENCE_LENGTH),
        )

        # Lazily-built static RoPE lookup table (function of the model
        # config only, not batch shapes — boogu_image precedent).
        self._freqs_cis_cache: tuple[Any, list[torch.Tensor]] | None = None

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded OmniGen2 components into driver state.

        ``scheduler`` (the LOADER-provided vendored instance carrying the
        checkpoint's ``dynamic_time_shift`` config) is assigned here so it
        is available before ``init_scheduler()`` is explicitly called.
        """
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.processor = components.get("processor")
        self.scheduler = components.get("scheduler")
        self._freqs_cis_cache = None

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """OmniGen2 LoRA targets — attention + feed-forward Linears.

        All four block groups (``layers`` ×32, ``noise_refiner`` ×2,
        ``ref_image_refiner`` ×2, ``context_refiner`` ×2) are the SAME
        ``OmniGen2TransformerBlock`` class with identical submodule names
        (vendored transformer_omnigen2.py), so short suffix patterns cover
        every group — the lumina2 convention. Upstream's own LoRA recipe
        (train.py L262: ``["to_k", "to_q", "to_v", "to_out.0"]``) targets
        attention only; the feed-forward Linears are added per this
        codebase's house convention (krea2/ovis/lumina2/boogu curated
        surface). ``attn.norm_q``/``norm_k`` are RMSNorm (nothing for PEFT
        to decompose) and the AdaLN ``norm1.linear`` is excluded per house
        convention.
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
        """Return the LOADER-provided vendored scheduler.

        Consumes ``self._components["scheduler"]`` (the checkpoint's own
        ``{"dynamic_time_shift": true, "num_train_timesteps": 1000}``
        config loaded via the vendored class's ``ConfigMixin.from_
        pretrained``) — never a fresh/stock instance (the boogu_image
        clobber lesson: a stock diffusers scheduler of the same NAME has
        incompatible semantics, see the vendored module's header).
        """
        scheduler = self._components.get("scheduler")
        if scheduler is None:
            raise RuntimeError(
                "omnigen2: init_scheduler() called with no 'scheduler' "
                "component assigned — assign_components() must run first "
                "(the scheduler comes from the loader, not a fresh default)."
            )
        self.scheduler = scheduler
        return scheduler

    def resolve_loading_dtype(self) -> torch.dtype:
        """OmniGen2 loads in bf16 (upstream README/inference default
        ``torch_dtype=torch.bfloat16``; train ``mixed_precision: 'bf16'``)."""
        return torch.bfloat16

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Qwen2.5-VL TEXT-ONLY encoding — chat template + last-layer tap.

        Mirrors ``_apply_chat_template`` + ``_get_qwen2_prompt_embeds``
        (module docstring §2). The mllm NEVER sees image pixels here —
        that is upstream's own behavior (recon §1), not a degraded mode.

        Args:
            captions: Batch of caption strings ("" is valid — dropout and
                the CFG negative both encode the empty string under the
                same template, matching the pipeline's negative branch).
            dtype: Target dtype for the returned embeddings.

        Returns:
            ``TextEncoderOutput`` with:
            - ``embeddings``: ``[B, L<=max_seq_len, 2048]`` (``te.hidden_size``)
            - ``attention_mask``: ``[B, L]`` (tokenizer's own mask)
        """
        if self.processor is None or self.text_encoder is None:
            raise RuntimeError(
                "omnigen2 encode_text: 'processor'/'text_encoder' not "
                "assigned — assign_components() must run first."
            )

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        texts = []
        for caption in captions:
            messages = [
                {"role": "system", "content": OMNIGEN2_SYSTEM_PROMPT},
                {"role": "user", "content": caption},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            )

        tokens = tokenizer(
            texts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Last decoder layer (pipeline ~L324-328) — Qwen2.5-VL's
        # hidden_states[-1] is post-final-norm, i.e. the same content as
        # train.py's TextEncoder(...).last_hidden_state tap.
        prompt_embeds = outputs.hidden_states[-1]

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder (Qwen2.5-VL mllm) LoRA not supported — stays frozen
        (upstream train.py freezes it too: TE forward under no_grad)."""
        return []

    def get_layer_manifest(self) -> Any:
        """OmniGen2 layer manifest across all four block groups."""
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
                ("noise_refiner", "noise_refiner"),
                ("ref_image_refiner", "ref_image_refiner"),
                ("context_refiner", "context_refiner"),
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

    # --- Phase 5: Training Loop Hooks ---

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Sample NATIVE reversed-clock timesteps in ``[0, 1)``.

        Draws a standard sigma from the shared ``TimestepSampler`` and
        returns ``1 - sigma`` (module docstring §4 — the identity that maps
        every house mode onto OmniGen2's reversed clock). Family default
        mode is ``flux_shift`` seeded with upstream's own training-shift
        constants (base 0.5 / max 1.15 over p=2 patch tokens), which is
        ALGEBRAICALLY IDENTICAL to upstream's ``lognorm + do_shift +
        dynamic_time_shift(v1)`` draw. User-set ``timestep_sampling`` /
        ``flux_shift_*`` config keys win over the seeded defaults.
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler  # noqa: PLC0415

        mode = config.get("timestep_sampling", "flux_shift")
        cfg = dict(config)
        cfg.setdefault("flux_shift_base", _SHIFT_BASE)
        cfg.setdefault("flux_shift_max", _SHIFT_MAX)
        cfg.setdefault("flux_shift_patchify_factor", 2)
        sigma = TimestepSampler.sample(
            mode, batch_size, device, cfg, latents=latents, progress=progress,
        )
        return 1.0 - sigma

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """OmniGen2's INVERTED flow-match lerp: ``x_t = (1-t)*noise + t*x0``.

        ``t=0`` -> pure noise, ``t=1`` -> clean latents (transport
        ``ICPlan.compute_mu_t``: ``alpha_t*x1 + sigma_t*x0`` with
        ``alpha_t=t``, ``sigma_t=1-t``; module docstring §3).
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
        """Velocity target ``x0 - noise`` — the correct sign for the
        inverted convention (transport ``ICPlan.compute_ut``:
        ``d_alpha_t*x1 + d_sigma_t*x0`` = ``x1 - x0`` with data=x1,
        noise=x0; opposite the house default ``noise - latents``). Paired
        exactly with ``add_noise`` — pinned by the perfect-velocity
        round-trip test against the vendored scheduler.
        """
        return latents - noise

    def _build_freqs_cis(self, model: nn.Module) -> list[torch.Tensor]:
        """Static RoPE lookup table from the model's own rope config.

        Matches the pipeline's ``OmniGen2RotaryPosEmbed.get_freqs_cis(
        axes_dim_rope, axes_lens, theta=10000)`` call made ONCE before the
        denoise loop (~L578-582) and train.py's identical module-level call
        (L209-213). Cached — pure function of the config.
        """
        from .vendor.models.transformers.repo import (  # noqa: PLC0415
            OmniGen2RotaryPosEmbed,
        )

        axes_dim_rope = tuple(model.config.axes_dim_rope)
        axes_lens = tuple(model.config.axes_lens)
        cache_key = (axes_dim_rope, axes_lens, _ROPE_THETA)

        if self._freqs_cis_cache is not None and self._freqs_cis_cache[0] == cache_key:
            return self._freqs_cis_cache[1]

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
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
        """Vendored ``OmniGen2Transformer2DModel`` forward.

        1. Unpack ``text_embeddings`` -> ``(embeddings, attention_mask)``
           tuple (the trainer's encode contract; the mask is REQUIRED —
           the rope embedder derives per-sample caption lengths from
           ``attention_mask.sum(dim=1)``).
        2. List-tensor adapter: split ``[B, C, H, W]`` into per-sample
           ``[C, H, W]`` tensors (the model's variable-resolution I/O
           contract) and re-stack the output (boogu_image contract-4
           precedent).
        3. RAW ``[0, 1)`` timestep — no scaling (module docstring §3).
        4. Static ``freqs_cis`` lookup table (contract above).
        5. ``ref_image_hidden_states``: ``batch["control_latents"]`` (the
           house paired-dataset layer's clean control latents — per-slot
           ``List[Tensor[B, C, h, w]]``) adapted to the model's per-item
           ``List[List[Tensor[C, h, w]]]`` contract; ``None`` when absent
           (pure T2I — pipeline's own no-input-images path, where
           ``flat_and_pad_to_seq`` degrades to zero-length ref sequences).

        Returns:
            Velocity prediction ``[B, C, H, W]`` (``x0 - noise`` convention).
        """
        if isinstance(text_embeddings, tuple):
            text_hidden_states, text_attention_mask = text_embeddings
        else:
            text_hidden_states = text_embeddings
            text_attention_mask = None

        if text_attention_mask is None:
            raise ValueError(
                "omnigen2 forward_pass: text_attention_mask is required "
                "(the rope embedder derives per-sample caption lengths "
                "from attention_mask.sum(dim=1)) — encode_text must supply "
                "the tokenizer's attention mask, not None."
            )

        model = self.get_primary_model()

        batch_size = noisy_input.shape[0]
        hidden_states_list = [noisy_input[i] for i in range(batch_size)]

        timestep = timesteps.reshape(batch_size).to(dtype=noisy_input.dtype)

        freqs_cis = self._build_freqs_cis(model)

        ref_image_hidden_states = self._build_ref_image_hidden_states(
            batch, batch_size, noisy_input,
        )

        output = model(
            hidden_states=hidden_states_list,
            timestep=timestep,
            text_hidden_states=text_hidden_states,
            freqs_cis=freqs_cis,
            text_attention_mask=text_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
            return_dict=False,
        )

        if isinstance(output, list):
            return torch.stack(output, dim=0)
        return output

    @staticmethod
    def _build_ref_image_hidden_states(
        batch: dict[str, Any], batch_size: int, noisy_input: torch.Tensor,
    ) -> list[list[torch.Tensor]] | None:
        """Adapt ``batch["control_latents"]`` to the model's own contract.

        The house paired-dataset layer (``pipeline_data.py::
        _load_control_latents``) produces per-SLOT ``List[Tensor[B, C, h,
        w]]``; ``OmniGen2Transformer2DModel.forward`` wants per-ITEM
        ``List[List[Tensor[C, h, w]]]`` (one inner list per batch item,
        slots in order — upstream supports up to 5 via
        ``image_index_embedding``). Identical adapter to boogu_image's.

        The house latent cache normalizes ``(sample - shift_factor) *
        scaling_factor`` — the same order as upstream ``encode_vae``
        (pipeline L227-233 / train.py L523-528), so cached control latents
        are byte-compatible with what upstream trains on.

        Returns ``None`` when no control latents are present — the pure-T2I
        fallback (also what a T2I preview's ``batch={}`` hits).
        """
        control_latents = batch.get("control_latents")
        if not control_latents:
            return None
        return [
            [
                slot[i].to(device=noisy_input.device, dtype=noisy_input.dtype)
                for slot in control_latents
            ]
            for i in range(batch_size)
        ]

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> IModelSaver:
        """Return the OmniGen2 diffusers-canonical LoRA saver."""
        from .saver import OmniGen2Saver  # noqa: PLC0415

        return OmniGen2Saver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """OmniGen2 block topology: layers + the three refiner stacks."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            for attr in ("layers", "noise_refiner", "ref_image_refiner", "context_refiner"):
                group = getattr(model, attr, None)
                if group is not None:
                    topology.append(
                        {
                            "name": attr,
                            "attr_path": attr,
                            "count": len(group),
                            "approx_vram_mb": 250,
                        },
                    )
        return topology
