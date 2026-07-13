"""Pre-training VRAM estimation utility.

Calculates expected GPU memory consumption for a training run *before*
any weights are loaded.  The estimate is intentionally conservative
so users see a worst-case budget.

Usage::

    from app.engine.utils.vram_estimator import VRAMEstimator
    report = VRAMEstimator.estimate(definition, config)
    # report.fits  → True / False
    # report.peak_mb  → worst-case peak VRAM in MB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Well-known model sizes (B params) — used when introspection is unavailable
# ---------------------------------------------------------------------------
# NOTE: a definition YAML's ``model_size_mb`` is authoritative when present —
# the estimator prefers it; this table is the fallback-only path. (Several
# families currently ship an empty/zero ``model_size_mb``, so for those the
# fallback is what actually drives the estimate.)
#
# Provenance (audit FAM-7, 2026-07): the six entries added for ideogram4,
# krea2, ltx2, microsoft_lens, wan21, and wan22 were derived by instantiating
# each family's vendored transformer/text-encoder config on the meta device
# (no weights loaded) and counting parameters — microsoft_lens is the
# exception, calibrated instead from its definition's real on-disk
# ``model_size_mb`` (size_mb / 2 bytes-per-param for bf16), since that
# definition ships concrete sizes rather than an empty/zero table. All six
# are pinned by ``backend/tests/test_vram_estimator_families.py``.
_FAMILY_PARAMS: dict[str, dict[str, float]] = {
    "sdxl": {
        "unet": 2.6,
        "text_encoder_1": 0.12,
        "text_encoder_2": 0.35,
        "vae": 0.08,
    },
    "flux1": {
        "transformer": 12.0,
        "text_encoder_clip": 0.12,
        "text_encoder_t5": 4.8,
        "vae": 0.08,
    },
    "flux2": {
        "transformer": 32.0,  # FLUX.2-dev default
        "text_encoder": 24.0,  # Mistral3 (dev)
        "vae": 0.17,
    },
    "chroma": {
        # Meta-instantiated diffusers ChromaTransformer2DModel with its 0.39
        # class defaults (== both chroma1-base's and chroma1-hd's real
        # transformer/config.json: 19 double + 38 single blocks, inner dim
        # 3072) → 8.899983424 B params (matches the "8.9B" community name).
        # Both definitions ship concrete on-disk model_size_mb (from the HF
        # tree API, no download needed), so this table is the fallback-only
        # path — but T5-XXL (text_encoder) is identical to flux1's own T5
        # entry, and the VAE is the FLUX.1-schnell AutoencoderKL verbatim.
        "transformer": 8.9,
        "text_encoder": 4.8,  # T5-XXL (google/t5-v1_1-xxl) — no CLIP at all
        "vae": 0.08,  # FLUX.1-schnell AutoencoderKL (~84M params)
    },
    "lumina2": {
        # Both fp32 shard totals (HF tree API, bytes/4) converge on ~2.6B —
        # matches the community "2.6B DiT" branding. The definition ships
        # concrete on-disk model_size_mb, so this table is the fallback-only
        # path; the VAE is the FLUX.1-dev AutoencoderKL verbatim (same 16ch
        # architecture as flux1's own VAE entry).
        "transformer": 2.6,
        "text_encoder": 2.6,  # Gemma-2-2B (google/gemma-2-2b)
        "vae": 0.08,  # FLUX.1-dev AutoencoderKL (~84M params)
    },
    "zimage": {
        "transformer": 6.2,
        "text_encoder": 4.0,  # Qwen3
        "vae": 0.08,
    },
    "omnigen2": {
        # fp32 shard totals (HF tree API, bytes/4, OmniGen2/OmniGen2):
        # transformer 15.87 GB → ~3.97 B; mllm 15.02 GB → ~3.75 B. The
        # definition ships concrete on-disk model_size_mb, so this table is
        # the fallback-only path; the VAE is the FLUX.1-dev AutoencoderKL
        # verbatim (same ~84M entry as flux1/lumina2).
        "transformer": 4.0,
        "text_encoder": 3.75,  # Qwen2.5-VL-3B-Instruct mllm (text-only use)
        "vae": 0.08,
    },
    "qwen_image": {
        "transformer": 20.4,
        "text_encoder": 8.3,  # Qwen2.5-VL
        "vae": 0.17,
    },
    "hidream_o1": {
        # Pixel-space UNIFIED transformer (visual blocks + language_model in a
        # single model): NO VAE and NO external text encoder. The explicit 0.0
        # entries override the generic fallbacks in _get_te_params/_get_vae_params
        # (which would otherwise invent a ~0.35B TE + ~0.08B VAE that don't exist).
        "transformer": 17.0,  # ~17B unified backbone (HiDream-O1-Image)
        "text_encoder": 0.0,  # none — text is handled inside the transformer
        "vae": 0.0,  # none — operates directly in pixel space
    },
    "ernie_image": {
        # ERNIE-Image-Base-8B. The definition ships concrete model_size_mb
        # (transformer 16000 MB, TE 6000 MB, VAE 335 MB) which the estimator
        # prefers; these are fallbacks calibrated to those on-disk sizes
        # (size_mb / 2 bytes-per-param for bf16).
        "transformer": 8.0,  # ~8B DiT (≈16 GB bf16)
        "text_encoder": 3.0,  # Mistral3/Ministral-3B + Pixtral vision (≈6 GB bf16)
        "vae": 0.17,  # AutoencoderKLFlux2 (≈335 MB)
    },
    "ideogram4": {
        # Counts meta-instantiated from the vendored Ideogram4Transformer2DModel
        # (34 layers, hidden 4608) — matches the definition's "9.3B" name.
        "transformer": 9.3,
        "text_encoder": 8.8,  # Qwen3-VL-8B (hidden 4096, 36 layers + vision)
        "vae": 0.08,  # AutoencoderKLFlux2 (84M params)
    },
    "krea2": {
        # Meta-instantiated from the diffusers Krea2Transformer2DModel defaults
        # (28 SwiGLU blocks, hidden 6144 — mirrors krea2_raw.yaml).
        "transformer": 12.8,
        "text_encoder": 4.4,  # Qwen3-VL-4B (12-layer-stacked features)
        "vae": 0.13,  # AutoencoderKLQwenImage (127M params)
    },
    "dreamlite": {
        # Meta-instantiated DreamLiteUNetModel with the REAL checkpoint
        # unet/config.json (block_out 256/512/896, tlpb 1/2/4, ff_mult 3,
        # sep-convs, MQA) → 0.390 B. Deliberately small — a mobile-class
        # U-Net, NOT a DiT.
        "unet": 0.39,
        "text_encoder": 2.1,  # Qwen3-VL-2B-class (text 2048/28L + vision 24L)
        "vae": 0.002,  # AutoencoderTiny / taesdxl (~2.4 M params)
    },
    "ovis_image": {
        # Meta-instantiated from the diffusers-0.39 OvisImageTransformer2DModel
        # checkpoint config (== class defaults: 6 double + 27 single blocks,
        # inner dim 3072) → 7.37B. TE meta-instantiated from the checkpoint's
        # text_encoder/config.json (Qwen3, hidden 2048, 28 layers) → 1.72B.
        "transformer": 7.4,
        "text_encoder": 1.7,  # Qwen3-1.7B (text-only)
        "vae": 0.04,  # Flux-style AutoencoderKL (16ch, 38M params)
    },
    "longcat_image": {
        # Meta-instantiated diffusers LongCatImageTransformer2DModel with its
        # 0.39 config defaults (19 double + 38 single blocks, inner dim 3072)
        # → 11.878 B params.
        "transformer": 11.9,
        "text_encoder": 8.3,  # Qwen2.5-VL-7B — same class/config as qwen_image
        "vae": 0.08,  # standard 16-channel AutoencoderKL (~84M params)
    },
    "prx": {
        # Meta-instantiated from the diffusers-0.39 PRXTransformer2DModel
        # checkpoint config (== class defaults: 16 blocks, hidden 1792)
        # → 1.17B. TE meta-instantiated from the checkpoint's
        # text_encoder/config.json (T5GemmaEncoder, hidden 2304, 26 layers)
        # → 2.61B. NOTE: the transformer is SMALLER than the generic 2.0B
        # fallback — the dedicated estimate test pins an upper bound to
        # prove this entry (not the default) drives the estimate.
        "transformer": 1.2,
        "text_encoder": 2.6,  # T5Gemma encoder (fp32 on disk, bf16 loaded)
        "vae": 0.08,  # Flux-style AutoencoderKL (84M params)
    },
    "prx_pixel": {
        # Meta-instantiated from the diffusers-0.39 PRXTransformer2DModel with
        # the prxpixel-t2i checkpoint config (PIXEL variant: 24 blocks, hidden
        # 3584, bottleneck img_in, resolution embeds) → 7.00B. TE
        # meta-instantiated from the checkpoint's text_encoder/config.json
        # (Qwen3VLTextModel, hidden 2048, 28 layers) → 1.72B. PIXEL-SPACE:
        # the explicit vae 0.0 overrides the generic ~0.08B fallback in
        # _get_vae_params (there is no VAE — the model denoises raw RGB).
        "transformer": 7.0,
        "text_encoder": 1.7,  # Qwen3-VL text backbone (no vision tower)
        "vae": 0.0,  # none — operates directly in pixel space
    },
    "kandinsky5": {
        # LITE (T2V) sizes — meta-instantiated from the checkpoint's
        # transformer/config.json (model_dim 1792, ff 7168, 32 visual blocks)
        # → 2.008 B. The I2V PRO definition (19.3 B) ships a concrete
        # ``model_size_mb`` (36833) which the estimator prefers, so the family
        # fallback deliberately carries the Lite numbers.
        "transformer": 2.0,
        "text_encoder": 8.3,  # Qwen2.5-VL-7B (same class/config as qwen_image)
        "text_encoder_2": 0.12,  # CLIP ViT-L text tower
        "vae": 0.25,  # AutoencoderKLHunyuanVideo (246M params)
    },
    "ltx2": {
        # Meta-instantiated diffusers LTX2VideoTransformer3DModel with the
        # ltx2_3.yaml arch (48 layers, hidden 4096, joint audio+video streams).
        "transformer": 18.9,
        "text_encoder": 12.0,  # Gemma3-12B (hidden 3840)
        "vae": 1.2,  # AutoencoderKLLTX2Video (1.22B params)
    },
    "microsoft_lens": {
        # lens_base.yaml ships concrete model_size_mb (transformer 7600 MB,
        # TE 40000 MB, VAE 335 MB) which the estimator prefers; these are
        # fallbacks calibrated to those on-disk sizes (size_mb / 2 for bf16).
        "transformer": 3.8,  # Lens Base 3.8B DiT
        "text_encoder": 20.0,  # GPT-OSS-20B
        "vae": 0.17,
    },
    "wan21": {
        # Meta-instantiated diffusers WanTransformer3DModel with the definition
        # arch (40 layers, hidden 5120, ffn 13824). 14B T2V default; the I2V
        # 14B variant is ≈16.4B and the 1.3B variant ≈1.4B.
        "transformer": 14.3,
        "text_encoder": 5.7,  # UMT5-XXL encoder
        "vae": 0.13,  # AutoencoderKLWan (127M params)
    },
    "hunyuan_video15": {
        # Meta-instantiated diffusers-0.39 HunyuanVideo15Transformer3DModel
        # with the verified 480p checkpoint config (54 dual-stream blocks,
        # inner dim 2048, 2 refiner layers) → 8.33 B params.
        "transformer": 8.3,
        # Dual TE (summed by _get_te_params): Qwen2.5-VL-7B TEXT tower
        # (hidden 3584, 28 layers → 7.07 B meta-measured) + ByT5 glyph
        # encoder (d_model 1472, 12 layers → 0.22 B).
        "text_encoder_qwen": 7.1,
        "text_encoder_byt5": 0.22,
        "vae": 1.26,  # AutoencoderKLHunyuanVideo15 (1.26 B, meta-measured)
    },
    "wan22": {
        # PER-EXPERT size (same arch as wan21 14B). The MoE second expert is
        # added by the dual_expert branch in VRAMEstimator.estimate, not here.
        "transformer": 14.3,
        "text_encoder": 5.7,  # UMT5-XXL encoder
        "vae": 0.13,  # AutoencoderKLWan (127M params)
    },
    "wan22_ti2v_5b": {
        # Meta-instantiated diffusers WanTransformer3DModel with the REAL
        # Wan2.2-TI2V-5B-Diffusers transformer/config.json (30 layers, hidden
        # 24*128=3072, ffn 14336, in/out channels 48) → exactly 5.000 B params
        # (the checkpoint's own "5B" name). Dense — no second expert.
        "transformer": 5.0,
        "text_encoder": 5.7,  # UMT5-XXL encoder — same TE as wan21/wan22
        # NOT the wan21/wan22 0.13 (127M) VAE — TI2V-5B ships a NEW higher-
        # compression AutoencoderKLWan (z_dim 48, base_dim 160, decoder_base_dim
        # 256 vs the older VAE's 16/96/96) meta-instantiated to 0.671 B params.
        "vae": 0.67,
    },
    "boogu_image": {
        # CORRECTED 2026-07 (final-review Finding 1): task-2/3-brief.md's
        # "transformer 10.3 GB, mllm 8.8 GB, vae 0.08 GB" was a transcription
        # error — those are PARAM COUNTS IN BILLIONS (10.29B / 8.77B / ~84M),
        # not on-disk GB, as the brief's own prose confirms elsewhere
        # ("~10.29B... hidden 3360" transformer; "~8.77B, 36 layers" mllm).
        # Verified directly via HfApi(files_metadata=True) shard-size totals
        # on Boogu/Boogu-Image-0.1-Base (identical on -Turbo): transformer
        # 20,585,331,562 bytes / 2 bytes-per-param (bf16) = 10.29B; mllm
        # 17,545,915,828 bytes / 2 (bf16) = 8.77B; vae 335,307,052 bytes / 4
        # bytes-per-param (fp32 on disk) = 0.084B (~84M). Both definitions'
        # model_size_mb now ship the true on-disk MB (19632/16733/320), which
        # the estimator prefers via _get_component_disk_mb; this entry is the
        # true FALLBACK, in billions of params — same convention as
        # ideogram4's identical Qwen3-VL-8B "text_encoder": 8.8 entry.
        "transformer": 10.3,  # BooguImageTransformer2DModel, ~10.29B
        "text_encoder": 8.8,  # Qwen3-VL-8B mllm, ~8.77B (== ideogram4's TE)
        "vae": 0.08,  # FLUX-style AutoencoderKL, ~84M params (fp32 on disk)
    },
}

# Bytes-per-param for common dtypes
_DTYPE_BYTES: dict[str, int] = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.int8": 1,
}

# Bits-per-param after quantization (mirrors QuantizationFactory.bits_map)
_QUANT_BITS: dict[str, float] = {
    "none": 0,  # sentinel — uses native dtype
    "bf16": 16,
    "nvfp4": 4,
    "fp8": 8,
    "nf4": 4,
    "int4": 4,
    "int5": 5,
    "int6": 6,
    "int7": 7,
    "int8": 8,
}

# Plausible band for a per-component VRAM calibration multiplier
# (measured ÷ analytic). Calibration is a modest correction; anything outside
# this band is treated as corrupt/stale and ignored (see § 7b in ``estimate``).
_CALIB_MIN = 0.1
_CALIB_MAX = 4.0


# ---------------------------------------------------------------------------
# Result data-class
# ---------------------------------------------------------------------------


@dataclass
class VRAMReport:
    """Structured VRAM estimation result."""

    # --- Per-category breakdown (MB) ---
    model_weights_mb: float = 0.0
    lora_adapters_mb: float = 0.0
    optimizer_states_mb: float = 0.0
    gradients_mb: float = 0.0
    activations_mb: float = 0.0
    overhead_mb: float = 1024.0  # CUDA context + kernels (~1 GB)

    # --- Phase peaks ---
    caching_peak_mb: float = 0.0  # TE caching phase
    training_peak_mb: float = (
        0.0  # training phase (model + adapters + optim + grads + acts)
    )

    # --- Summary ---
    peak_mb: float = 0.0  # max(caching, training)
    available_mb: float = 0.0  # FREE VRAM (total − used by all processes)
    total_mb: float = 0.0  # total card VRAM
    used_mb: float = 0.0  # already in use by other processes (ComfyUI, browser, …)
    fits: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_weights_mb": round(self.model_weights_mb),
            "lora_adapters_mb": round(self.lora_adapters_mb),
            "optimizer_states_mb": round(self.optimizer_states_mb),
            "gradients_mb": round(self.gradients_mb),
            "activations_mb": round(self.activations_mb),
            "overhead_mb": round(self.overhead_mb),
            "caching_peak_mb": round(self.caching_peak_mb),
            "training_peak_mb": round(self.training_peak_mb),
            "peak_mb": round(self.peak_mb),
            "available_mb": round(self.available_mb),
            "total_mb": round(self.total_mb),
            "used_mb": round(self.used_mb),
            "fits": self.fits,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class VRAMEstimator:
    """Estimate peak VRAM for a model + config before loading weights.

    The estimator works from:
    - ``ModelDefinition`` metadata (family, detected_precision, architecture_params)
    - Training config dict (quantization, lora_rank, train_text_encoder, resolution, batch_size, etc.)
    - Live GPU info (optional)
    """

    @staticmethod
    def estimate(
        definition: Any,
        config: dict[str, Any],
        calibration: dict[str, float] | None = None,
    ) -> VRAMReport:
        """Build a VRAM report for the given model + training config.

        Args:
            definition: ``ModelDefinition`` instance (or dict-like with
                        ``family``, ``detected_precision``, ``architecture_params``).
            config:     Training config dict.
            calibration: Optional per-definition multipliers learned from
                        measured peaks, e.g. ``{"train": 0.9, "cache": 1.1}``.
                        When present, the analytic ``training_peak_mb`` /
                        ``caching_peak_mb`` are scaled toward observed reality.

        Returns:
            ``VRAMReport`` with per-category breakdown and fit assessment.
        """
        report = VRAMReport()

        family = getattr(definition, "family", None) or "unknown"
        precision = getattr(definition, "detected_precision", {}) or {}
        arch = getattr(definition, "architecture_params", {}) or {}
        size_mb = getattr(definition, "model_size_mb", {}) or {}

        # ── 1. Model weight size (primary trainable component) ───────────
        primary_key = "unet"
        native_bpp = _bytes_per_param(precision.get(primary_key, "torch.bfloat16"))

        quant_scheme = config.get("quantization", "none")
        if quant_scheme != "none" and quant_scheme in _QUANT_BITS:
            effective_bpp = _QUANT_BITS[quant_scheme] / 8
        else:
            effective_bpp = native_bpp

        # Use model_size_mb if available (most accurate), else fall back to param estimate
        primary_disk_mb = _get_component_disk_mb(size_mb, primary_key)
        if primary_disk_mb > 0:
            # Scale on-disk size by quantization ratio
            report.model_weights_mb = primary_disk_mb * (effective_bpp / native_bpp)
            primary_params_b = (primary_disk_mb * 1024 * 1024) / (native_bpp * 1e9)
        else:
            primary_params_b = _get_primary_params(family, arch, primary_key)
            model_bytes = primary_params_b * 1e9 * effective_bpp
            report.model_weights_mb = model_bytes / (1024 * 1024)

        # ── 1b. Dual-expert (WAN 2.2 MoE) second transformer ─────────────
        # WAN 2.2 is a Mixture-of-Experts: a high-noise + a low-noise expert,
        # each the same size as ``model_weights_mb`` above. How much extra GPU
        # VRAM the second expert costs depends on the placement mode:
        #   - "resident": BOTH experts live on the GPU → +1× weight term (2×).
        #   - "swap":     one expert on GPU, the other pinned in CPU RAM and
        #                 swapped in on the boundary → ~0 extra GPU bytes.
        #   - "auto":     conservative — assume resident (worst case) so the
        #                 budget never under-promises.
        # Image / single-transformer families never enter this branch (guarded
        # by ``dual_expert``), so their estimate is byte-identical.
        caps = _resolve_caps(definition)
        is_dual_expert = bool(caps.get("dual_expert", False))
        if is_dual_expert:
            swap_mode = config.get("expert_swap_mode", "auto")
            if swap_mode == "swap":
                # GPU holds one expert; the second is pinned in host RAM.
                expert_weights_mb = 0.0
            else:  # "resident" or "auto" (conservative)
                expert_weights_mb = report.model_weights_mb
            report.model_weights_mb += expert_weights_mb

        # ── 2. LoRA adapters ─────────────────────────────────────────────
        lora_rank = config.get("lora_rank", config.get("rank", 16))
        # Rough estimate: each target module gets rank×in + rank×out params
        # Typical ratio: ~1-3% of model params at rank 16
        lora_ratio = min(lora_rank / 16 * 0.015, 0.10)  # cap at 10%
        lora_params = primary_params_b * 1e9 * lora_ratio
        lora_bpp = 2  # adapters always bf16/fp16
        report.lora_adapters_mb = (lora_params * lora_bpp) / (1024 * 1024)

        # ── 3. Optimizer states (AdamW: 2 fp32 moments per trainable param) ──
        trainable_params = lora_params
        train_te = config.get("train_text_encoder", False)
        if train_te:
            te_params_b = _get_te_params(family)
            trainable_params += te_params_b * 1e9

        optimizer = config.get("optimizer", "adamw")
        if optimizer in ("adamw", "adam", "adam8bit", "adamw8bit"):
            # 2 moments × fp32 (4 bytes) = 8 bytes per trainable param
            # 8-bit optimizers halve this
            moment_bytes = 4 if "8bit" not in optimizer else 2
            report.optimizer_states_mb = (trainable_params * 2 * moment_bytes) / (
                1024 * 1024
            )
        elif optimizer in ("prodigy", "prodigyopt"):
            # Prodigy stores ~3× fp32 states per param
            report.optimizer_states_mb = (trainable_params * 12) / (1024 * 1024)
        else:
            # SGD / other — 1 momentum buffer
            report.optimizer_states_mb = (trainable_params * 4) / (1024 * 1024)

        # ── 4. Gradients ─────────────────────────────────────────────────
        report.gradients_mb = (trainable_params * lora_bpp) / (1024 * 1024)

        # ── 5. Activations (highly dependent on resolution + batch) ──────
        # The scalar ``resolution``/``width`` keys (image-archetype defaults)
        # win when present; otherwise the spatial term derives from the
        # resolution LISTS. Phase 3: F=1 stills mixed into a video job bucket at
        # ``still_resolutions`` and can exceed the video ``resolutions`` — fold
        # them in via the shared resolver (single source of truth) so a
        # high-res still isn't silently under-budgeted. Monotonic: this can only
        # RAISE the (already conservative) scalar default, never lower it, so
        # every existing estimate is unchanged unless a real bucket edge is
        # genuinely larger. Image families inherit ``resolutions`` (the field is
        # is_video-gated), so a stale ``still_resolutions`` can't affect them.
        is_video = bool(_is_video_definition(definition))
        resolution = config.get("resolution", config.get("width", 1024))
        from app.engine.core.pipeline.pipeline_data import resolve_still_resolutions

        bucket_edges = [
            int(r) for r in (config.get("resolutions") or []) if int(r) > 0
        ] + [
            int(r)
            for r in resolve_still_resolutions(config, is_video)
            if int(r) > 0
        ]
        if bucket_edges:
            resolution = max(int(resolution), max(bucket_edges))
        batch_size = config.get("batch_size", 1)
        grad_checkpointing = config.get("gradient_checkpointing", True)
        # gradient_accumulation_steps doesn't affect peak VRAM (same batch in memory)

        # ── 5a. Video temporal scaling ───────────────────────────────────
        # A video clip carries a temporal axis: the transformer processes
        # ``latent_frames`` latent timesteps at once, so its activation memory
        # scales ~linearly with the number of *latent* frames. The VAE encodes
        # ``num_frames`` pixel frames to ``latent_frames = (F - 1) / t + 1``
        # latent frames, where ``t`` is the VAE temporal-compression ratio
        # (``video.vae_temporal`` — 4 for WAN, 8 for LTX2).
        #
        # For an IMAGE family ``latent_frames`` collapses to 1 (temporal_ratio
        # defaults to 1 and num_frames defaults to 1), so the activation term —
        # and therefore the whole estimate — stays BYTE-IDENTICAL to before.
        # (``is_video`` resolved above, before the spatial term.)
        latent_frames = 1
        if is_video:
            temporal_ratio = int(arch.get("video.vae_temporal", 1) or 1)
            num_frames = int(config.get("num_frames", 1) or 1)
            if temporal_ratio > 1:
                latent_frames = max((num_frames - 1) // temporal_ratio + 1, 1)
            else:
                latent_frames = max(num_frames, 1)

        # Rough activation estimate:
        # Without grad checkpointing: ~resolution² × depth × hidden × batch × 2 bytes
        # With grad checkpointing: ~1/3 of above
        hidden_size = arch.get("hidden_size", 3072)
        depth = arch.get("depth", 19) + arch.get("depth_single_blocks", 38)
        pixels = (resolution // 8) ** 2  # latent space
        # Multiply by latent_frames so more frames → more activation memory
        # (image: latent_frames=1 → unchanged).
        act_bytes = (
            pixels * latent_frames * depth * hidden_size * batch_size * 2
        )  # bf16
        act_factor = 0.33 if grad_checkpointing else 1.0
        report.activations_mb = (act_bytes * act_factor) / (1024 * 1024)

        # Cap activations at a reasonable max (empirical). Video clips legitimately
        # need a higher ceiling than stills — scale the cap by latent_frames so a
        # long clip's frame-driven growth isn't immediately clamped away (image:
        # latent_frames=1 → identical caps to before).
        max_act_mb = (8192 if not grad_checkpointing else 4096) * latent_frames
        report.activations_mb = min(report.activations_mb, max_act_mb)

        # ── 6. Training peak ─────────────────────────────────────────────
        report.training_peak_mb = (
            report.model_weights_mb
            + report.lora_adapters_mb
            + report.optimizer_states_mb
            + report.gradients_mb
            + report.activations_mb
            + report.overhead_mb
        )

        # ── 7. Caching peak (TE on GPU during embedding generation) ─────
        te_bpp = _bytes_per_param(precision.get("text_encoder", "torch.bfloat16"))
        te_quant = config.get("te_quantization", "none")
        if te_quant != "none" and te_quant in _QUANT_BITS:
            te_effective_bpp = _QUANT_BITS[te_quant] / 8
        else:
            te_effective_bpp = te_bpp

        te_disk_mb = _get_component_disk_mb(size_mb, "text_encoder")
        if te_disk_mb > 0:
            te_mb = te_disk_mb * (te_effective_bpp / te_bpp)
        else:
            te_total_params_b = _get_te_params(family)
            te_mb = (te_total_params_b * 1e9 * te_effective_bpp) / (1024 * 1024)

        # During caching: TE on GPU + VAE might be loaded too
        vae_disk_mb = _get_component_disk_mb(size_mb, "vae")
        if vae_disk_mb > 0:
            vae_mb = vae_disk_mb
        else:
            vae_mb = (_get_vae_params(family) * 1e9 * 2) / (1024 * 1024)
        report.caching_peak_mb = te_mb + vae_mb + report.overhead_mb

        # ── 7b. Calibration (measured ÷ analytic, learned from local runs) ──
        # Per-component multipliers refine each analytic row toward measured
        # reality; the training peak is re-summed from the calibrated parts.
        # ``caching_peak_mb`` scales the (single-number) caching-phase peak.
        #
        # A calibration coefficient is a MODEST correction — measured VRAM is
        # within a small factor of the (conservative, worst-case) analytic
        # estimate. A multiplier well outside ``[_CALIB_MIN, _CALIB_MAX]`` is
        # not "reality is 10× bigger" — it signals STALE / ORPHANED / unit-
        # corrupt stats (e.g. a definition whose ``job_history`` rows were
        # deleted so ``recompute`` can never self-correct the coefficients).
        # Applying such a value unbounded produced the live 587 GB ltx2 estimate
        # (caching_peak 26 GB × 22.9 → 601 GB). Reject the implausible ones so
        # the affected component reverts to its uncalibrated analytic value.
        if calibration:
            for field in (
                "model_weights_mb",
                "lora_adapters_mb",
                "optimizer_states_mb",
                "gradients_mb",
                "activations_mb",
                "overhead_mb",
            ):
                k = _sane_calibration(calibration.get(field), field, report)
                if k is not None:
                    setattr(report, field, getattr(report, field) * k)
            report.training_peak_mb = (
                report.model_weights_mb
                + report.lora_adapters_mb
                + report.optimizer_states_mb
                + report.gradients_mb
                + report.activations_mb
                + report.overhead_mb
            )
            cache_k = _sane_calibration(
                calibration.get("caching_peak_mb"), "caching_peak_mb", report
            )
            if cache_k is not None:
                report.caching_peak_mb *= cache_k

        # ── 8. Overall peak ──────────────────────────────────────────────
        report.peak_mb = max(report.training_peak_mb, report.caching_peak_mb)

        # ── 9. GPU availability ──────────────────────────────────────────
        # Fit against ACTUALLY FREE VRAM (total − used). NVML's ``used`` is
        # device-wide, so VRAM held by other processes (ComfyUI, the browser,
        # another training run) is already accounted for — our analytic peak
        # only models our own consumption, so the headroom must come from the
        # live device free figure, not the card's total capacity.
        try:
            from app.core.system_monitor import system_monitor

            snap = system_monitor.snapshot()
            if snap.gpus:
                gpu = snap.gpus[0]
                report.total_mb = gpu.vram_total_mb
                report.used_mb = gpu.vram_used_mb
                report.available_mb = max(gpu.vram_total_mb - gpu.vram_used_mb, 0)
                report.fits = report.peak_mb < report.available_mb * 0.95  # 5% headroom
        except Exception:
            report.warnings.append("Could not query GPU — fit check skipped")

        # ── 10. Warnings ─────────────────────────────────────────────────
        # Flag significant VRAM already held by other apps (>1 GB) so a tight
        # fit is explainable ("would fit on an empty card, but ComfyUI…").
        if report.used_mb > 1024:
            report.warnings.append(
                f"{round(report.used_mb / 1024, 1)} GB VRAM is already in use by "
                f"other processes — only {round(report.available_mb / 1024, 1)} GB "
                f"of {round(report.total_mb / 1024, 1)} GB is free. Close other GPU "
                f"apps (e.g. ComfyUI) to reclaim it."
            )

        if report.peak_mb > 0 and report.available_mb > 0:
            ratio = report.peak_mb / report.available_mb
            if ratio > 1.0:
                overshoot_mb = round(report.peak_mb - report.available_mb)
                report.warnings.append(
                    f"Estimated peak VRAM ({round(report.peak_mb)} MB) exceeds "
                    f"free ({round(report.available_mb)} MB) by {overshoot_mb} MB. "
                    f"Consider quantization, lower resolution, or smaller batch size."
                )
            elif ratio > 0.85:
                report.warnings.append(
                    f"Estimated VRAM usage is {round(ratio * 100)}% of free VRAM — "
                    f"tight fit. May OOM under activation spikes."
                )

        if not config.get("gradient_checkpointing", True):
            report.warnings.append(
                "Gradient checkpointing is OFF — activation memory will be significantly higher."
            )

        # ── Quantization arch compatibility ──────────────────────────
        _check_quant_compat(quant_scheme, "Model quantization", report, config)
        _check_quant_compat(te_quant, "TE quantization", report, config)

        logger.info(
            "vram_estimate",
            family=family,
            peak_mb=round(report.peak_mb),
            available_mb=round(report.available_mb),
            fits=report.fits,
        )

        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_caps(definition: Any) -> dict:
    """Resolve the merged capability descriptor for *definition*.

    Returns the ``capabilities`` sub-dict from
    :func:`app.engine.core.archetypes.resolve_capabilities` (which merges the
    archetype template with the family's ``capability_overrides``), or an empty
    dict when the definition can't be resolved (unknown family, dict stub in a
    unit test, registry not initialised). Best-effort and never raises — the
    estimator must keep working for every input.
    """
    try:
        from app.engine.core.archetypes import resolve_capabilities

        return resolve_capabilities(definition).get("capabilities", {}) or {}
    except Exception:
        return {}


def _is_video_definition(definition: Any) -> bool:
    """True when *definition* describes a video family.

    Prefers the resolved ``is_video`` capability flag; falls back to the
    presence of a ``video.vae_temporal`` architecture param so the temporal
    scaling still kicks in even if the family lookup fails. Image families have
    neither, so this returns False and the estimate is unchanged.
    """
    if _resolve_caps(definition).get("is_video", False):
        return True
    arch = getattr(definition, "architecture_params", {}) or {}
    return "video.vae_temporal" in arch


def _get_component_disk_mb(size_mb: dict, key: str) -> float:
    """Look up on-disk component size in MB from model_size_mb dict.

    Tries *key* first, then common aliases (transformer/unet).
    Returns 0 when not found.
    """
    for k in (key, "transformer", "unet"):
        val = size_mb.get(k, 0)
        if val > 0:
            return float(val)
    return 0.0


def _get_primary_params(family: str, arch: dict, key: str = "unet") -> float:
    """Get primary model param count in billions (fallback only)."""
    # Try architecture params first (from introspection)
    total_params = arch.get("total_params", 0)
    if total_params > 0:
        return total_params / 1e9

    # Fall back to well-known sizes
    family_data = _FAMILY_PARAMS.get(family, {})
    for k in (key, "transformer", "unet", "model"):
        if k in family_data:
            return family_data[k]
    return 2.0  # conservative default


def _get_te_params(family: str) -> float:
    """Get total text encoder param count in billions (fallback only).

    A *known* family that declares one or more ``text_encoder*`` keys is
    authoritative — including the case where they sum to 0.0 (e.g. unified
    pixel-space models like HiDream-O1 that have no external text encoder).
    Only fall back to the generic ~350M default for families we don't know.
    """
    family_data = _FAMILY_PARAMS.get(family, {})
    te_keys = [v for k, v in family_data.items() if "text_encoder" in k]
    if te_keys:
        return sum(te_keys)
    return 0.35  # default ~350M for unknown families


def _get_vae_params(family: str) -> float:
    """Get VAE param count in billions (fallback only)."""
    return _FAMILY_PARAMS.get(family, {}).get("vae", 0.08)


def _bytes_per_param(dtype_str: str) -> int:
    """Get bytes per parameter for a dtype string."""
    return _DTYPE_BYTES.get(dtype_str, 2)  # default bf16


def _sane_calibration(
    k: Any, field: str, report: VRAMReport
) -> float | None:
    """Validate a per-component calibration multiplier.

    Returns the multiplier when it is a positive number inside the plausible
    ``[_CALIB_MIN, _CALIB_MAX]`` band, else ``None`` (caller leaves the
    component uncalibrated). An out-of-band value indicates stale/orphaned or
    unit-corrupt stats, so it is ignored and a warning is recorded rather than
    inflating the estimate by an order of magnitude.
    """
    try:
        k = float(k)
    except (TypeError, ValueError):
        return None
    if k <= 0:
        return None
    if k < _CALIB_MIN or k > _CALIB_MAX:
        report.warnings.append(
            f"Ignored an implausible VRAM calibration factor for {field} "
            f"({k:.2f}× — outside {_CALIB_MIN}–{_CALIB_MAX}×). The stored "
            f"per-definition stats look stale or corrupt; using the analytic "
            f"estimate for this component."
        )
        return None
    return k


def _check_quant_compat(
    scheme: str, label: str, report: VRAMReport, config: dict[str, Any]
) -> None:
    """Add a warning to *report* if *scheme* isn't supported on this GPU."""
    if scheme in ("none", "bf16"):
        return

    try:
        import torch

        if not torch.cuda.is_available():
            return
        cap = torch.cuda.get_device_capability()
        sm = cap[0] * 10 + cap[1]
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        return

    # NVFP4 requires SM >= 100
    if scheme == "nvfp4" and sm < 100:
        from app.engine.factories.quantization import QuantizationFactory

        backend_name = config.get("te_quantization_backend", "auto")
        fallback, scheme = QuantizationFactory.validate_and_fallback(
            scheme, backend_name
        )
        report.warnings.append(
            f"{label} '{scheme}' requires Blackwell (SM ≥ 100) but {gpu_name} has SM {sm}. "
            f"Will fall back to '{fallback}' at runtime."
        )

    # FP8 requires SM >= 89
    elif scheme == "fp8" and sm < 89:
        from app.engine.factories.quantization import QuantizationFactory

        backend_name = config.get("quantization_backend", "auto")
        fallback, scheme = QuantizationFactory.validate_and_fallback(
            scheme, backend_name
        )
        report.warnings.append(
            f"{label} '{scheme}' requires Ada/Hopper/Blackwell (SM ≥ 89) but {gpu_name} has SM {sm}. "
            f"Will fall back to '{fallback}' at runtime."
        )
