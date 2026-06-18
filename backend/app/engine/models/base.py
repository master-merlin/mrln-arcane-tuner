"""Base training configuration schema and plugin abstraction.

Defines the Pydantic ``BaseTrainingConfig`` schema (consumed by the UI
and engine) and the ``TrainingPlugin`` abstraction for pluggable
training backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


# ── Configuration Schema ─────────────────────────────────────────────────


class SamplePromptConfig(BaseModel):
    """A single sampling prompt with generation parameters.

    Used by ``GenericSamplingPipeline`` to generate sample images during
    training.  Supports ``[triggerword]`` and ``[captionprefix]`` wildcards
    in the ``prompt`` field.
    """

    prompt: str = Field(
        "",
        description="Text prompt (supports [triggerword] and [captionprefix] wildcards)",
        json_schema_extra={"group": "SAMPLING"},
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility",
        json_schema_extra={"group": "SAMPLING", "min": 0, "max": 4294967295},
    )
    width: int = Field(
        1024,
        description="Output width (px, quantized to 64)",
        json_schema_extra={"group": "SAMPLING", "min": 256, "max": 2048, "step": 64},
    )
    height: int = Field(
        1024,
        description="Output height (px, quantized to 64)",
        json_schema_extra={"group": "SAMPLING", "min": 256, "max": 2048, "step": 64},
    )
    num_inference_steps: int = Field(
        20,
        description="Denoising steps",
        json_schema_extra={"group": "SAMPLING", "min": 1, "max": 100, "step": 1},
    )
    guidance_scale: float = Field(
        3.5,
        description="CFG scale (0 = no guidance)",
        json_schema_extra={"group": "SAMPLING", "min": 0.0, "max": 20.0, "step": 0.5},
    )
    control_images: list[str] = Field(
        default_factory=list,
        description="Control image path(s) for edit-model sampling (the 'before' image). Ignored by standard models.",
        json_schema_extra={"group": "SAMPLING"},
    )
    # ── Video sampling (optional) ────────────────────────────────────────
    # When set, a video-capable family samples a short clip instead of a
    # still.  Left ``None`` for image families so existing configs are
    # unaffected.  Capability-based UI gating is handled in a later phase.
    num_frames: int | None = Field(
        None,
        description="Frames to sample for video models (None = still image)",
        json_schema_extra={
            "group": "SAMPLING",
            "min": 1,
            "max": 256,
            "step": 1,
            "video_only": True,
        },
    )
    fps: float | None = Field(
        None,
        description="Frames-per-second for the sampled video clip (None = image)",
        json_schema_extra={
            "group": "SAMPLING",
            "min": 1.0,
            "max": 60.0,
            "step": 1.0,
            "video_only": True,
        },
    )


class DatasetItem(BaseModel):
    """Single dataset entry within a training configuration."""

    dataset_name: str = Field(
        ...,
        description="Name of the dataset to use",
        json_schema_extra={"group": "CONCEPTS"},
    )
    caption_prefix: str = Field(
        "",
        description="Prefix to add to all captions in this dataset",
        json_schema_extra={"group": "CONCEPTS"},
    )
    caption_dropout_rate: float = Field(
        0.1,
        description="Chance of dropping the caption (enables CFG at inference)",
        json_schema_extra={"group": "CONCEPTS", "min": 0.0, "max": 1.0, "step": 0.05},
    )
    num_repeats: int = Field(
        1,
        description="Number of times to repeat this dataset",
        json_schema_extra={"group": "CONCEPTS"},
    )
    num_frames: int = Field(
        0,
        description=(
            "Frames to use from each VIDEO in this dataset. 0 = inherit the "
            "run's general Video setting; a value overrides it for this dataset "
            "only (snapped to the model's frame rule). Images are always 1."
        ),
        json_schema_extra={
            "group": "CONCEPTS",
            "video_only": True,
            "min": 0,
            "max": 257,
            "step": 1,
        },
    )
    ignore_filter: bool = Field(
        False,
        description="If True, use ALL images regardless of exclusions",
        json_schema_extra={"group": "CONCEPTS", "hidden": True},
    )

    # Caption usage (per-dataset)
    use_captions: bool = Field(
        True,
        description="Train with this dataset's captions (off = trigger word / prefix only)",
        json_schema_extra={"group": "CONCEPTS", "inline_group": "caption_toggles"},
    )
    use_model_aware_captions: bool = Field(
        True,
        description="Prefer the model-aware caption variant for the selected model",
        json_schema_extra={
            "group": "CONCEPTS",
            "inline_group": "caption_toggles",
            "depends_on": "use_captions",
        },
    )

    # Masked training (per-dataset)
    masking_enabled: bool = Field(
        False,
        description="Include masked variants",
        json_schema_extra={"group": "CONCEPTS", "inline_group": "masking_toggles"},
    )
    recreate_masks: bool = Field(
        False,
        description="Re-generate masked images",
        json_schema_extra={
            "group": "CONCEPTS",
            "inline_group": "masking_toggles",
            "depends_on": "masking_enabled",
        },
    )
    mask_opacity: float = Field(
        0.0,
        description="Opacity used when re-creating masked images (0 = fully transparent background)",
        json_schema_extra={
            "group": "CONCEPTS",
            "depends_on": "masking_enabled",
            "disabled_if": {"recreate_masks": False},
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )
    original_weight: float = Field(
        1.0,
        description="Probability of selecting original variant (≥ 0.50)",
        json_schema_extra={
            "group": "CONCEPTS",
            "depends_on": "masking_enabled",
            "min": 0.50,
            "max": 1.0,
            "step": 0.05,
        },
    )


class BaseTrainingConfig(BaseModel):
    """Master training configuration schema.

    Groups: BASE, STRATEGY, NETWORK, OPTIMIZER, ENGINE, VIDEO, SAMPLING.
    Field metadata drives the dynamic UI form generation.
    """

    # [BASE] General Settings
    lora_prefix: str = Field(
        "",
        description="Prefix for LoRA filename (auto-derived from dataset name)",
        json_schema_extra={"group": "BASE"},
    )
    lora_suffix: str = Field(
        "",
        description="Suffix for LoRA filename (auto-derived from dataset name)",
        json_schema_extra={"group": "BASE"},
    )
    lora_name: str = Field(
        "my_lora",
        description="LoRA filename — supports {placeholder} syntax for dynamic naming",
        json_schema_extra={"group": "BASE"},
    )
    global_triggerword: str = Field(
        "",
        description="Global triggerword (e.g. 'CarConcepts')",
        json_schema_extra={"group": "BASE"},
    )
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(
        "fp16", description="Training precision", json_schema_extra={"group": "BASE"}
    )
    save_precision: Literal["fp16", "bf16", "fp32"] = Field(
        "fp16",
        description="Precision of the saved LoRA (FP32 = 2x Size)",
        json_schema_extra={"group": "BASE"},
    )
    model_family: str = Field(
        "",
        description="The architectural family of the model",
        json_schema_extra={"group": "MODEL_SELECTION"},
    )
    definition_id: str = Field(
        "",
        description="The ID of the model definition to use (from YAML)",
        json_schema_extra={
            "group": "MODEL_SELECTION",
            "depends_on": "model_family",
            "hide_unsupported": True,
        },
    )
    quantization_backend: Literal[
        "auto", "torchao", "optimum-quanto", "bitsandbytes"
    ] = Field(
        "auto",
        description="The engine to use for base model quantization",
        json_schema_extra={"group": "MODEL_SELECTION"},
    )
    te_quantization_backend: Literal[
        "auto", "torchao", "optimum-quanto", "bitsandbytes"
    ] = Field(
        "auto",
        description="The engine to use for text encoder quantization",
        json_schema_extra={"group": "MODEL_SELECTION"},
    )
    quantization: Literal[
        "none",
        "fp8",
        "nvfp4",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
        "nf4",
        "qint8",
        "qint4",
        "qfloat8",
    ] = Field(
        "none",
        description="Quantize base model — FP8 auto-selects: training acceleration on Blackwell, VRAM savings on Ada/Hopper",
        json_schema_extra={
            "group": "MODEL_SELECTION",
            "depends_on": "quantization_backend",
        },
    )
    te_quantization: Literal[
        "none",
        "fp8",
        "nvfp4",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
        "nf4",
        "qint8",
        "qint4",
        "qfloat8",
    ] = Field(
        "none",
        description="Quantize frozen text encoder(s) to save VRAM",
        json_schema_extra={
            "group": "MODEL_SELECTION",
            "depends_on": "te_quantization_backend",
        },
    )
    quantization_strategy: Literal["fastest", "vram_safe"] = Field(
        "fastest",
        description="Quantization loading strategy: 'fastest' quantizes on GPU in-place, 'vram_safe' quantizes one component at a time",
        json_schema_extra={
            "group": "MODEL_SELECTION",
            "depends_on": "quantization:!none|te_quantization:!none",
        },
    )
    store_quantized_version: bool = Field(
        True,
        description="Cache quantized model to disk for instant reload (auto-skipped on Blackwell — not needed with native FP8 training)",
        json_schema_extra={
            "group": "MODEL_SELECTION",
            "depends_on": "quantization:!none|te_quantization:!none",
        },
    )
    output_dir: str = Field(
        "./outputs",
        description="Where to save results",
        json_schema_extra={"group": "BASE", "input_type": "path"},
    )
    datasets: list[DatasetItem] = Field(
        ...,
        min_length=1,
        description="List of datasets to train on",
        json_schema_extra={"group": "BASE"},
    )
    cache_latents: bool = Field(
        True,
        description="Cache latents to disk for speed",
        json_schema_extra={"group": "BASE", "inline_group": "data_toggles"},
    )
    h_flip: bool = Field(
        False,
        description="Random horizontal flip augmentation (50% chance per sample)",
        json_schema_extra={"group": "BASE", "inline_group": "data_toggles"},
    )
    v_flip: bool = Field(
        False,
        description="Random vertical flip augmentation (50% chance per sample)",
        json_schema_extra={"group": "BASE", "inline_group": "data_toggles"},
    )

    # [STRATEGY] Training Dynamics
    max_train_steps: int = Field(
        1000,
        description="Maximum number of steps",
        json_schema_extra={"group": "STRATEGY", "min": 1, "step": 100},
    )
    train_batch_size: int = Field(
        1,
        description="Batch size",
        json_schema_extra={"group": "STRATEGY", "min": 1, "max": 32, "step": 1},
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Steps before optimizer update",
        json_schema_extra={"group": "STRATEGY", "min": 1, "max": 128, "step": 1},
    )
    gradient_checkpointing: bool = Field(
        True,
        description="Trade speed for VRAM savings by recomputing activations (disable on 96GB+ for faster training)",
        json_schema_extra={"group": "STRATEGY"},
    )
    save_every_n_steps: int = Field(
        0,
        description="Save a checkpoint every N steps (0 to disable)",
        json_schema_extra={"group": "STRATEGY", "min": 0, "step": 50},
    )
    keep_last_checkpoints: int = Field(
        0,
        description="Keep only the last N checkpoints (0 = keep all)",
        json_schema_extra={
            "group": "STRATEGY",
            "min": 0,
            "max": 99,
            "step": 1,
            "depends_on": "save_every_n_steps:!0",
        },
    )
    persist_latents: bool = Field(
        True,
        description="Store latent cache manifest in checkpoints for resume",
        json_schema_extra={"group": "STRATEGY", "depends_on": "save_every_n_steps:!0"},
    )
    persist_embeddings: bool = Field(
        True,
        description="Store embedding cache manifest in checkpoints for resume",
        json_schema_extra={"group": "STRATEGY", "depends_on": "save_every_n_steps:!0"},
    )
    resume_from_checkpoint: str = Field(
        "",
        description="Path to a checkpoint directory to resume from",
        json_schema_extra={"group": "STRATEGY", "input_type": "path"},
    )
    use_cached_latents: bool = Field(
        False,
        description="Re-use latent cache from prior run (only encode new/changed images)",
        json_schema_extra={"group": "STRATEGY", "depends_on": "resume_from_checkpoint"},
    )
    use_cached_embeddings: bool = Field(
        False,
        description="Re-use embedding cache from prior run (only encode new captions)",
        json_schema_extra={"group": "STRATEGY", "depends_on": "resume_from_checkpoint"},
    )
    resolutions: list[int] = Field(
        [1024],
        description="Target resolutions for bucketing",
        json_schema_extra={"group": "STRATEGY"},
    )
    control_resolution: int = Field(
        0,
        description="Base resolution for control images in paired edit training (0 = follow the target's bucket). Qwen-Edit recommends 1024.",
        json_schema_extra={"group": "STRATEGY", "min": 0, "max": 2048, "step": 64},
    )
    resolution_strategy: Literal["mixed", "progressive"] = Field(
        "mixed",
        description="Mixed: all resolutions at once. Progressive: small resolutions first, scale up during training.",
        json_schema_extra={"group": "STRATEGY"},
    )
    bucketing_mode: Literal["kohya", "multi"] = Field(
        "kohya",
        description="Kohya: single best resolution per image. Multi: image appears at every qualifying resolution (more latent diversity).",
        json_schema_extra={"group": "STRATEGY"},
    )
    timestep_sampling: Literal[
        "logit_normal", "uniform", "sigmoid", "cosmap", "mode", "flux_shift", "radc", "model_shift"
    ] = Field(
        "logit_normal",
        description="Timestep sampling strategy for training",
        json_schema_extra={"group": "STRATEGY"},
    )
    logit_normal_mu: float = Field(
        0.0,
        description="Mean of the logit-normal distribution",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:logit_normal",
            "min": -2.0,
            "max": 2.0,
            "step": 0.1,
        },
    )
    logit_normal_sigma: float = Field(
        1.0,
        description="Std of the logit-normal distribution",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:logit_normal",
            "min": 0.1,
            "max": 3.0,
            "step": 0.1,
        },
    )
    model_shift_std: float = Field(
        1.0,
        description="Std of the logit-normal draw for model_shift timestep sampling",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:model_shift",
            "min": 0.1, "max": 3.0, "step": 0.1,
        },
    )
    timestep_uniform_prob: float = Field(
        0.1,
        description="Fraction of timesteps drawn uniformly (mixed into shifted modes)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:model_shift",
            "min": 0.0, "max": 1.0, "step": 0.05,
        },
    )
    mode_scale: float = Field(
        1.5,
        description="Scale for mode sampling (>1 = more mid-range emphasis)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:mode",
            "min": 1.0,
            "max": 5.0,
            "step": 0.1,
        },
    )
    flux_shift_base: float = Field(
        0.5,
        description="Base shift for low-res images",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:flux_shift",
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
        },
    )
    flux_shift_max: float = Field(
        1.16,
        description="Max shift for high-res images",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:flux_shift",
            "min": 0.5,
            "max": 3.0,
            "step": 0.1,
        },
    )
    radc_start: float = Field(
        0.8,
        description="Noise focus at training start (1.0=high noise, 0.0=clean)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:radc",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )
    radc_end: float = Field(
        0.2,
        description="Noise focus at training end (detail refinement)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:radc",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )
    radc_width: float = Field(
        0.5,
        description="Curve width (0.1=focused, 1.0=broad)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:radc",
            "min": 0.05,
            "max": 1.0,
            "step": 0.05,
        },
    )
    radc_res_influence: float = Field(
        0.15,
        description="Resolution cross-influence (0=off)",
        json_schema_extra={
            "group": "STRATEGY",
            "depends_on": "timestep_sampling:radc",
            "min": 0.0,
            "max": 0.5,
            "step": 0.05,
        },
    )

    # [NETWORK] LoRA Parameters
    network_rank: int = Field(
        16,
        description="Dimension of the LoRA network",
        json_schema_extra={"group": "NETWORK", "min": 1, "max": 256, "step": 1},
    )
    network_alpha: float = Field(
        8.0,
        description="Alpha scaling factor",
        json_schema_extra={"group": "NETWORK", "min": 0.1, "max": 256, "step": 0.5},
    )
    train_text_encoder: bool = Field(
        False,
        description="Train text encoder along with UNet",
        json_schema_extra={"group": "NETWORK"},
    )

    # [OPTIMIZER] Optimizer Settings
    optimizer_type: Literal[
        "AdamW",
        "AdamW8bit",
        "Prodigy",
        "ProdigyPlusSF",
        "SophiaH",
        "SophiaG",
        "Lion",
        "Adafactor",
        "StableAdamW",
        "Shampoo",
        "RAdam",
        "AdEMAMix",
    ] = Field(
        "AdamW8bit",
        description="Optimizer algorithm for weight updates",
        json_schema_extra={"group": "OPTIMIZER"},
    )
    learning_rate: float = Field(
        1e-4,
        description="Learning rate (Prodigy/PPSF recommend 1.0)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "min": 0,
            "max": 10,
            "step": 0.00001,
            "display": "scientific",
        },
    )
    weight_decay: float = Field(
        0.01,
        description="Weight decay for regularization",
        json_schema_extra={"group": "OPTIMIZER", "min": 0, "max": 1, "step": 0.001},
    )
    lr_scale_mode: Literal["none", "batch", "sqrt"] = Field(
        "none",
        description="Scale LR by effective batch size (batch×accum). 'batch'=linear, 'sqrt'=conservative",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:AdamW,AdamW8bit,SophiaH,SophiaG,Lion,Adafactor,StableAdamW,Shampoo,RAdam,AdEMAMix",
        },
    )
    beta1: float = Field(
        0.9,
        description="Adam beta1 (momentum)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:AdamW,AdamW8bit,SophiaH,SophiaG,Lion,Adafactor,StableAdamW,Shampoo,RAdam,AdEMAMix",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
    )
    beta2: float = Field(
        0.999,
        description="Adam beta2 (variance smoothing)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:AdamW,AdamW8bit,SophiaH,SophiaG,Lion,StableAdamW,Shampoo,RAdam,AdEMAMix",
            "min": 0.0,
            "max": 1.0,
            "step": 0.001,
        },
    )
    d_coef: float = Field(
        0.8,
        description="Prodigy adaptive LR coefficient",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:Prodigy",
            "min": 0.1,
            "max": 2.0,
            "step": 0.1,
        },
    )
    growth_rate: float = Field(
        1.02,
        description="Max growth factor for d-estimate per step (1.02 = safe warmup)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:Prodigy",
            "min": 1.0,
            "max": 2.0,
            "step": 0.01,
        },
    )
    decouple: bool = Field(
        True,
        description="Decoupled weight decay (AdamW-style)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:Prodigy",
        },
    )
    safeguard_warmup: bool = Field(
        True,
        description="Prevent early training instability",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:Prodigy",
        },
    )
    use_bias_correction: bool = Field(
        True,
        description="Enable bias correction for better convergence",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:Prodigy",
        },
    )
    lr_warmup_steps: int = Field(
        0,
        description="Warmup steps for scheduler",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:AdamW,AdamW8bit,SophiaH,SophiaG,Lion,Adafactor,StableAdamW,Shampoo,RAdam,AdEMAMix",
            "min": 0,
            "step": 10,
        },
    )
    lr_scheduler: Literal["constant", "cosine", "linear"] = Field(
        "constant",
        description="Learning rate scheduler",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:AdamW,AdamW8bit,SophiaH,SophiaG,Lion,Adafactor,StableAdamW,Shampoo,RAdam,AdEMAMix",
        },
    )

    # [OPTIMIZER] ProdigyPlusSF Core Settings
    ppsf_d_coef: float = Field(
        1.0,
        description="Prodigy d-estimate coefficient",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
            "min": 0.1,
            "max": 2.0,
            "step": 0.1,
        },
    )
    ppsf_prodigy_steps: int = Field(
        0,
        description="Steps to run Prodigy before switching to Adam (0 = always Prodigy)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
            "min": 0,
            "step": 100,
        },
    )
    ppsf_use_bias_correction: bool = Field(
        False,
        description="RAdam-style automatic warmup",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_stableadamw: bool = Field(
        True,
        description="StableAdamW gradient scaling (RMS-based)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_factored: bool = Field(
        True,
        description="Factored second moment — saves memory (Adafactor-like)",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_eps: float = Field(
        1e-8,
        description="Numerical stability term",
        json_schema_extra={
            "group": "OPTIMIZER",
            "depends_on": "optimizer_type:ProdigyPlusSF",
            "min": 1e-10,
            "max": 1e-4,
            "step": 1e-8,
        },
    )

    # [OPTIMIZER_EXPERT] ProdigyPlusSF Expert Features
    ppsf_use_cautious: bool = Field(
        False,
        description="Cautious updates — isolate values aligning with gradient",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_grams: bool = Field(
        False,
        description="Sign-based updates aligning with gradient",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_adopt: bool = Field(
        False,
        description="Partial ADOPT implementation (delayed moment update)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_orthograd: bool = Field(
        False,
        description="Use gradient component orthogonal to weights",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_focus: bool = Field(
        False,
        description="Noise handling at large step sizes",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_use_speed: bool = Field(
        False,
        description="Simplified momentum-based Prodigy estimate",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )
    ppsf_split_groups: bool = Field(
        True,
        description="Calculate d independently per parameter group",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:ProdigyPlusSF",
        },
    )

    # [OPTIMIZER_EXPERT] SophiaH Settings
    sophia_rho: float = Field(
        0.04,
        description="Hessian clipping threshold (rho)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaH,SophiaG",
            "min": 0.001,
            "max": 0.5,
            "step": 0.01,
        },
    )
    sophia_p: float = Field(
        0.01,
        description="Clip effective gradient (p)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaH",
            "min": 0.001,
            "max": 0.1,
            "step": 0.001,
        },
    )
    sophia_update_period: int = Field(
        10,
        description="Hessian update period",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaH",
            "min": 1,
            "step": 1,
        },
    )
    sophia_num_samples: int = Field(
        1,
        description="Times to sample z for Hessian trace",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaH",
            "min": 1,
            "max": 10,
            "step": 1,
        },
    )
    sophia_hessian_distribution: Literal["gaussian", "rademacher"] = Field(
        "gaussian",
        description="Distribution to initialize Hessian",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaH",
        },
    )

    # [OPTIMIZER_EXPERT] SophiaG Expert Settings
    sophia_maximize: bool = Field(
        False,
        description="Maximize objective instead of minimize",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaG",
        },
    )
    sophia_capturable: bool = Field(
        False,
        description="Enable CUDA graph capture (experimental, CUDA-only)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:SophiaG",
        },
    )

    # [OPTIMIZER_EXPERT] Adafactor Settings
    adafactor_relative_step: bool = Field(
        False,
        description="Scale LR by parameter magnitude (set LR to 1.0 when enabled)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:Adafactor",
            "title": "Relative Step",
        },
    )
    adafactor_warmup_init: bool = Field(
        False,
        description="Use warmup initialization (only allowed if relative_step=True and LR=None)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:Adafactor",
        },
    )
    adafactor_clip_threshold: float = Field(
        1.0,
        description="Clip threshold for root mean square of updates",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:Adafactor",
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
        },
    )
    adafactor_decay_rate: float = Field(
        -0.8,
        description="Coefficient to compute running averages of square",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:Adafactor",
            "min": -1.0,
            "max": -0.1,
            "step": 0.1,
        },
    )

    # [OPTIMIZER_EXPERT] RAdam / Shampoo / StableAdamW Settings
    radam_n_sma_threshold: int = Field(
        5,
        description="Length of SMA threshold (Rectified Adam)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:RAdam",
        },
    )
    shampoo_preconditioning_compute_steps: int = Field(
        1,
        description="Steps between preconditioning matrix updates",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:Shampoo",
        },
    )
    stableadamw_kahan_sum: bool = Field(
        False,
        description="Enable Kahan summation for high precision (StableAdamW)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:StableAdamW",
        },
    )

    # [OPTIMIZER_EXPERT] AdEMAMix Settings
    ademamix_beta3: float = Field(
        0.9999,
        description="AdEMAMix beta3 (slow momentum)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:AdEMAMix",
            "min": 0.9,
            "max": 1.0,
            "step": 0.0001,
        },
    )
    ademamix_alpha: float = Field(
        5.0,
        description="AdEMAMix alpha (mix factor)",
        json_schema_extra={
            "group": "OPTIMIZER_EXPERT",
            "depends_on": "optimizer_type:AdEMAMix",
            "min": 1.0,
            "max": 10.0,
            "step": 0.5,
        },
    )

    # [ENGINE] Advanced Features
    ema: bool = Field(
        False,
        description="Enable Exponential Moving Average",
        json_schema_extra={"group": "ENGINE"},
    )
    ema_decay: float = Field(
        0.999,
        description="EMA decay rate",
        json_schema_extra={
            "group": "ENGINE",
            "depends_on": "ema",
            "min": 0.9,
            "max": 1.0,
            "step": 0.001,
        },
    )
    noise_offset: float = Field(
        0.0,
        description="Noise offset for dynamic range",
        json_schema_extra={"group": "ENGINE", "min": 0.0, "max": 0.2, "step": 0.005},
    )
    min_snr_gamma: float = Field(
        5.0,
        description="Min-SNR Gamma weighting",
        json_schema_extra={"group": "ENGINE", "min": 0.0, "max": 20.0, "step": 0.5},
    )
    low_vram: bool = Field(
        True,
        description="Offload VAE to CPU after caching (disable to keep VAE on GPU for sampling speed)",
        json_schema_extra={"group": "ENGINE"},
    )
    offload_to_cpu: bool = Field(
        False,
        description="Offload model blocks to CPU to save VRAM",
        json_schema_extra={"group": "ENGINE"},
    )
    cache_text_embeddings: bool = Field(
        True,
        description="Cache text embeddings and offload text encoders to CPU (frees VRAM for training)",
        json_schema_extra={
            "group": "ENGINE",
            "help": "When enabled, all captions are encoded once and cached. Text encoders are then offloaded to CPU, freeing ~11GB VRAM. Enable 'Unload Text Encoder' to delete them entirely for maximum savings.",
        },
    )
    unload_text_encoder: bool = Field(
        False,
        description="Delete text encoders after caching (max VRAM savings, disables live prompt encoding for sampling)",
        json_schema_extra={"group": "ENGINE", "depends_on": "cache_text_embeddings"},
    )
    block_swap_config: dict[str, int] = Field(
        default_factory=dict,
        description="Per-block-group CPU offload percentage (0-100). Keys are block group names from model topology.",
        json_schema_extra={"group": "ENGINE", "ui_type": "block_swap_sliders"},
    )
    targeted_layers: list[str] = Field(
        default_factory=list,
        description="Layer names to train (empty = all trainable layers). Populated from model capabilities.",
        json_schema_extra={
            "group": "ENGINE",
            "ui_type": "layer_checklist",
            "hidden": True,
        },
    )

    # [VIDEO] Video-model training (optional; gated by family capability)
    #
    # All optional with sane defaults so image configs are unaffected — an
    # image family never sees these (capability-gated by ``is_video`` /
    # ``has_audio`` / ``dual_expert`` in ``core/archetypes.py``). The training
    # pipeline (B1) already reads ``num_frames`` / ``target_fps`` defensively.
    num_frames: int = Field(
        81,
        description="Max frames per clip (snapped to the family's frame rule at runtime)",
        json_schema_extra={"group": "VIDEO", "min": 1, "max": 257, "step": 1},
    )
    target_fps: float = Field(
        0,
        description="Training frame rate (0 = use the model's native fps)",
        json_schema_extra={"group": "VIDEO", "min": 0.0, "max": 60.0, "step": 1.0},
    )
    video_mode: Literal["t2v", "i2v"] = Field(
        "t2v",
        description="Text-to-video or image-to-video (first-frame conditioning)",
        json_schema_extra={"group": "VIDEO"},
    )
    i2v_image_dropout: float = Field(
        0.1,
        description="Chance of dropping the conditioning image (enables CFG for I2V)",
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "video_mode:i2v",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )
    train_audio: bool = Field(
        False,
        description="Jointly train the audio stream (audio-capable models only)",
        json_schema_extra={"group": "VIDEO"},
    )
    audio_loss_weight: float = Field(
        1.0,
        description="Relative weight of the audio loss term",
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "train_audio",
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
        },
    )
    expert_mode: Literal["both", "high", "low"] = Field(
        "both",
        description=(
            "Which WAN 2.2 experts to train: both (dual, default) or a single "
            "noise expert (high or low) — single-expert loads ONE transformer, "
            "halving VRAM (ai-toolkit style)"
        ),
        json_schema_extra={"group": "VIDEO"},
    )
    expert_swap_mode: Literal["auto", "swap", "resident"] = Field(
        "auto",
        description="Dual-expert placement: auto, swap (1 expert on GPU + pinned CPU) or resident (both on GPU)",
        json_schema_extra={"group": "VIDEO", "depends_on": "expert_mode:both"},
    )
    expert_switch_interval: int = Field(
        1,
        description="Steps between high/low expert swaps (swap mode only)",
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "expert_swap_mode:auto,swap",
            "min": 1,
            "max": 1000,
            "step": 1,
        },
    )
    boundary_ratio_override: float = Field(
        0,
        description="Override the MoE high/low timestep boundary (0 = use the definition default)",
        json_schema_extra={"group": "VIDEO", "min": 0.0, "max": 1.0, "step": 0.025},
    )

    # ── [VIDEO] Temporal sampling (Phase 1: Axis A tiled + Axis B stride) ──
    temporal_coverage: Literal["first", "tiled", "sliding"] = Field(
        "first",
        description=(
            "How the LoRA sees the whole clip: first (opening window only, "
            "default/backward-compatible), tiled (K windows per clip across "
            "epochs), sliding (Phase 2 — full-clip cache + random slice)"
        ),
        json_schema_extra={"group": "VIDEO"},
    )
    window_overlap: float = Field(
        0.0,
        description="Fractional overlap between tiled windows (0 = abutting)",
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "temporal_coverage:tiled",
            "min": 0.0,
            "max": 0.95,
            "step": 0.05,
        },
    )
    max_windows: int = Field(
        10,
        description="Upper bound on tiled windows emitted per clip",
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "temporal_coverage:tiled",
            "min": 1,
            "max": 999,
            "step": 1,
        },
    )
    frame_stride: int = Field(
        1,
        description=(
            "Sample every Nth frame so a window spans N× the motion at 1/N the "
            "effective fps (1 = native rate). The model is told the effective "
            "fps. Keep target_fps at 0/native when using stride."
        ),
        json_schema_extra={"group": "VIDEO", "min": 1, "max": 8, "step": 1},
    )
    sliding_max_clip_seconds: float = Field(
        0.0,
        description=(
            "Sliding mode: clips longer than this (seconds) fall back to tiled "
            "windows instead of one full-clip latent (0 = no limit; the frame "
            "ladder still caps the cached length)."
        ),
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "temporal_coverage:sliding",
            "min": 0.0,
            "step": 1.0,
        },
    )
    # ── Forward-compat (Phase 3): declared now so configs validate, inert until P3 ──
    still_resolutions: list[int] = Field(
        default=[],
        description=(
            "F=1 (stills) resolutions when mixing stills + video. Empty list "
            "means INHERIT from `resolutions` (the Phase-3 contract). Phase 3 — "
            "has no effect in Phase 1."
        ),
        json_schema_extra={"group": "VIDEO"},
    )
    radc_seqlen_influence: float = Field(
        0.0,
        description=(
            "RADC SNR-shift weight on total sequence length F×H×W (0 = off). "
            "Phase 3 — has no effect in Phase 1."
        ),
        json_schema_extra={
            "group": "VIDEO",
            "depends_on": "timestep_sampling:radc",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )

    # [SAMPLING] Sample Generation During Training
    sample_every_n_steps: int = Field(
        0,
        description="Generate samples every N steps (0 = disabled, independent from checkpoint save)",
        json_schema_extra={"group": "SAMPLING", "min": 0, "step": 50},
    )
    sample_skip_first_n_steps: int = Field(
        0,
        description="Skip sampling for the first N steps",
        json_schema_extra={
            "group": "SAMPLING",
            "depends_on": "sample_every_n_steps:!0",
            "min": 0,
            "step": 50,
        },
    )
    sample_prompts: list[SamplePromptConfig] = Field(
        default_factory=list,
        description="Prompts to generate at each sample interval. Supports [triggerword] and [captionprefix] wildcards.",
        json_schema_extra={
            "group": "SAMPLING",
            "depends_on": "sample_every_n_steps:!0",
        },
    )


class TrainingPlugin(ABC):
    """Pluggable training backend (v1 subprocess-based interface)."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def get_config_schema(self) -> type[BaseModel]:
        pass

    @abstractmethod
    def start_training(self, config: dict[str, any]) -> any:
        pass

    def enrich_schema(
        self, schema: dict[str, any], project_id: str | None = None
    ) -> dict[str, any]:
        """Optional: enrich the JSON schema with dynamic data (e.g. dataset names).

        ``project_id`` lets subclasses scope dynamic options to a project (e.g.
        the dataset dropdown). It is unused here (family/definition injection is
        global) but kept in the signature so the route can pass it uniformly.
        """
        from app.engine.factories.quantization import QuantizationFactory
        from app.engine.models.registry import registry

        # Inject model families and definitions for frontend dependent dropdowns
        registry.initialize()

        families = []
        definition_map = {}
        all_definitions = []
        all_definition_labels = []
        # definition_id -> control_inputs; lets the frontend dataset picker
        # require an edit (paired) dataset when an edit model is selected.
        edit_map = {}

        for model in registry._definitions.values():
            if model.family not in families:
                families.append(model.family)
            if model.family not in definition_map:
                definition_map[model.family] = []

            definition_map[model.family].append(model.id)
            all_definitions.append(model.id)
            all_definition_labels.append(f"{model.name} v{model.version}")
            edit_map[model.id] = int(getattr(model, "control_inputs", 0) or 0)

        if "properties" in schema:
            if "model_family" in schema["properties"]:
                schema["properties"]["model_family"]["enum"] = families
                schema["properties"]["model_family"]["enum_labels"] = [
                    f.upper() for f in families
                ]
                schema["properties"]["model_family"]["default"] = (
                    families[0] if families else ""
                )

            if "definition_id" in schema["properties"]:
                schema["properties"]["definition_id"]["enum"] = all_definitions
                schema["properties"]["definition_id"]["enum_labels"] = (
                    all_definition_labels
                )
                schema["properties"]["definition_id"]["backend_map"] = definition_map
                schema["properties"]["definition_id"]["edit_map"] = edit_map
                schema["properties"]["definition_id"]["default"] = (
                    all_definitions[0] if all_definitions else ""
                )

        # Inject backend-to-scheme map for frontend dependent dropdowns
        capabilities = QuantizationFactory.get_supported_capabilities()
        if "properties" in schema:
            if "quantization" in schema["properties"]:
                schema["properties"]["quantization"]["backend_map"] = capabilities
            if "te_quantization" in schema["properties"]:
                schema["properties"]["te_quantization"]["backend_map"] = capabilities

        return schema
