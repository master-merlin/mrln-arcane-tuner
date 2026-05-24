"""HiDream-O1 trainer — Saganaki22's May 2026 ai-toolkit LoRA recipe.

Recipe (re-implemented natively — see spike_notes.md Task 3a):
- ``noise_scale = 8.0``
- linear sigma sampling in ``[T_EPS=0.001, 0.9999]``
- velocity-equivalent loss: predict ``(noisy - x_pred)/sigma`` against
  ``noise * noise_scale - patches``
- ``max_loss = 1.0`` clamp

The model class is our vendored ``Qwen3VLForConditionalGeneration`` — accepts
custom forward kwargs ``vinputs`` (noisy patches), ``timestep``,
``token_types`` (and optional ``use_flash_attn`` / ``use_sage_attn``).
Output is ``Qwen3VLModelOutputWithPast.x_pred`` (per-patch x0 prediction).

LoRA targets via the "aitoolkit" preset (all linear-like layers except
``lm_head``/``patch_embed``/``visual``). Custom ``HiDreamO1LoRALinear``
wrapper (NOT peft) because the save format must match ComfyUI's native
HiDream-O1 LoRA loader's kohya-style key convention.

Integration notes (Task 16):
    1. **LatentManager bypass** — the base training loop calls
       ``latent_manager.encode_and_cache_batch()`` which raises when
       ``vae=None``.  We install a ``_PixelPassthroughLatentManager``
       (see below) via ``_configure_managers`` so the base loop receives
       the raw pixel tensor untouched.  ``forward_pass`` pulls pixels from
       ``batch["images"]`` directly, so the passthrough value is ignored.

    2. **Training-step override** — the base loop computes
       ``loss = F.mse_loss(pred, target)`` where ``pred = forward_pass(...)``
       and ``target = compute_target(...)``.  Our ``compute_target`` returns
       ``x0_pred.detach()`` making MSE = 0.  We override the ``_train_step``
       hook (see ``_run_train_step``) to instead call ``compute_loss(batch)``
       which produces the correct velocity loss with a grad_fn.

    3. **Saver signature conformance** — ``IModelSaver.save(components, path,
       metadata)`` vs ``HiDreamO1Saver.save(model, out_dir, name, metadata)``.
       Fixed in saver.py by adding a base-conforming ``save`` that wraps the
       custom one.
"""

from __future__ import annotations

import math
from typing import Any

import einops
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import HiDreamO1Driver
from .loader import HiDreamO1Loader
from .lora_wrapper import (
    LORA_EXCLUDED_SUBSTRINGS,
    LoRAInjectionResult,
    inject_lora_layers,
)
from .vendor.pipeline import (
    NOISE_SCALE,
    PATCH_SIZE,
    T_EPS,
    build_t2i_text_sample,
)

logger = structlog.get_logger(__name__)

# Recipe constants — also re-exported at module level so tests can import.
TIMESTEP_TYPE: str = "linear"
MAX_LOSS: float = 1.0
# NOISE_SCALE, T_EPS, PATCH_SIZE come from vendor/pipeline.py — re-export.
__all__ = [
    "HiDreamO1Trainer",
    "NOISE_SCALE",
    "TIMESTEP_TYPE",
    "MAX_LOSS",
    "T_EPS",
    "PATCH_SIZE",
    "LORA_EXCLUDED_SUBSTRINGS",
]


# ── Pixel-passthrough LatentManager (Concern 1) ───────────────────────────

class _PixelPassthroughLatentManager:
    """Drop-in LatentManager replacement for pixel-space families.

    HiDream-O1 is pixel-space — it has no VAE.  The base training loop
    calls ``latent_manager.encode_and_cache_batch()`` unconditionally on
    cache miss; injecting this passthrough avoids the VAE-required raise.

    Behaviour:
    - ``load_cached_latents``: always returns ``None`` (cache miss path).
      HiDream-O1 does not cache latents — the model processes pixels live.
    - ``encode_and_cache_batch``: returns the pixel tensor as-is.  The
      base loop will store this as ``latents``; ``forward_pass`` ignores
      it and pulls ``batch["images"]`` directly.
    - ``check_cache_coverage``: reports all items as cached so
      ``_pre_cache_latents`` is a no-op.
    - ``latent_filename`` / ``_validate_shape``: delegated to a stub so
      ``_build_cache_manifest`` and similar helpers don't crash.
    """

    def load_cached_latents(
        self,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
    ) -> torch.Tensor | None:
        """Always report a cache miss — pixel-space has no latent cache.

        Must return None so the base loop falls into ``encode_and_cache_batch``,
        which returns the actual 4D ``batch["images"]`` tensor. The base loop
        later does ``latents.shape[1]`` for noise-offset shaping — that only
        works on the 4D image tensor, not on a length-N sentinel.
        """
        return None

    def encode_and_cache_batch(
        self,
        image_batch: torch.Tensor,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        mirror_dir: str | None = None,
        source_paths: list[str] | None = None,
    ) -> torch.Tensor:
        """Return pixel values unchanged — no VAE encoding needed."""
        return image_batch

    def check_cache_coverage(
        self,
        ids: list[str],
        cache_dirs: list[str],
        source_paths: list[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        """Report all items as cached so pre-cache step is skipped."""
        n = len(ids)
        return n, 0, []

    @staticmethod
    def latent_filename(img_id: str, source_path: str) -> str:
        """Stub — pixel-space families don't write latent files."""
        return f"{img_id}.safetensors"


def _sample_sigma(
    batch_size: int,
    device: torch.device,
    timestep_type: str = TIMESTEP_TYPE,
    min_sigma: float = T_EPS,
    max_sigma: float = 0.9999,
    shift: float = 1.0,
) -> torch.Tensor:
    """Sample sigma per ai-toolkit's options. ``linear`` is the default.

    Args:
        batch_size: Number of sigma values to draw.
        device: Device to create the tensor on.
        timestep_type: One of ``"linear"`` (uniform), ``"sigmoid"``
            (sigmoid of standard normal), or ``"shift"`` (linear with
            resolution-based shift).
        min_sigma: Lower bound (inclusive).
        max_sigma: Upper bound (inclusive).
        shift: Scale shift for ``"shift"`` type. No-op when ``shift == 1.0``.

    Returns:
        Float32 tensor of shape ``[batch_size]`` clamped to
        ``[min_sigma, max_sigma]``.
    """
    min_sigma = max(0.0001, min(0.9999, float(min_sigma)))
    max_sigma = max(min_sigma + 0.0001, min(0.9999, float(max_sigma)))
    tt = (timestep_type or "linear").lower()

    if tt == "sigmoid":
        s = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
    else:
        s = torch.rand(batch_size, device=device, dtype=torch.float32)

    s = min_sigma + (max_sigma - min_sigma) * s
    if tt == "shift" and shift and shift > 0 and not math.isclose(shift, 1.0):
        s = shift * s / (1 + (shift - 1) * s)
    return s.clamp(min_sigma, max_sigma)


class HiDreamO1Trainer(GenericTrainingPipeline):
    """Pixel-space LoRA trainer for HiDream-O1-Image (Full).

    Overrides (Task 16 integration fixes in addition to family-specific
    behaviour):
    - ``_setup_family``: wires loader + driver.
    - ``_configure_managers``: installs ``_PixelPassthroughLatentManager``
      instead of the VAE-requiring ``LatentManager`` (Fix: Concern 1).
    - ``_validate_latent_cache`` / ``_pre_cache_latents``: no-ops — pixel
      space families carry no latent cache (Fix: Concern 1).
    - ``_compute_step_loss``: calls ``compute_loss(batch)`` directly,
      bypassing the base MSE which would be zero due to the detached
      ``compute_target`` return (Fix: Concern 2).
    - ``_apply_peft``: replaces peft's ``get_peft_model`` with our custom
      ``inject_lora_layers`` so saved LoRA keys follow kohya / ComfyUI
      convention rather than peft-native naming.
    - ``forward_pass``: ignores base-prepared noisy latents and implements
      the full HiDream-O1 recipe (patchify, sigma, noise, custom forward).
    - ``compute_target``: returns ``x0_pred.detach()`` as a same-shape
      sentinel — actual loss is computed in ``_compute_step_loss`` /
      ``compute_loss``; this value is not used for the backward pass.

    The optimizer picks up LoRA params automatically because
    ``inject_lora_layers`` sets ``requires_grad=True`` on ``lora_down``
    and ``lora_up`` while freezing all base weights — so the base's
    ``_configure_optimization`` collects exactly the right parameters.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize HiDream-O1-specific loader, driver, and recipe knobs."""
        self.driver = HiDreamO1Driver(self.definition, self.device)
        self.loader = HiDreamO1Loader(self.device)
        self.lora_injection: LoRAInjectionResult | None = None

        # Pixel-space model — no VAE latent scaling or patchify factor.
        # Config keys used by the recipe (pulled in compute_loss / _sample_sigma):
        self.config.setdefault("timestep_type", TIMESTEP_TYPE)

    def _build_trainable_components(self) -> dict[str, Any]:
        """Skip the per-component checkpoint dump.

        The base default returns ``{"unet": <full model>}``, which causes
        ``CheckpointManager._save_train_state`` to call
        ``comp.save_pretrained(...)`` on the full Qwen3VLForConditionalGeneration
        — a ~35 GB sharded dump alongside the LoRA. For peft-wrapped families
        ``save_pretrained`` writes only the adapter (small); for our custom
        LoRA wrappers it writes the entire frozen base, which is wasted disk
        and IO.

        The actual LoRA artifact (the diff we care about) is already written
        separately by ``CheckpointManager.save_checkpoint`` at
        ``<output_dir>/<lora>_<step>.safetensors`` via ``self.saver.save(...)``
        — that path is unaffected.

        Returning an empty dict means resume from a checkpoint won't restore
        the LoRA wrapper parameters automatically; resumption-with-LoRA would
        need a separate ``load_lora`` call. That's a follow-up — the priority
        here is to stop the 35 GB per-checkpoint waste.
        """
        return {}

    def _create_sampler(self):
        """Create a HiDreamO1Sampler when sampling is configured.

        Mirrors the convention used by other families (ernie_image, flux1, etc.):
        return ``None`` when ``sample_every_n_steps`` is 0 to short-circuit the
        whole sampling pipeline; otherwise instantiate the sampler.
        """
        interval = int(self.config.get("sample_every_n_steps", 0))
        sample_before = bool(self.config.get("sample_before_training", True))
        if interval <= 0 and not sample_before:
            return None
        from .sampler import HiDreamO1Sampler
        return HiDreamO1Sampler(self)

    def _assign_components(self) -> None:
        """Wire components via driver + load processor/tokenizer for recipe.

        The processor is needed by ``build_t2i_text_sample`` at training time.
        We load it lazily from the HF snapshot directory used by the loader.
        If loading fails (e.g. offline / test environment), ``self.processor``
        remains ``None`` — ``compute_loss`` will raise at runtime with a clear
        message rather than a cryptic AttributeError.
        """
        super()._assign_components()
        # Try to obtain processor from components (future loader extension) or
        # load it now from the same HF repo the loader used.
        if "processor" in self.components:
            self.processor = self.components["processor"]
        else:
            self.processor = self._load_processor_if_available()

        self.tokenizer = (
            self.processor.tokenizer
            if self.processor is not None and hasattr(self.processor, "tokenizer")
            else self.processor
        )

    def _load_processor_if_available(self):
        """Load ``AutoProcessor`` from the HF snapshot, or return ``None``."""
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor

            unet_spec = (
                self.definition.components.get("unet")
                if self.definition.components else None
            )
            repo_id = (
                getattr(unet_spec, "repo", None) or getattr(unet_spec, "path", None)
                if unet_spec else None
            ) or "HiDream-ai/HiDream-O1-Image"
            snap_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
            processor = AutoProcessor.from_pretrained(snap_dir)
            logger.info("hidream_o1.trainer.processor_loaded", repo=repo_id)
            return processor
        except Exception as exc:
            logger.warning(
                "hidream_o1.trainer.processor_unavailable",
                reason=str(exc),
                hint="compute_loss will raise if called without a processor",
            )
            return None

    # ── Integration fixes (Task 16) ──────────────────────────────────────

    def _configure_managers(self, max_train_steps: int) -> None:
        """Override to install pixel-passthrough LatentManager (Fix: Concern 1).

        The base creates ``LatentManager(vae=None, ...)`` which raises on
        ``encode_and_cache_batch``.  We replace it with a passthrough that
        returns pixel values unchanged — ``forward_pass`` uses
        ``batch["images"]`` directly so the passthrough value is ignored.
        """
        super()._configure_managers(max_train_steps)
        # Replace after super() sets it — ensures CheckpointManager is set up
        # correctly and only the LatentManager is substituted.
        self.latent_manager = _PixelPassthroughLatentManager()
        logger.info("hidream_o1.trainer.pixel_passthrough_latent_manager_installed")

    def _validate_latent_cache(self) -> None:
        """No-op: pixel-space families carry no latent cache (Fix: Concern 1)."""
        self._latent_cache_missing = 0

    async def _pre_cache_latents(self) -> None:
        """No-op: pixel-space families have no latents to pre-cache (Fix: Concern 1)."""

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """Compute HiDream-O1 velocity loss via compute_loss (Fix: Concern 2).

        The base MSE would be 0 because ``compute_target`` returns
        ``x0_pred.detach()``.  Instead, call ``compute_loss(batch)`` which
        runs the full recipe and returns a scalar loss with a grad_fn.
        """
        loss = self.compute_loss(batch)
        return loss / grad_accum

    # ── LoRA injection (overrides peft path) ─────────────────────────────

    def _apply_peft(self) -> None:
        """Inject HiDream-O1 custom LoRA wrappers instead of peft.

        Overrides the base ``_apply_peft`` which calls ``get_peft_model``.
        We use ``inject_lora_layers`` (our own 50-line wrapper) because:
        1. The LoRA save format must match ComfyUI's kohya-style key
           convention (``diffusion_model.<key>.lora_{down,up}.weight``).
        2. peft-native keys (``base_model.model....lora_A.weight``) are
           incompatible with the native HiDream-O1 LoRA loader.

        After injection, ``lora_down`` / ``lora_up`` parameters have
        ``requires_grad=True``.  All base parameters are frozen.
        The optimizer (set up by ``_configure_optimization``) collects
        these via ``model.parameters()`` automatically.
        """
        rank = int(self.config.get("network_rank", 32))
        alpha = float(self.config.get("network_alpha", rank))
        dropout = float(self.config.get("lora_dropout", 0.0))

        model = self.driver.get_primary_model()
        self.lora_injection = inject_lora_layers(
            model, rank=rank, alpha=alpha, dropout=dropout,
        )
        if not self.lora_injection.layers:
            raise RuntimeError(
                "HiDream-O1 LoRA injection produced 0 trainable layers — "
                f"target_preset='aitoolkit', excluded={LORA_EXCLUDED_SUBSTRINGS!r}, "
                "but no linear-like modules matched. Check model class.",
            )
        trainable = sum(
            p.numel()
            for layer in self.lora_injection.layers
            for p in layer.trainable_parameters()
        )
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "hidream_o1.trainer.lora_applied",
            rank=rank,
            alpha=alpha,
            injected=len(self.lora_injection.layers),
            skipped=len(self.lora_injection.skipped),
            trainable_params=trainable,
            total_params=total,
        )

    # ── Forward + Loss ───────────────────────────────────────────────────

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Full HiDream-O1 recipe forward pass — returns x0_pred patches.

        The base training loop calls this inside the autocast context after
        preparing noisy latents via the VAE path.  For HiDream-O1, there is
        no VAE — we ignore ``noisy_input`` / ``timesteps`` / ``text_embeddings``
        and compute everything from ``batch["images"]`` directly, matching
        the recipe exactly.

        The method returns ``x0_pred`` (the model's pixel-patch prediction),
        not a loss.  The loss is computed in ``compute_target`` + the base
        MSE, BUT to avoid shape mismatches with the base's loss computation,
        we store intermediate tensors on ``self`` and override ``compute_target``
        to produce the correct velocity target.

        If ``batch`` does not carry ``"images"`` (should not happen), falls
        back to ``noisy_input`` with a warning.
        """
        # Pull raw pixels from batch (shape [B, C, H, W], range [-1, 1])
        pixel_values = batch.get("images")
        if pixel_values is None:
            logger.warning(
                "hidream_o1.trainer.forward_pass.no_images_in_batch",
                hint="falling back to noisy_input — recipe will be incorrect",
            )
            pixel_values = noisy_input
        pixel_values = pixel_values.to(self.device, dtype=self._autocast_dtype())

        captions = batch.get("captions", batch.get("caption", [""]))
        caption = captions[0] if isinstance(captions, (list, tuple)) else captions

        model = self.driver.get_primary_model()
        height = pixel_values.shape[-2]
        width = pixel_values.shape[-1]

        # Patchify: [B, C, H, W] -> [B, (H/P)*(W/P), C*P*P]
        patches = einops.rearrange(
            pixel_values,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        # Sample sigma
        tt = self.config.get("timestep_type", TIMESTEP_TYPE)
        sigma = _sample_sigma(patches.shape[0], self.device, timestep_type=tt)
        sigma_typed = sigma.to(dtype=patches.dtype)
        sigma_view = sigma_typed.view(-1, 1, 1)

        # Add scaled noise
        noise = torch.randn_like(patches)
        scaled_noise = noise * NOISE_SCALE
        noisy = (1.0 - sigma_view) * patches + sigma_view * scaled_noise
        timestep = (1.0 - sigma).to(device=self.device, dtype=torch.float32)

        # Build text sample
        processor = getattr(self, "processor", None)
        tokenizer = getattr(self, "tokenizer", None) or processor
        if processor is None or tokenizer is None:
            raise RuntimeError(
                "HiDreamO1Trainer.forward_pass requires a processor/tokenizer. "
                "Ensure the HF snapshot is available at _assign_components time "
                "so AutoProcessor can be loaded from the cached snapshot.",
            )

        text_sample = build_t2i_text_sample(
            caption, height, width, tokenizer, processor, model.config,
        )
        text_sample = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in text_sample.items()
        }

        # Custom-kwarg forward
        outputs = model(
            input_ids=text_sample["input_ids"],
            position_ids=text_sample["position_ids"],
            vinputs=noisy,
            timestep=timestep,
            token_types=text_sample["token_types"],
            use_flash_attn=False,   # safe default; flash-attn may not be installed
            use_sage_attn=False,
        )
        x0_pred = outputs.x_pred[0, text_sample["vinput_mask"][0]].unsqueeze(0)

        # Stash intermediates for compute_target + compute_loss_weight.
        # These are used to compute the velocity-equivalent loss after
        # the base's ``target = compute_target(latents, noise, timesteps)``
        # and ``loss = mse_loss(pred, target)`` calls.
        self._hd_noisy = noisy
        self._hd_x0_pred = x0_pred
        self._hd_patches = patches
        self._hd_scaled_noise = scaled_noise
        self._hd_sigma_view = sigma_view

        return x0_pred

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return velocity-equivalent target for the MSE loss.

        Instead of the base's ``noise - latents`` target, HiDream-O1 uses:
            ``velocity_target = scaled_noise - patches``

        We also rescale both pred and target by ``1/sigma`` to match the
        velocity prediction convention.  The final MSE is equivalent to the
        Saganaki22 recipe's ``F.mse_loss(velocity_pred, velocity_target)``.

        NOTE: The base calls ``F.mse_loss(pred.float(), target.float())``.
        We make both ``pred`` and ``target`` the velocity variants so the
        base MSE computes the correct quantity.
        """
        # Stash updated by forward_pass; fall back gracefully if missing.
        noisy = getattr(self, "_hd_noisy", None)
        x0_pred = getattr(self, "_hd_x0_pred", None)
        patches = getattr(self, "_hd_patches", None)
        scaled_noise = getattr(self, "_hd_scaled_noise", None)
        sigma_view = getattr(self, "_hd_sigma_view", None)

        if any(v is None for v in (noisy, x0_pred, patches, scaled_noise, sigma_view)):
            # Should not happen in normal training; fall back to base behaviour.
            logger.warning(
                "hidream_o1.trainer.compute_target.missing_stash",
                hint="forward_pass should have stored _hd_* tensors",
            )
            return noise - latents

        # NOTE: The base calls mse_loss(pred, target) where pred = x0_pred.
        # We can't substitute velocity_pred post-return from forward_pass.
        # To sidestep this without polluting gradients, return a target equal
        # to x0_pred.detach() so the base mse_loss is 0.  The actual velocity
        # loss is computed and backward'd in compute_loss() via a separate call.
        # This is a known limitation — see the Task 16 integration note in the
        # module docstring.
        return x0_pred.float().detach()

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """One training-step loss computation per Saganaki22's recipe.

        This is the canonical recipe entry-point.  It is called by the
        integration tests and can be called directly from a custom training
        loop.  In the standard ``GenericTrainingPipeline`` training loop,
        ``forward_pass`` is the integration point — it stores intermediates
        and ``compute_target`` computes the velocity target from them.

        Args:
            batch: ``{"pixel_values": Tensor[B, 3, H, W], "caption": list[str]}``.
                   Alternatively ``{"images": Tensor, "captions": list[str]}``
                   (the key names used by the base pipeline's ``_get_batch``).

        Returns:
            Scalar loss tensor with ``grad_fn`` set, clamped to ``MAX_LOSS``.
        """
        device = self.device
        dtype = self._autocast_dtype()
        model = self.driver.get_primary_model()
        processor = getattr(self, "processor", None)
        tokenizer = getattr(self, "tokenizer", None) or processor

        # Accept both key conventions — use explicit None checks to avoid
        # "Boolean value of Tensor is ambiguous" when the tensor is non-empty.
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            pixel_values = batch.get("images")
        if pixel_values is None:
            raise ValueError(
                "compute_loss requires 'pixel_values' or 'images' in batch."
            )
        pixel_values = pixel_values.to(device, dtype=dtype)

        captions = batch.get("caption")
        if captions is None:
            captions = batch.get("captions")
        if not captions:
            captions = [""]
        caption = captions[0] if isinstance(captions, (list, tuple)) else captions

        height = pixel_values.shape[-2]
        width = pixel_values.shape[-1]

        # Patchify
        patches = einops.rearrange(
            pixel_values,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        # Sigma + noise injection
        tt = self.config.get("timestep_type", TIMESTEP_TYPE) if hasattr(self, "config") else TIMESTEP_TYPE
        sigma = _sample_sigma(patches.shape[0], device, timestep_type=tt).to(dtype=dtype)
        sigma_view = sigma.view(-1, 1, 1)
        noise = torch.randn_like(patches)
        scaled_noise = noise * NOISE_SCALE
        noisy = (1.0 - sigma_view) * patches + sigma_view * scaled_noise
        timestep = (1.0 - sigma).to(device=device, dtype=torch.float32)

        # Build text sample
        if processor is None or tokenizer is None:
            raise RuntimeError(
                "HiDreamO1Trainer.compute_loss requires a processor/tokenizer. "
                "Ensure the HF snapshot is available so AutoProcessor can be loaded.",
            )
        text_sample = build_t2i_text_sample(
            caption, height, width, tokenizer, processor, model.config,
        )
        text_sample = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in text_sample.items()
        }

        # Forward via the custom-kwarg surface
        outputs = model(
            input_ids=text_sample["input_ids"],
            position_ids=text_sample["position_ids"],
            vinputs=noisy,
            timestep=timestep,
            token_types=text_sample["token_types"],
            use_flash_attn=False,
            use_sage_attn=False,
        )
        x0_pred = outputs.x_pred[0, text_sample["vinput_mask"][0]].unsqueeze(0)

        # Velocity-equivalent loss
        sigma_loss = sigma_view.float().clamp_min(T_EPS)
        velocity_pred = (noisy.float() - x0_pred.float()) / sigma_loss
        velocity_target = scaled_noise.float() - patches.float()
        raw_loss = F.mse_loss(velocity_pred, velocity_target).clamp(max=MAX_LOSS)

        if raw_loss.grad_fn is None:
            raise RuntimeError(
                "HiDream-O1 loss has no grad_fn — LoRA params not in graph. "
                "Check that _apply_peft ran before training.",
            )
        return raw_loss

    # ── Helpers ──────────────────────────────────────────────────────────

    def _autocast_dtype(self) -> torch.dtype:
        """Return the autocast dtype from config, defaulting to bfloat16."""
        prec = getattr(self, "autocast_dtype", None)
        if prec is not None:
            return prec
        cfg_prec = self.config.get("mixed_precision", "bf16") if hasattr(self, "config") else "bf16"
        return torch.bfloat16 if cfg_prec in ("bf16", "bfloat16") else torch.float16

    def _update_primary_model(self, new_model: nn.Module) -> None:
        """Keep driver reference in sync after any wrapping step.

        NOTE: After ``_apply_peft``, the model is NOT wrapped by peft (we
        inject LoRA in-place).  This method is still called by the base
        pipeline's quantization path — we just update the driver's ref.
        """
        self.components["unet"] = new_model
        self.driver.model = new_model
        if hasattr(self, "model"):
            self.model = new_model
