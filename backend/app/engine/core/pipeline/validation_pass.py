"""One validation pass: the held-out images at fixed noise, and no trace in the run.

Numbers from different steps and different runs have to be comparable. So a pass uses
one fixed representative per source image, noise fixed by the split seed and the image,
the family's own hooks for everything else, and plain MSE: a family's loss weighting
can depend on progress, and would make two passes incomparable.
"""
import hashlib
import json
import math
import os
import random
import time
from typing import Any

import numpy as np
import torch

from app.engine.core.pipeline.hessian_estimate import _capture_random_state, _restore_random_state
from app.engine.core.pipeline.noise_bands import band_centres, timesteps_for_bands
from app.engine.core.training_adapter import adapter_off as training_adapter_off


def is_due(step: int, every: int, max_steps: int) -> bool:
    """At step 0, every ``every`` steps, and at the last step."""
    return step % every == 0 or step == max_steps - 1


def _image_seed(split_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{split_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def fixed_noise(split_seed: int, key: str, like: torch.Tensor) -> torch.Tensor:
    """The noise for one image: the same in every pass and every arm."""
    generator = torch.Generator(device="cpu").manual_seed(_image_seed(split_seed, key))
    return torch.randn(like.shape, generator=generator, dtype=like.dtype).to(like.device)


def _seed_the_streams(split_seed: int, key: str) -> None:
    """Fix whatever one image draws: without a latent cache, a VAE samples its posterior."""
    seed = _image_seed(split_seed, key)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)


def _clean_latents(trainer, batch: dict[str, Any]) -> torch.Tensor:
    manager = trainer.latent_manager
    use_cache = trainer.config.get("cache_latents", True)
    latents = None
    if use_cache:
        latents = manager.load_cached_latents(
            batch["ids"],
            batch["cache_dirs"],
            source_paths=batch["paths"],
            device=trainer.device,
            dtype=trainer.autocast_dtype,
        )
    if latents is None:
        latents = manager.encode_and_cache_batch(
            batch["images"],
            ids=batch["ids"],
            cache_dirs=batch["cache_dirs"] if use_cache else None,
            source_paths=batch["paths"],
        ).to(trainer.device, dtype=trainer.autocast_dtype)
    return latents


def _band_timesteps(trainer, count: int, prepared: torch.Tensor) -> torch.Tensor:
    """The family's timesteps for the bands, measured once per latent shape."""
    known = trainer.__dict__.setdefault("_validation_timesteps", {})
    shape = tuple(prepared.shape)
    if shape not in known:
        known[shape] = timesteps_for_bands(trainer, count, like=prepared.float()[:1])
    return known[shape]


def _image_band_losses(trainer, representative: dict[str, Any], bands: int, split_seed: int) -> list[float]:
    # The original variant, with its caption: no masked draw and no caption dropout.
    _seed_the_streams(split_seed, representative["key"])
    item = {**representative, "has_masked": False, "dropout_rate": 0.0}
    batch = trainer._get_batch([item])
    latents = _clean_latents(trainer, batch)
    trainer._attach_conditioning(batch, latents)
    text_emb = trainer.encode_text(batch["captions"], trainer.autocast_dtype, batch=batch)

    noise = fixed_noise(split_seed, representative["key"], latents)
    prepared_latents = trainer.prepare_latents_for_training(latents)
    prepared_noise = trainer.prepare_noise_for_training(noise)
    timesteps = _band_timesteps(trainer, bands, prepared_latents).to(trainer.device)

    losses = []
    for band in range(bands):
        timestep = timesteps[band].reshape(1)
        with torch.autocast("cuda", dtype=trainer.autocast_dtype, enabled=trainer.use_amp):
            noisy_input = trainer.add_noise(prepared_latents, prepared_noise, timestep)
            prediction = trainer.forward_pass(noisy_input, timestep, text_emb, batch)
            target = trainer.compute_target(prepared_latents, prepared_noise, timestep)
        losses.append(torch.nn.functional.mse_loss(prediction.float(), target.float()).item())
    return losses


def _record(step: int, per_image: list[list[float]], bands: int, seconds: float) -> dict[str, Any]:
    band_loss = [sum(losses[b] for losses in per_image) / len(per_image) for b in range(bands)]
    finite = all(math.isfinite(loss) for loss in band_loss)
    return {
        "step": step,
        "images": len(per_image),
        "band_centres": band_centres(bands).tolist(),
        "band_loss": [loss if math.isfinite(loss) else None for loss in band_loss],
        "mean": sum(band_loss) / bands if finite else None,
        "worst_band": band_loss.index(max(band_loss)) if finite else None,
        "seconds": round(seconds, 3),
    }


def run_validation_pass(trainer, split, step: int) -> dict[str, Any]:
    """Measure every representative at every band; report and return the numbers."""
    started = time.perf_counter()
    bands = int(trainer.config.get("validation_timestep_bands", 8))
    split_seed = int(trainer.config.get("validation_split_seed", 0))

    model = trainer._get_primary_model()
    modes = [(module, module.training) for module in model.modules()]
    random_state = _capture_random_state()
    # The weights a sample would use: a schedule-free optimizer's evaluation weights, and EMA's,
    # without the training adapter.
    optimizer = getattr(trainer, "optimizer", None)
    ema = getattr(trainer, "ema_handler", None)
    optimizer_switched = swapped = False
    try:
        model.eval()
        if optimizer is not None and hasattr(optimizer, "eval"):
            optimizer.eval()
            optimizer_switched = True
        if ema:
            ema.store_and_swap()
            swapped = True
        # A training adapter is off here too, and back on in the context's own finally.
        with torch.no_grad(), training_adapter_off(model):
            per_image = [
                _image_band_losses(trainer, representative, bands, split_seed)
                for representative in split.representatives
            ]
    finally:
        if swapped:
            ema.restore()
        if optimizer_switched and hasattr(optimizer, "train"):
            optimizer.train()
        for module, training in modes:
            module.training = training
        _restore_random_state(random_state)

    record = _record(step, per_image, bands, time.perf_counter() - started)
    path = os.path.join(trainer.checkpoint_manager.output_dir, "validation_metrics.jsonl")
    with open(path, "a", encoding="utf-8") as metrics:
        metrics.write(json.dumps(record) + "\n")
    trainer.logger.info("validation_pass", step=step, mean=record["mean"], seconds=record["seconds"])
    log_writer = getattr(trainer, "_log_writer", None)
    if log_writer is not None:
        log_writer.emit("validation", record)
    return record
