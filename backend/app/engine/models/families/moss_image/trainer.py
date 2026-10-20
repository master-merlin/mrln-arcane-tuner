"""MOSS-Image trainer -- family hooks for the generic training pipeline (LANE-165).

- Text pre-cache (the flux3 flow, MOSS's recipe): ``_pre_cache_text_embeddings`` encodes every
  caption hint the shared ``_build_caption_hints()`` returns (composites, dropout variants, the empty
  string, the sample prompts, the sample negative prompt) one caption at a time BEFORE the text
  encoder is offloaded, so the ~16.8 GB Qwen3.5 encoder is not resident during training. Each entry is
  the caption's unpadded ``[S, 4096]`` final-layer state in the encoder's own dtype (every token valid,
  so the mask is its length). The disk directory carries the driver's ``te_cache_tag()`` (template
  flags + token cap) and the filename hashes the caption: a change to any of the three misses.
- Sigmas: ``sigma = s*u / (1 + (s-1)*u)``, ``u = sigmoid(z)``, ``z ~ N(0, 1)``, ``s`` = the YAML's
  ``scheduler.training_shift`` (3.0). Evidence: research Decision D5 (kohya
  ``docs/moss_image_train_network.md`` §7.2). Assumption: the bounded approximation of the
  unconfirmed pre-training schedule (spec risk 2) -- the GPU UAT side-by-side judges it. The run
  config's ``timestep_sampling`` is not consulted (no run-config key for this family).
- Timesteps ``sigma * 1000`` (the pipeline scale; the driver passes them through unchanged).
- Noisy input ``(1 - sigma) * x0 + sigma * noise`` and target ``noise - x0`` -- fixed here, never the
  configurable ``noise_interpolation`` mode.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput

logger = structlog.get_logger(__name__)

# Research Decision D5 / the YAML ``scheduler.training_shift``.
DEFAULT_TRAINING_SHIFT = 3.0
# The disk filename's fixed source part: the entry's identity is (caption, template flags, token cap) --
# the caption hashed in the filename, the other two in the directory's ``te_cache_tag()`` -- never the
# image that happened to carry the caption first (LANE-165 verify 1.02).
_TE_CACHE_HINT = "moss"


def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """``s*u / (1 + (s-1)*u)`` on ``u`` in ``[0, 1]``; ``s = 1`` returns ``u`` unchanged."""
    return shift * u / (1.0 + (shift - 1.0) * u)


class MossImageTrainer(GenericTrainingPipeline):
    """MOSS-Image (Base / SFT) LoRA trainer."""

    # -- Setup --

    def _setup_family(self) -> None:
        from .driver import MossImageDriver  # noqa: PLC0415
        from .loader import MossImageLoader  # noqa: PLC0415

        self.driver = MossImageDriver(self.definition, self.device)
        self.loader = MossImageLoader(self.device)

    def _create_sampler(self):
        """Create a MossImageSampler if sampling is configured."""
        if int(self.config.get("sample_every_n_steps", 0)) > 0:
            from .sampler import MossImageSampler  # noqa: PLC0415

            return MossImageSampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer, the components dict and the driver in sync after PEFT wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # -- Text-embedding pre-cache --

    def _te_cache_dir_for(self, base_dir: str) -> str:
        """``{dataset}/.cache/{model}/{version}/embeddings/{te_quant}/te1/{te_cache_tag}``."""
        te_quant = self.config.get("te_quantization", "none")
        return os.path.join(base_dir, "embeddings", te_quant, "te1", self.driver.te_cache_tag())

    def _precache_hints(self) -> dict[str, str]:
        hints = dict(self._build_caption_hints())
        if self.config.get("sample_prompts"):
            negative = str(self.config.get("sample_negative_prompt", "") or "")
            hints.setdefault(negative, "sample_negative")
        return hints

    def _with_inventory(self, items: list[dict], fn):
        """Run ``fn()`` with ``self.inventory`` narrowed to ``items`` (restored in ``finally``)."""
        full = self.inventory
        self.inventory = items
        try:
            return fn()
        finally:
            self.inventory = full

    def _te_cache_owners(self) -> dict[str, list[str]]:
        """Each pre-cache caption -> the TE cache dirs of every dataset that needs it.

        The inventory is grouped by its OWNING dataset cache dir (the shared ``_resolve_te_cache_dirs``
        on a one-item inventory), and each group's hints are what that dataset trained alone needs:
        its composites and dropout variants plus the empty string, the sample prompts and the negative
        prompt. So a dataset trained alone later finds every entry it needs under its own dir. Captions
        with no dataset dir (no inventory, or caching off on disk) map to ``[]``: memory only.
        """
        groups: dict[str, list[dict]] = {}
        for item in self.inventory:
            dirs = self._with_inventory([item], self._resolve_te_cache_dirs)
            groups.setdefault(dirs[0] if dirs else "", []).append(item)
        if not groups:
            return {caption: [] for caption in self._precache_hints()}
        owners: dict[str, list[str]] = {}
        for base_dir, items in groups.items():
            te_dir = self._te_cache_dir_for(base_dir) if base_dir else ""
            for caption in self._with_inventory(items, self._precache_hints):
                dirs = owners.setdefault(caption, [])
                if te_dir and te_dir not in dirs:
                    dirs.append(te_dir)
        return owners

    def _pre_cache_text_embeddings(self) -> None:
        """Encode every caption (memory, then any owner's disk, then the encoder) before the TE offload.

        The disk key is the caption's identity alone (``_TE_CACHE_HINT``, never the image that first
        carried it); the template flags and token cap are in the directory. An entry found under one
        owning dataset is copied to the owners that lack it, an encoded one is written to all of them.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None or self.driver is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        owners = self._te_cache_owners()
        disk_loaded = 0
        need_encode: list[str] = []
        for caption, dirs in owners.items():
            tensor = self.text_cache.get(caption)
            missing = list(dirs)
            for te_dir in dirs:
                found = TextEmbeddingCache.load(caption, te_dir, _TE_CACHE_HINT)
                if found is not None and found.ndim == 2:
                    missing.remove(te_dir)
                    if tensor is None:
                        tensor = found
                        disk_loaded += 1
            if tensor is None:
                need_encode.append(caption)
                continue
            self.text_cache[caption] = tensor
            for te_dir in missing:
                TextEmbeddingCache.save(caption, tensor, te_dir, _TE_CACHE_HINT)

        self.logger.info("te_disk_cache_status", total=len(owners), from_disk=disk_loaded,
                         need_encode=len(need_encode), cache_tag=self.driver.te_cache_tag(),
                         dataset_dirs=len({d for dirs in owners.values() for d in dirs}))
        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        total = len(need_encode)
        for i, caption in enumerate(need_encode):
            row = self.driver.encode_caption(caption).detach().cpu()
            self.text_cache[caption] = row
            for te_dir in owners[caption]:
                TextEmbeddingCache.save(caption, row, te_dir, _TE_CACHE_HINT)
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Caching Text Embeddings ({round((i + 1) / total * 100)}%)")
        self.logger.info("text_embedding_cache_complete", cached_embeddings=len(self.text_cache),
                         newly_encoded=total)

    def _offload_text_encoders(self) -> None:
        """Base TE offload/unload; when unloading, drop the tokenizer too (the flux3 behaviour)."""
        unloading = self.config.get("unload_text_encoder", False)
        super()._offload_text_encoders()
        if unloading and not hasattr(self, "_tok_cleaned"):
            self.tokenizer = None
            self.components.pop("tokenizer", None)
            self._tok_cleaned = True

    # -- Text encoding --

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> TextEncoderOutput:
        """Embeddings + bool mask; served from the cache when caching is on (``batch`` is ignored)."""
        from .driver import pad_rows  # noqa: PLC0415

        if not self.config.get("cache_text_embeddings", True):
            return self.driver.encode_text(list(captions), dtype)
        self._encode_misses([cap for cap in dict.fromkeys(captions) if cap not in self.text_cache])
        return pad_rows([self.text_cache[cap] for cap in captions], dtype, self.device)

    def _encode_misses(self, uncached: list[str]) -> None:
        """A miss is encoded with the encoder brought back (flux3's offload-miss path) and cached."""
        if not uncached:
            return
        if self.driver.text_encoder is None:
            raise RuntimeError(
                "moss_image: the text encoder was unloaded but these captions were never pre-cached: "
                + ", ".join(repr(cap[:50]) for cap in uncached)
            )
        te = self.driver.text_encoder
        te_device = next(te.parameters()).device
        te_was_offloaded = te_device != torch.device(self.device)
        if te_was_offloaded:
            self.logger.warning("te_cache_miss_after_offload", count=len(uncached),
                                hint="pre-caching should have covered all captions")
            te.to(self.device)
        try:
            for cap in uncached:
                self.text_cache[cap] = self.driver.encode_caption(cap).detach().cpu()
        finally:
            if te_was_offloaded:
                te.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -- Sigmas, timesteps, noise, target --

    def _training_shift(self) -> float:
        arch = getattr(getattr(self, "definition", None), "architecture_params", None) or {}
        return float(arch.get("scheduler.training_shift", DEFAULT_TRAINING_SHIFT))

    def sample_sigmas(self, batch_size: int) -> torch.Tensor:
        """``shift_sigma(sigmoid(z), s)``, ``z ~ N(0, 1)`` per sample, in ``(0, 1)``."""
        u = torch.sigmoid(torch.randn(batch_size)).to(self.device)
        return shift_sigma(u, self._training_shift())

    def sample_timesteps(self, batch_size: int, latents: torch.Tensor | None = None) -> torch.Tensor:
        """``sigma * 1000``: the pipeline scale the driver hands to the transformer unchanged."""
        return self.sample_sigmas(batch_size) * 1000.0

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """``(1 - sigma) * x0 + sigma * noise`` with ``sigma = t / 1000`` (sigma 1 = pure noise)."""
        sigma = (timesteps.float() / 1000.0).view(-1, *([1] * (latents.ndim - 1)))
        sigma = sigma.to(device=latents.device, dtype=latents.dtype)
        return (1.0 - sigma) * latents + sigma * noise

    def compute_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """The flow-matching velocity ``noise - x0``."""
        return noise - latents
