
import hashlib
import os
import torch

from safetensors.torch import save_file, load_file
import structlog

logger = structlog.get_logger(__name__)


class LatentManager:
    """
    Manages VAE encoding and disk caching of latents.
    Handles both Diffusers (AutoencoderKLOutput) and Flux (raw Tensor) VAE outputs.
    """
    def __init__(self, vae, device="cuda", cache_dir=None, arch_params: dict | None = None):
        self.vae = vae
        self.device = device
        self.cache_dir = cache_dir
        arch = arch_params or {}

        # Scaling factor priority: arch_params → vae attribute → vae.config → 1.0
        if "vae_scaling_factor" in arch:
            self.scaling_factor = arch["vae_scaling_factor"]
        elif hasattr(vae, "scaling_factor"):
            self.scaling_factor = vae.scaling_factor
        elif hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
            self.scaling_factor = vae.config.scaling_factor
        else:
            self.scaling_factor = 1.0

        # Spatial downscale: arch_params → isinstance fallback
        if "vae_downsample_factor" in arch:
            self.spatial_downscale = int(arch["vae_downsample_factor"])
        else:
            # Flux2 VAEs (AutoencoderKLFlux2) apply standard 8× spatial
            # downscale.  The 2×2 patchify step is separate (pack_latents).
            from diffusers import AutoencoderKLFlux2
            if isinstance(vae, AutoencoderKLFlux2):
                self.spatial_downscale = 8
            elif hasattr(vae, 'config') and getattr(vae.config, 'latent_channels', 0) == 16:
                # FLUX.1 AutoencoderKL: 16 latent channels but standard 8× spatial.
                # Packing (2×2 pixel-unshuffle) is separate from VAE encoding.
                self.spatial_downscale = 8
            else:
                self.spatial_downscale = 8

        logger.debug(
            "latent_manager_init",
            scaling_factor=self.scaling_factor,
            spatial_downscale=self.spatial_downscale,
            device=str(device),
        )

    def _is_video_vae(self) -> bool:
        """Check if the VAE expects 5D video input [B, C, F, H, W].

        QwenImage's ``AutoencoderKLQwenImage`` is a video-style VAE whose
        ``_encode`` method unpacks ``(_, _, num_frame, H, W) = x.shape``,
        requiring a frame dimension even for still images.
        """
        try:
            from diffusers.models import AutoencoderKLQwenImage
            return isinstance(self.vae, AutoencoderKLQwenImage)
        except ImportError:
            return False

    def _has_latent_norm_stats(self) -> bool:
        """Check if the VAE has per-channel latents_mean/latents_std config.

        Used by QwenImage and Wan VAEs for channel-wise normalization
        instead of a scalar scaling_factor.
        """
        cfg = getattr(self.vae, "config", None)
        return (
            cfg is not None
            and getattr(cfg, "latents_mean", None) is not None
            and getattr(cfg, "latents_std", None) is not None
        )

    @staticmethod
    def resolve_cache_dir(dataset_path: str, model_name: str, dataset_version: str, resolution: str, variant: str = "original") -> str:
        """
        Standardized resolution of latent cache directory.
        Path: {dataset_path}/.cache/{model_name}/{dataset_version}/latents/{variant}/{resolution}
        """
        return os.path.join(dataset_path, ".cache", model_name, dataset_version, "latents", variant, resolution)

    @staticmethod
    def hash_source_file(source_path: str) -> str:
        """Compute SHA-256 hex digest of a source image file.

        Used for content-addressed latent cache filenames so that
        replacing an image (same name, different pixels) invalidates
        the stale cache entry.
        """
        h = hashlib.sha256()
        with open(source_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def latent_filename(img_id: str, source_path: str) -> str:
        """Build a content-addressed latent cache filename.

        Format: ``{img_id}_{sha256_of_source_bytes}.safetensors``

        Args:
            img_id: Image identifier (typically the relative path stem).
            source_path: Absolute path to the source image file.

        Returns:
            Filename string (without directory).
        """
        file_hash = LatentManager.hash_source_file(source_path)
        # Use first 16 hex chars (64 bits) — collision-safe for <1M images
        return f"{img_id}_{file_hash[:16]}.safetensors"

    def encode_and_cache_batch(self, image_batch: torch.Tensor, ids: list[str], 
                               cache_dirs: list[str] | None = None,
                               mirror_dir: str | None = None,
                               source_paths: list[str] | None = None) -> torch.Tensor:
        """
        Encode a batch of images and cache results to disk.

        Args:
            image_batch: Batch of images [B, C, H, W] in [-1, 1] range.
            ids: List of image identifiers for cache filenames.
            cache_dirs: Per-item cache directories (overrides self.cache_dir).
            mirror_dir: If provided, also save a copy here.
            source_paths: Source image paths for content-addressed filenames.

        Returns:
            Encoded latent tensor [B, C_latent, H_latent, W_latent].
        """
        if self.vae is None:
            raise ValueError("VAE needed for encoding.")

        images = image_batch.to(self.device, dtype=self.vae.dtype)

        # QwenImage VAE expects 5D video tensor [B, C, num_frame, H, W].
        # For still images, add a frame dimension of 1.
        needs_5d = self._is_video_vae()
        if needs_5d and images.ndim == 4:
            images = images.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]

        with torch.no_grad():
            encoded = self.vae.encode(images)

            if isinstance(encoded, torch.Tensor):
                # Flux/BFL VAE returns a raw tensor, already scaled internally
                latents = encoded
            elif hasattr(encoded, "latent_dist"):
                # AutoencoderKLFlux2 uses BatchNorm normalization (applied
                # in the trainer after patchify), NOT a scalar scaling_factor.
                # Use mode() for deterministic encoding matching diffusers.
                from diffusers import AutoencoderKLFlux2
                if isinstance(self.vae, AutoencoderKLFlux2):
                    latents = encoded.latent_dist.mode()
                elif self._has_latent_norm_stats():
                    # QwenImage VAE: per-channel (z - mean) / std normalization
                    # using pretrained latents_mean and latents_std from config.
                    latents = encoded.latent_dist.mode()
                    mean = torch.tensor(
                        self.vae.config.latents_mean,
                        device=latents.device, dtype=latents.dtype,
                    )
                    std = torch.tensor(
                        self.vae.config.latents_std,
                        device=latents.device, dtype=latents.dtype,
                    )
                    # Reshape for broadcasting: [1, C, 1, 1]
                    mean = mean.view(1, -1, *([1] * (latents.ndim - 2)))
                    std = std.view(1, -1, *([1] * (latents.ndim - 2)))
                    latents = (latents - mean) / std
                else:
                    # Standard Diffusers AutoencoderKL uses sample() + scaling.
                    # Some VAEs (e.g. Z-Image) also have a shift_factor that
                    # must be subtracted before scaling:
                    #   z = scaling_factor * (sample - shift_factor)
                    # When shift_factor is 0/None this reduces to the original
                    # formula: z = sample * scaling_factor.
                    latents = encoded.latent_dist.sample()
                    shift = getattr(self.vae.config, "shift_factor", None) or 0.0
                    latents = self.scaling_factor * (latents - shift)
            else:
                latents = encoded

        # Remove the frame dimension if we added it
        if needs_5d and latents.ndim == 5:
            latents = latents.squeeze(2)  # [B, C, 1, H, W] → [B, C, H, W]

        # Validate latent shape
        self._validate_shape(latents, image_batch.shape)

        # Log stats at DEBUG level
        logger.debug(
            "latent_encoded",
            batch_size=latents.shape[0],
            latent_shape=list(latents.shape),
            mean=round(latents.mean().item(), 4),
            std=round(latents.std().item(), 4),
            dtype=str(latents.dtype),
        )

        # Cache to disk if configured
        if cache_dirs or self.cache_dir or mirror_dir:
            self._save_to_disk(latents, ids, cache_dirs, mirror_dir=mirror_dir, source_paths=source_paths)
            
        return latents

    def check_cache_coverage(
        self, ids: list[str], cache_dirs: list[str],
        source_paths: list[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        """Count cached vs missing latent files.

        Scans every ``(id, cache_dir)`` pair and checks for the
        corresponding ``.safetensors`` file on disk.

        When *source_paths* is provided, content-addressed filenames
        (``{id}_{hash}.safetensors``) are used.  Without source paths
        the legacy bare-name lookup (``{id}.safetensors``) is used as
        a fallback.

        Args:
            ids: Image identifiers (relative paths).
            cache_dirs: Per-item cache directories.
            source_paths: Absolute paths to the source images.

        Returns:
            ``(cached_count, missing_count, missing_ids)`` where
            *missing_ids* contains up to 50 sample IDs for logging.
        """
        cached = 0
        missing = 0
        missing_ids: list[str] = []
        seen: set[str] = set()

        for i, (img_id, c_dir) in enumerate(zip(ids, cache_dirs)):
            key = f"{c_dir}/{img_id}"
            if key in seen:
                continue
            seen.add(key)

            if source_paths:
                fname = self.latent_filename(img_id, source_paths[i])
            else:
                fname = f"{img_id}.safetensors"
            path = os.path.join(c_dir, fname)
            if os.path.exists(path):
                cached += 1
            else:
                missing += 1
                if len(missing_ids) < 50:
                    missing_ids.append(img_id)

        return cached, missing, missing_ids

    def load_cached_latents(
        self, ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
    ) -> torch.Tensor | None:
        """
        Try to load latents from disk. Returns None if any are missing.

        When *source_paths* is provided, uses content-addressed filenames.
        """
        if not cache_dirs and not self.cache_dir:
            return None
            
        loaded = []
        for i, img_id in enumerate(ids):
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if not c_dir:
                return None
            
            if source_paths:
                fname = self.latent_filename(img_id, source_paths[i])
            else:
                fname = f"{img_id}.safetensors"
            path = os.path.join(c_dir, fname)
            if not os.path.exists(path):
                return None
            
            try:
                data = load_file(path)
                loaded.append(data["latents"])
            except (OSError, KeyError) as e:
                logger.warning("latent_cache_load_failed", path=path, error=str(e))
                return None
            
        return torch.stack(loaded).to(self.device)

    def _validate_shape(self, latents: torch.Tensor, input_shape: torch.Size):
        """
        Validate latent shape is consistent with input image dimensions.
        Spatial downscale factor is set per-VAE (8× for standard, 16× for Flux2).
        """
        if latents.ndim != 4:
            logger.warning("latent_shape_unexpected", ndim=latents.ndim, expected=4)
            return

        b, c, h, w = latents.shape
        _, _, ih, iw = input_shape
        expected_h = ih // self.spatial_downscale
        expected_w = iw // self.spatial_downscale

        if h != expected_h or w != expected_w:
            logger.warning(
                "latent_spatial_mismatch",
                expected=f"{expected_h}x{expected_w}",
                actual=f"{h}x{w}",
                input=f"{ih}x{iw}",
            )

    def _save_to_disk(self, latents: torch.Tensor, ids: list[str], 
                      cache_dirs: list[str] | None = None, 
                      mirror_dir: str | None = None,
                      source_paths: list[str] | None = None):
        """Save encoded latents to safetensors files.

        When *source_paths* is provided, uses content-addressed filenames
        (``{id}_{hash}.safetensors``).  Otherwise falls back to bare names.
        """
        latents_cpu = latents.detach().cpu()
        for i, img_id in enumerate(ids):
            if source_paths:
                fname = self.latent_filename(img_id, source_paths[i])
            else:
                fname = f"{img_id}.safetensors"

            # 1. Primary Cache
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if c_dir:
                os.makedirs(c_dir, exist_ok=True)
                path = os.path.join(c_dir, fname)
                save_file({"latents": latents_cpu[i]}, path)
            
            # 2. Mirror (Output) Cache
            if mirror_dir:
                os.makedirs(mirror_dir, exist_ok=True)
                m_path = os.path.join(mirror_dir, fname)
                if not os.path.exists(m_path):
                    save_file({"latents": latents_cpu[i]}, m_path)
