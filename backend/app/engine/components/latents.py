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

    def __init__(
        self,
        vae,
        device: str | torch.device | None = None,
        cache_dir=None,
        arch_params: dict | None = None,
    ):
        self.vae = vae
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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

        # Spatial downscale: arch_params → VAE's own ratio → isinstance fallback
        if "vae_downsample_factor" in arch:
            self.spatial_downscale = int(arch["vae_downsample_factor"])
        elif getattr(vae, "spatial_compression_ratio", None):
            # Video VAEs (LTX-2 32×, WAN, QwenImage, …) expose their true spatial
            # compression directly; trust it over the 8× image-VAE default so the
            # latent-shape validator doesn't false-alarm on every sample.
            self.spatial_downscale = int(vae.spatial_compression_ratio)
        else:
            # Flux2 VAEs (AutoencoderKLFlux2) apply standard 8× spatial
            # downscale.  The 2×2 patchify step is separate (pack_latents).
            from diffusers import AutoencoderKLFlux2

            if isinstance(vae, AutoencoderKLFlux2):
                self.spatial_downscale = 8
            elif (
                hasattr(vae, "config")
                and getattr(vae.config, "latent_channels", 0) == 16
            ):
                # FLUX.1 AutoencoderKL: 16 latent channels but standard 8× spatial.
                # Packing (2×2 pixel-unshuffle) is separate from VAE encoding.
                self.spatial_downscale = 8
            else:
                self.spatial_downscale = 8

        # Temporal downscale (video VAEs): arch_params override, else inferred
        # lazily in temporal_downscale() from the VAE class. None → infer.
        td = arch.get("vae_temporal_downsample")
        self._temporal_downscale = int(td) if td else None

        logger.debug(
            "latent_manager_init",
            scaling_factor=self.scaling_factor,
            spatial_downscale=self.spatial_downscale,
            device=str(device),
        )

    # VAEs that consume a 5D [B, C, F, H, W] tensor. Matched by class name so
    # importing the heavy classes is unnecessary at call time.
    _VIDEO_VAE_NAMES = (
        "AutoencoderKLQwenImage",
        "AutoencoderKLWan",
        "AutoencoderKLLTX2Video",
        "AutoencoderKLLTXVideo",
        "AutoencoderKLHunyuanVideo",
    )

    def _is_video_vae(self) -> bool:
        """Check if the VAE expects 5D video input [B, C, F, H, W].

        QwenImage's ``AutoencoderKLQwenImage`` is a video-style VAE whose
        ``_encode`` method unpacks ``(_, _, num_frame, H, W) = x.shape``,
        requiring a frame dimension even for still images. WAN
        (``AutoencoderKLWan``) and LTX (``AutoencoderKLLTX2Video`` /
        ``AutoencoderKLLTXVideo``) are true temporal-compression video VAEs.

        Matched by class name (walking the MRO) so this stays cheap and does
        not import the heavy diffusers classes.
        """
        for cls in type(self.vae).__mro__:
            if cls.__name__ in self._VIDEO_VAE_NAMES:
                return True
        return False

    def temporal_downscale(self) -> int:
        """Temporal compression factor of the VAE (frames → latent frames).

        The encoder maps ``F`` input frames to ``(F - 1) / t + 1`` latent
        frames. ``t`` is read from ``arch_params['vae_temporal_downsample']``
        when present, else inferred from the VAE class:

        - WAN: ``4``  (latent_f = (F-1)/4 + 1)
        - HunyuanVideo (Kandinsky 5.0): ``4``
        - LTX: ``8``  (latent_f = (F-1)/8 + 1)
        - QwenImage / anything else: ``1`` (no temporal compression; the
          frame dim is preserved 1:1, matching the still-image unsqueeze path).
        """
        if self._temporal_downscale is not None:
            return self._temporal_downscale
        name = type(self.vae).__name__
        # Walk the MRO so subclasses still match.
        names = {c.__name__ for c in type(self.vae).__mro__}
        if "AutoencoderKLWan" in names or "AutoencoderKLHunyuanVideo" in names:
            return 4
        if "AutoencoderKLLTX2Video" in names or "AutoencoderKLLTXVideo" in names:
            return 8
        logger.debug("temporal_downscale_default", vae=name)
        return 1

    @staticmethod
    def latent_frames(num_frames: int, temporal_downscale: int) -> int:
        """Latent frame count: ``(F - 1) / t + 1`` (floor), min 1."""
        if temporal_downscale <= 1:
            return max(int(num_frames), 1)
        return max((int(num_frames) - 1) // int(temporal_downscale) + 1, 1)

    @staticmethod
    def slice_latent_window(
        latents: torch.Tensor, window_frames: int, start: int
    ) -> torch.Tensor:
        """Slice ``window_frames`` contiguous latent frames starting at ``start``.

        The temporal (frame) axis is the third-from-last: dim 1 for a per-item
        ``[C, f, h, w]`` latent and dim 2 for a batched ``[B, C, f, h, w]``.
        Used by sliding-mode training to cut a per-step window out of the cached
        full-clip latent (Option A: treated as a 0-based clip — no RoPE offset).
        """
        frame_axis = latents.ndim - 3
        idx = [slice(None)] * latents.ndim
        idx[frame_axis] = slice(int(start), int(start) + int(window_frames))
        return latents[tuple(idx)]

    def _has_latent_norm_stats(self) -> bool:
        """Check if the VAE uses per-channel latents_mean/latents_std normalization.

        Used by QwenImage / Wan (stats on ``vae.config``) AND LTX-2, whose
        ``AutoencoderKLLTX2Video`` registers ``latents_mean``/``latents_std`` as
        persistent BUFFERS (``vae.latents_mean``), not config entries — checking
        config alone missed them, so LTX-2 latents were cached UN-normalized
        (std≈0.15) while the model works in normalized space → decoded samples
        were pure noise.
        """
        return LatentManager._resolve_norm_stats(self.vae) is not None

    @staticmethod
    def _resolve_norm_stats(vae):
        """Return ``(latents_mean, latents_std)`` for *vae*, or ``None``.

        Prefers ``vae.config`` (Wan / QwenImage) then falls back to module
        buffers ``vae.latents_mean`` / ``vae.latents_std`` (LTX-2).
        """
        cfg = getattr(vae, "config", None)
        mean = getattr(cfg, "latents_mean", None) if cfg is not None else None
        std = getattr(cfg, "latents_std", None) if cfg is not None else None
        if mean is None or std is None:
            mean = getattr(vae, "latents_mean", None)
            std = getattr(vae, "latents_std", None)
        if mean is None or std is None:
            return None
        return mean, std

    @staticmethod
    def _norm_view(latents: torch.Tensor, stat) -> torch.Tensor:
        """Channel-broadcast a 1-D stat to ``latents`` rank ([1, C, 1, …])."""
        t = torch.as_tensor(stat, device=latents.device, dtype=latents.dtype)
        return t.view(1, -1, *([1] * (latents.ndim - 2)))

    @staticmethod
    def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
        """``(z - mean) / std`` (scaling_factor folded in upstream; LTX-2 sf=1).

        No-op for VAEs without latents_mean/std (image VAEs) — returns *latents*.
        """
        stats = LatentManager._resolve_norm_stats(vae)
        if stats is None:
            return latents
        mean, std = stats
        return (latents - LatentManager._norm_view(latents, mean)) / LatentManager._norm_view(latents, std)

    @staticmethod
    def denormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
        """Inverse of :meth:`normalize_latents` — ``z * std + mean``.

        Samplers MUST call this before ``vae.decode``: the model emits latents in
        the normalized space, but the VAE decoder expects raw-scale latents.
        No-op for VAEs without latents_mean/std.
        """
        stats = LatentManager._resolve_norm_stats(vae)
        if stats is None:
            return latents
        mean, std = stats
        return latents * LatentManager._norm_view(latents, std) + LatentManager._norm_view(latents, mean)

    @staticmethod
    def resolve_cache_dir(
        dataset_path: str,
        model_name: str,
        dataset_version: str,
        resolution: str,
        variant: str = "original",
    ) -> str:
        """
        Standardized resolution of latent cache directory.
        Path: {dataset_path}/.cache/{model_name}/{dataset_version}/latents/{variant}/{resolution}
        """
        return os.path.join(
            dataset_path,
            ".cache",
            model_name,
            dataset_version,
            "latents",
            variant,
            resolution,
        )

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
    def latent_filename(img_id: str, source_path: str, extra_key: str = "") -> str:
        """Build a content-addressed latent cache filename.

        Format: ``{img_id}_{hash16}.safetensors`` where ``hash16`` is the first
        16 hex chars of ``sha256(source_bytes [+ extra_key])``.

        ``extra_key`` folds extra cache-identity (e.g. a video clip's
        ``t{start}-{end}`` trim window) into the content hash so two clips of
        the same file with different trims don't collide.

        REGRESSION CONTRACT: when ``extra_key`` is empty the digest is computed
        from the source bytes ALONE — byte-identical to the pre-video
        behavior, so image cache filenames are unchanged.

        Args:
            img_id: Image identifier (typically the relative path stem).
            source_path: Absolute path to the source media file.
            extra_key: Optional discriminator folded into the hash.

        Returns:
            Filename string (without directory).
        """
        if extra_key:
            h = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            h.update(extra_key.encode("utf-8"))
            file_hash = h.hexdigest()
        else:
            # Empty extra_key → identical to legacy path (source bytes only).
            file_hash = LatentManager.hash_source_file(source_path)
        # Use first 16 hex chars (64 bits) — collision-safe for <1M images
        return f"{img_id}_{file_hash[:16]}.safetensors"

    def encode_and_cache_batch(
        self,
        image_batch: torch.Tensor,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        mirror_dir: str | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Encode a batch of images and cache results to disk.

        Args:
            image_batch: Batch of images [B, C, H, W] (or 5D [B, C, F, H, W]
                for video) in [-1, 1] range.
            ids: List of image identifiers for cache filenames.
            cache_dirs: Per-item cache directories (overrides self.cache_dir).
            mirror_dir: If provided, also save a copy here.
            source_paths: Source media paths for content-addressed filenames.
            extra_keys: Per-item discriminators folded into the cache-file hash
                (e.g. ``t{start}-{end}`` for videos). Empty/absent → legacy
                image filenames (byte-identical to pre-video behavior).

        Returns:
            Encoded latent tensor [B, C_latent, H_latent, W_latent] (or 5D).
        """
        if self.vae is None:
            raise ValueError("VAE needed for encoding.")

        # Co-locate the VAE with the compute device. After pre-caching the VAE is
        # offloaded to CPU (low_vram), but encode can still be called afterwards —
        # a train-loop cache MISS routes here with the input already on CUDA. Move
        # the VAE just-in-time (a no-op when already resident) so its weights meet
        # the input's device; otherwise: "Input type (CUDABFloat16Type) and weight
        # type (CPUBFloat16Type) should be the same". Mirrors the audio-VAE /
        # connector JIT co-location used elsewhere for offloaded modules.
        self.vae.to(self.device)

        images = image_batch.to(self.device, dtype=self.vae.dtype)

        # Distinguish two 5D regimes:
        #  - a STILL image fed to a video-style VAE → we unsqueeze a dummy
        #    frame dim (F=1) on the way in and squeeze it back out, so the
        #    cached latent stays 4D and image-family behavior is unchanged.
        #  - a genuine VIDEO clip (input already 5D [B, C, F, H, W]) → keep the
        #    temporal dim and persist a 5D latent.
        is_real_video = image_batch.ndim == 5
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
                    # Per-channel (z - mean) / std normalization using the VAE's
                    # pretrained latents_mean/latents_std (QwenImage/Wan: config;
                    # LTX-2: module buffers). Shared with the sampler's decode
                    # denormalization so the two can't drift.
                    latents = LatentManager.normalize_latents(
                        encoded.latent_dist.mode(), self.vae
                    )
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
            elif hasattr(encoded, "latents"):
                # AutoencoderTiny (dreamlite) has NO latent_dist — encode
                # returns AutoencoderTinyOutput(.latents). Mirrors diffusers'
                # ``retrieve_latents`` helper; the scalar scaling formula is
                # shared with the AutoencoderKL branch above so the sampler's
                # decode (z / scale + shift) stays its exact inverse.
                # (Tiny ships scale 1.0 / shift 0.0 → identity in practice.)
                shift = getattr(self.vae.config, "shift_factor", None) or 0.0
                latents = self.scaling_factor * (encoded.latents - shift)
            else:
                latents = encoded

        # Remove the dummy frame dimension ONLY for stills (we added it). A
        # genuine video clip keeps its temporal axis → persist a 5D latent.
        if needs_5d and latents.ndim == 5 and not is_real_video:
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
            self._save_to_disk(
                latents,
                ids,
                cache_dirs,
                mirror_dir=mirror_dir,
                source_paths=source_paths,
                extra_keys=extra_keys,
            )

        return latents

    @staticmethod
    def _fname_for(
        i: int,
        img_id: str,
        source_paths: list[str] | None,
        extra_keys: list[str] | None,
    ) -> str:
        """Resolve the cache filename for item ``i`` (content-addressed or legacy)."""
        if source_paths:
            ek = extra_keys[i] if extra_keys else ""
            return LatentManager.latent_filename(img_id, source_paths[i], ek)
        return f"{img_id}.safetensors"

    def check_cache_coverage(
        self,
        ids: list[str],
        cache_dirs: list[str],
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
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

            fname = self._fname_for(i, img_id, source_paths, extra_keys)
            path = os.path.join(c_dir, fname)
            if os.path.exists(path):
                cached += 1
            else:
                missing += 1
                if len(missing_ids) < 50:
                    missing_ids.append(img_id)

        return cached, missing, missing_ids

    def load_cached_latents(
        self,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> torch.Tensor | None:
        """
        Try to load latents from disk. Returns None if any are missing.

        When *source_paths* is provided, uses content-addressed filenames.
        ``extra_keys`` (per-item) fold a discriminator (e.g. a video trim
        window) into the filename hash.
        """
        if not cache_dirs and not self.cache_dir:
            return None

        loaded = []
        for i, img_id in enumerate(ids):
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if not c_dir:
                return None

            fname = self._fname_for(i, img_id, source_paths, extra_keys)
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

    def load_cached_latent_windows(
        self,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
        window_frames: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None:
        """Load each FULL-CLIP latent, slice a random ``window_frames`` window, stack.

        Sliding mode (Option A): the cache holds one full-length latent per clip
        (keyed by a ``slideF{N}`` discriminator). Per step we cut an independent,
        uniform-length window from each item's full latent so the batch stacks —
        the homogeneous ``(w,h,frames)`` grouping guarantees one ``window_frames``
        per batch. Returns ``None`` on ANY miss (same contract as
        :meth:`load_cached_latents`) so the caller falls back to a direct encode.
        """
        if not cache_dirs and not self.cache_dir:
            return None

        windows: list[torch.Tensor] = []
        for i, img_id in enumerate(ids):
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if not c_dir:
                return None
            fname = self._fname_for(i, img_id, source_paths, extra_keys)
            path = os.path.join(c_dir, fname)
            if not os.path.exists(path):
                return None
            try:
                full = load_file(path)["latents"]  # [C, f, h, w]
            except (OSError, KeyError) as e:
                logger.warning("latent_cache_load_failed", path=path, error=str(e))
                return None

            frame_axis = full.ndim - 3
            f = int(full.shape[frame_axis])
            max_start = max(f - int(window_frames), 0)
            if max_start > 0:
                start = int(
                    torch.randint(0, max_start + 1, (1,), generator=generator).item()
                )
            else:
                start = 0
            windows.append(self.slice_latent_window(full, int(window_frames), start))

        return torch.stack(windows).to(self.device)

    def _validate_shape(self, latents: torch.Tensor, input_shape: torch.Size):
        """
        Validate latent shape is consistent with input image dimensions.
        Spatial downscale factor is set per-VAE (8× for standard, 16× for Flux2).

        For genuine 5D video latents ``[B, C, f, h, w]`` the temporal axis is
        validated against ``latent_f = (F - 1) / temporal_downscale + 1`` in
        addition to the spatial dims.
        """
        if latents.ndim == 5 and len(input_shape) == 5:
            self._validate_shape_5d(latents, input_shape)
            return

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

    def _validate_shape_5d(self, latents: torch.Tensor, input_shape: torch.Size):
        """Validate a 5D video latent ``[B, C, f, h, w]`` against the input."""
        _, _, lf, h, w = latents.shape
        _, _, in_f, ih, iw = input_shape
        td = self.temporal_downscale()
        expected_f = self.latent_frames(in_f, td)
        expected_h = ih // self.spatial_downscale
        expected_w = iw // self.spatial_downscale

        if lf != expected_f or h != expected_h or w != expected_w:
            logger.warning(
                "latent_video_shape_mismatch",
                expected=f"{expected_f}x{expected_h}x{expected_w}",
                actual=f"{lf}x{h}x{w}",
                input=f"{in_f}x{ih}x{iw}",
                temporal_downscale=td,
            )

    def _save_to_disk(
        self,
        latents: torch.Tensor,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        mirror_dir: str | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ):
        """Save encoded latents to safetensors files.

        When *source_paths* is provided, uses content-addressed filenames
        (``{id}_{hash}.safetensors``).  Otherwise falls back to bare names.
        ``extra_keys`` (per-item) fold a discriminator into the filename hash.
        """
        latents_cpu = latents.detach().cpu()
        for i, img_id in enumerate(ids):
            fname = self._fname_for(i, img_id, source_paths, extra_keys)

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
