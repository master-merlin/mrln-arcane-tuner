
import os
import threading

import numpy as np
import structlog
from PIL import Image

from app.core.dataset.media_types import (
    IMAGE_EXTENSION_PREFERENCE,
    IMAGE_EXTENSIONS,
)
from app.core.gpu_unload import gpu_batch_active, unload_gpu_plugins
from app.core.masking.models import (
    MaskingModel,
    RemBGModel,
    SAM3Model
)

logger = structlog.get_logger(__name__)

# Task types that own the masking GPU plugins for their duration, so a global
# unload must not rip a model out from under them.
#
# `mask_generate_batch` is here because its worker calls
# `service.generate_mask(...)` per item (mask_generate_batch.py:103), which
# loads the plugin.
#
# `mask_apply_batch` is deliberately ABSENT: it wraps `MaskingService.mass_apply`
# (mask_apply_batch.py:47), which composites already-written mask files onto the
# images and never touches `self.plugins` — grepping the whole `mass_apply` body
# for `plugin`/`load`/`generate_mask` returns nothing. It therefore holds no GPU
# model and cannot be interrupted by an unload.
_MASKING_GUARD_TASK_TYPES = ("mask_generate_batch",)

class MaskingService:
    """
    Singleton service for AI-powered image masking.
    
    Manages model loading, caching, and mask generation using a plugin
    architecture.
    """
    _instance = None
    _active_model_id = None
    # Guards unload_models(skip_if_batch_active=True)'s check-then-act, so a
    # mask batch cannot start between "no batch is running" and the unload.
    # Same shape and same reason as CaptionService._unload_lock.
    _unload_lock = threading.Lock()

    def __init__(self):
        # Register available model plugins
        self.plugins: dict[str, MaskingModel] = {
            "rembg": RemBGModel(self),
            "sam3": SAM3Model(self),
        }

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = MaskingService()
        return cls._instance

    def generate_mask(self, image_path: str, model_id: str, params: dict) -> Image.Image:
        """
        Generates a mask for the given image using the specified model plugin.
        Returns a PIL Image (L mode - greyscale).
        """
        if model_id not in self.plugins:
            raise ValueError(f"Unknown masking model: {model_id}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            input_image = Image.open(image_path).convert("RGB")
        except Exception as e:
             raise ValueError(f"Failed to open image: {e}")

        plugin = self.plugins[model_id]

        # Handle model switching/loading
        if self.__class__._active_model_id and self.__class__._active_model_id != model_id:
            self.unload_models()
        
        if self.__class__._active_model_id != model_id:
            plugin.load()
            self.__class__._active_model_id = model_id

        # Generate mask
        mask = plugin.generate(input_image, params)
        
        # Return as L mode (Greyscale)
        if mask:
            return mask.convert("L")
        
        return Image.new("L", input_image.size, 0)

    def combine_mask(self, image_path: str, mask_path: str, output_path: str, opacity: float = 0.0):
        """
        Applies the mask to the image and saves it.
        If output is .jpg, composites onto a black background respecting background opacity.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with Image.open(image_path).convert("RGB") as img:
            with Image.open(mask_path).convert("L") as mask:
                is_jpg = output_path.lower().endswith((".jpg", ".jpeg"))
                
                # Logic: Result Alpha = Mask + (1 - Mask) * opacity
                mask_np = np.array(mask).astype(float) / 255.0
                alpha_np = (mask_np + (1.0 - mask_np) * opacity) * 255.0
                alpha_np = np.clip(alpha_np, 0, 255).astype(np.uint8)
                alpha_mask = Image.fromarray(alpha_np)

                if is_jpg:
                    # For JPEG, blend with black background using the calculated alpha
                    black_bg = Image.new("RGB", img.size, (0, 0, 0))
                    # Composite img onto black_bg using alpha_mask
                    result = Image.composite(img, black_bg, alpha_mask)
                    result.save(output_path, "JPEG", quality=95)
                else:
                    # For PNG, keep the calculated alpha channel
                    rgba_img = img.convert("RGBA")
                    rgba_img.putalpha(alpha_mask)
                    rgba_img.save(output_path, "PNG")

    def generate_preview(self, image_path: str, mask_path: str, opacity: float) -> Image.Image:
        """
        Creates a preview image where the subject is 100% opaque and the 
        background has the specified opacity. Returns RGBA PIL Image.
        """
        with Image.open(image_path).convert("RGB") as img:
            with Image.open(mask_path).convert("L") as mask:
                # Logic: Result Alpha = Mask + (1 - Mask) * opacity
                mask_np = np.array(mask).astype(float) / 255.0
                alpha_np = (mask_np + (1.0 - mask_np) * opacity) * 255.0
                alpha_np = np.clip(alpha_np, 0, 255).astype(np.uint8)
                
                rgba_img = img.convert("RGBA")
                rgba_img.putalpha(Image.fromarray(alpha_np))
                return rgba_img

    def mass_apply(
        self,
        dataset_path: str,
        opacity: float,
        overwrite: bool,
        progress_callback=None,
    ) -> dict:
        """
        Apply masks to all images that have a corresponding mask file.
        Returns {applied: int, skipped: int, missing_masks: list[str]}.
        """
        masks_dir = os.path.join(dataset_path, "masks")
        masked_dir = os.path.join(dataset_path, "masked")
        os.makedirs(masked_dir, exist_ok=True)

        if not os.path.isdir(masks_dir):
            return {"applied": 0, "skipped": 0, "missing_masks": []}

        # Collect all mask files
        mask_files = [
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(".png")
        ]

        # Find which images are missing masks
        all_images = [
            f for f in os.listdir(dataset_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        mask_stems = {os.path.splitext(f)[0] for f in mask_files}
        missing_masks = [
            f for f in all_images
            if os.path.splitext(f)[0] not in mask_stems
        ]

        applied = 0
        skipped = 0
        total = len(mask_files)

        for i, mask_file in enumerate(mask_files):
            stem = os.path.splitext(mask_file)[0]
            mask_path = os.path.join(masks_dir, mask_file)

            # Find matching source image
            source_path = None
            for ext in IMAGE_EXTENSION_PREFERENCE:
                candidate = os.path.join(dataset_path, f"{stem}{ext}")
                if os.path.exists(candidate):
                    source_path = candidate
                    break

            if source_path is None:
                skipped += 1
                continue

            output_path = os.path.join(masked_dir, f"{stem}.jpg")

            if not overwrite and os.path.exists(output_path):
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, stem)
                continue

            try:
                self.combine_mask(source_path, mask_path, output_path, opacity)
                applied += 1
            except Exception as e:
                logger.error("mass_apply_failed", stem=stem, error=str(e))
                skipped += 1

            if progress_callback:
                progress_callback(i + 1, total, stem)

        return {
            "applied": applied,
            "skipped": skipped,
            "missing_masks": missing_masks,
        }

    @classmethod
    def unload_models(cls, *, skip_if_batch_active: bool = False) -> bool:
        """Unload all model plugins and clear memory.

        A classmethod so the global GPU-unload route can reach it without
        constructing the singleton (and therefore without loading anything);
        still callable as ``service.unload_models()`` from the existing
        instance call sites.

        ``skip_if_batch_active=True`` (the ``/system/gpu/unload`` route's mode)
        — the check for a pending/running mask batch PLUS the unload itself run
        under ``_unload_lock``, atomically, mirroring
        ``CaptionService.unload_models``. Internal callers (``generate_mask``
        switching models, the batch worker's own ``finally``) always pass the
        default ``False`` and unload unconditionally: a batch calling this on
        itself is expected, not a race.

        Returns ``True`` if the unload actually ran, ``False`` if skipped.
        """
        with cls._unload_lock:
            if skip_if_batch_active and gpu_batch_active(_MASKING_GUARD_TASK_TYPES):
                return False

            # The singleton's plugin dict, not `self.plugins`: the only
            # non-singleton construction in the tree is
            # `pipeline_data.py:583`, which calls `mass_apply` only (no plugin
            # load) and runs in the trainer subprocess, so no other instance
            # can be holding a loaded plugin.
            unload_gpu_plugins(
                cls,
                plugins=cls._instance.plugins if cls._instance else {},
                active_attr="_active_model_id",
                service_label="masking",
            )
            return True
