"""
Captioning Service for generating image captions using AI models.

Uses a plugin architecture where each model is implemented as a separate class.
"""

import threading

from PIL import Image

from app.core.captioning.models import (
    CaptionModel,
    Florence2Model,
    Qwen3VLModel,
    JoyCaptionModel,
    YoutuVLModel,
)
from app.core.gpu_unload import unload_gpu_plugins

# Default number of frames sampled per video clip for captioning. Overridable
# per-call via ``params['video_frames']`` (threaded through the batch/route the
# same way other caption params flow).
DEFAULT_VIDEO_FRAMES = 8


class CaptionService:
    """
    Singleton service for AI-powered image captioning.

    Manages model loading, caching, and caption generation using a plugin
    architecture.
    """

    _instance = None
    _model_instances: dict[str, CaptionModel] = {}
    _active_model_key = None
    # Guards unload_models(skip_if_batch_active=True)'s check-then-act: the
    # batch-active check and the actual unload now happen under this ONE
    # lock, instead of the /unload route checking task_manager.list() and
    # THEN separately calling unload_models() — a caption batch could start
    # in the window between those two steps (the unload runs in a thread
    # pool via asyncio.to_thread, so the event loop keeps serving other
    # requests, including a new POST /batch, while it's in flight).
    _unload_lock = threading.Lock()

    def __init__(self):
        # Register available models
        self.plugins: dict[str, CaptionModel] = {
            "florence-2": Florence2Model(self),
            "qwen3-vl": Qwen3VLModel(self),
            "joycaption": JoyCaptionModel(self),
            "youtu-vl": YoutuVLModel(self),
        }
        # External OpenAI-compatible API providers (no VRAM; load/unload no-op)
        from app.core.captioning.models.api_model import ApiCaptionModel

        for provider in ("openai", "anthropic", "gemini", "openrouter", "custom"):
            self.plugins[f"api-{provider}"] = ApiCaptionModel(self, provider)

    @classmethod
    def get_instance(cls) -> "CaptionService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = CaptionService()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        cls.unload_models()
        cls._instance = None

    @classmethod
    def unload_models(cls, *, skip_if_batch_active: bool = False) -> bool:
        """Unload all models from memory and clear CUDA cache.

        ``skip_if_batch_active=True`` (the ``/unload`` API route's mode) —
        the whole check for a pending/running ``caption_batch`` task PLUS
        the unload itself run under ``_unload_lock``, atomically. This is
        the ONLY mode that can no-op; internal callers (``generate_caption``
        switching models mid-batch, the batch worker's own ``finally``
        unload) always call with the default ``False`` and unload
        unconditionally, unaffected by whether a batch happens to be active
        (a batch calling this on itself is expected, not a race).

        Returns ``True`` if the unload actually ran, ``False`` if skipped.
        """
        with cls._unload_lock:
            if skip_if_batch_active:
                from app.core.tasks.task import TaskStatus
                from app.core.tasks.task_manager import task_manager

                active_batch = any(
                    t.type == "caption_batch"
                    and t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                    for t in task_manager.list()
                )
                if active_batch:
                    return False

            unload_gpu_plugins(
                cls,
                plugins=cls._instance.plugins if cls._instance else {},
                active_attr="_active_model_key",
                service_label="caption",
            )
            return True

    def supports_multi_image(self, model_id: str) -> bool:
        """Whether *model_id* can caption from multiple images (edit captions)."""
        base = "qwen3-vl" if model_id.startswith("qwen3-vl-") else model_id
        plugin = self.plugins.get(base)
        return bool(plugin and getattr(plugin, "supports_multi_image", False))

    def _resolve_extra_images(
        self,
        model_id: str,
        extra_image_paths: list[str] | None,
        params: dict,
    ) -> None:
        """Load any extra (control) images into ``params['extra_images']``.

        Raises ``ValueError`` if extras are supplied for a model that can't
        consume them, so a misconfigured edit-caption run fails loudly rather
        than silently dropping the control image.
        """
        if not extra_image_paths:
            return
        if not self.supports_multi_image(model_id):
            raise ValueError(
                f"Model '{model_id}' does not support multi-image captioning; "
                "remove the control image(s) or pick a multi-image model."
            )
        params["extra_images"] = [self._load_image(p) for p in extra_image_paths]

    @staticmethod
    def _is_video_path(path: str, params: dict) -> bool:
        """Whether *path* should be captioned as a video clip.

        Driven by file extension (a trainable, probeable video — .gif is an
        animated *image*, not a clip) OR an explicit ``params['is_video']``
        flag carried from the media item's metadata.
        """
        if params.get("is_video"):
            return True
        from pathlib import Path

        from app.core.dataset.media_types import is_probeable_video

        return is_probeable_video(Path(path).suffix)

    def _sample_video_frames(self, video_path: str, params: dict) -> list:
        """Sample N evenly-spaced RGB frames from *video_path* for captioning.

        Frame count comes from ``params['video_frames']`` (default
        ``DEFAULT_VIDEO_FRAMES``); the sample window honours user trim bounds
        ``trim_start_s`` / ``trim_end_s`` when present on the item.
        """
        from app.core.video.frames import sample_frames

        n = int(params.get("video_frames", DEFAULT_VIDEO_FRAMES))
        start_s = params.get("trim_start_s")
        end_s = params.get("trim_end_s")
        return sample_frames(video_path, n=n, start_s=start_s, end_s=end_s)

    def generate_caption(
        self,
        image_path: str,
        model_id: str,
        params: dict,
        extra_image_paths: list[str] | None = None,
    ) -> str:
        """
        Generate a caption for the given image or video using the model.

        ``extra_image_paths`` supplies additional (e.g. control/"before")
        images for two-image edit-instruction captioning; they are loaded into
        ``params['extra_images']`` for multi-image-capable models.

        Video items (detected by extension via ``is_probeable_video`` or an
        explicit ``params['is_video']`` flag) are sampled into N frames and
        dispatched to ``model.generate_video(frames, params)`` instead of the
        single-image ``generate``. Image items keep the original path.
        """
        is_video = self._is_video_path(image_path, params)

        if not is_video:
            self._resolve_extra_images(model_id, extra_image_paths, params)

        # API-provider plugins are stateless (no VRAM): bypass the shared
        # load/unload bookkeeping entirely so a background-lane API batch can
        # never unload a local model that the GPU lane is actively using.
        if model_id.startswith("api-"):
            plugin = self.plugins.get(model_id)
            if plugin is None:
                raise ValueError(f"Model '{model_id}' not supported.")
            params["image_path"] = image_path
            if is_video:
                frames = self._sample_video_frames(image_path, params)
                return plugin.generate_video(frames, params)
            image = self._load_image(image_path)
            return plugin.generate(image, params)

        # Parse model_id and variant
        base_model_id = model_id
        variant = None

        if model_id.startswith("qwen3-vl-"):
            base_model_id = "qwen3-vl"
            variant = model_id.replace("qwen3-vl-", "")

        if base_model_id not in self.plugins:
            raise ValueError(f"Model '{base_model_id}' not supported.")

        plugin = self.plugins[base_model_id]

        # Check if we need to unload previous model
        cache_key = model_id
        if self._active_model_key and self._active_model_key != cache_key:
            self.unload_models()

        # Load if not already loaded
        if self._active_model_key != cache_key:
            plugin.load(variant=variant)
            self.__class__._active_model_key = cache_key

        # Prepare params (ensure variant is passed for Qwen if needed)
        if variant:
            params["variant"] = variant

        # Ensure image_path is available for models that need it (like Youtu-VL)
        params["image_path"] = image_path

        # Video → sample frames and dispatch to the model's video path.
        if is_video:
            frames = self._sample_video_frames(image_path, params)
            return plugin.generate_video(frames, params)

        # Load image and generate (single-image path).
        image = self._load_image(image_path)
        return plugin.generate(image, params)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB format."""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Could not open image {image_path}: {e}")
