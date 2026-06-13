"""
Captioning Service for generating image captions using AI models.

Uses a plugin architecture where each model is implemented as a separate class.
"""
import gc

import structlog
import torch
from PIL import Image

from app.core.captioning.models import (
    CaptionModel, 
    Florence2Model, 
    Qwen3VLModel, 
    JoyCaptionModel,
    YoutuVLModel
)

logger = structlog.get_logger(__name__)

class CaptionService:
    """
    Singleton service for AI-powered image captioning.
    
    Manages model loading, caching, and caption generation using a plugin
    architecture.
    """
    _instance = None
    _model_instances: dict[str, CaptionModel] = {}
    _active_model_key = None

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
    def unload_models(cls):
        """Unload all models from memory and clear CUDA cache."""
        if cls._active_model_key:
            logger.info("unloading_caption_models", active_model=cls._active_model_key)
        
        if cls._instance:
            for plugin in cls._instance.plugins.values():
                plugin.unload()
        
        cls._active_model_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("all_caption_models_unloaded")

    def supports_multi_image(self, model_id: str) -> bool:
        """Whether *model_id* can caption from multiple images (edit captions)."""
        base = "qwen3-vl" if model_id.startswith("qwen3-vl-") else model_id
        plugin = self.plugins.get(base)
        return bool(plugin and getattr(plugin, "supports_multi_image", False))

    def _resolve_extra_images(
        self, model_id: str, extra_image_paths: list[str] | None, params: dict,
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

    def generate_caption(
        self, image_path: str, model_id: str, params: dict,
        extra_image_paths: list[str] | None = None,
    ) -> str:
        """
        Generate a caption for the given image using the specified model.

        ``extra_image_paths`` supplies additional (e.g. control/"before")
        images for two-image edit-instruction captioning; they are loaded into
        ``params['extra_images']`` for multi-image-capable models.
        """
        self._resolve_extra_images(model_id, extra_image_paths, params)

        # API-provider plugins are stateless (no VRAM): bypass the shared
        # load/unload bookkeeping entirely so a background-lane API batch can
        # never unload a local model that the GPU lane is actively using.
        if model_id.startswith("api-"):
            plugin = self.plugins.get(model_id)
            if plugin is None:
                raise ValueError(f"Model '{model_id}' not supported.")
            image = self._load_image(image_path)
            params["image_path"] = image_path
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
            
        # Load image
        image = self._load_image(image_path)
        
        # Prepare params (ensure variant is passed for Qwen if needed)
        if variant:
            params["variant"] = variant
            
        # Ensure image_path is available for models that need it (like Youtu-VL)
        params["image_path"] = image_path
            
        # Generate
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
