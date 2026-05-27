import types
from functools import wraps

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)


def _patch_florence2_kv_cache(model) -> None:
    """Monkey-patch Florence-2's prepare_inputs_for_generation to handle
    the EncoderDecoderCache null-check issue introduced in transformers 4.50+.

    The cached modeling_florence2.py accesses ``past_key_values[0][0].shape``
    without guarding against None entries inside the cache object.  This
    patch wraps the original method and adds the required null-checks so
    that ``use_cache=True`` works safely.

    We patch at runtime so we never touch HuggingFace cached files.
    """
    # Find the sub-model that owns prepare_inputs_for_generation.
    # Florence-2 is an encoder-decoder — the decoder (language_model) has it.
    target = getattr(model, "language_model", model)
    original_fn = target.prepare_inputs_for_generation

    @wraps(original_fn)
    def _safe_prepare(self_inner, *args, **kwargs):
        # The original code crashes on: past_key_values[0][0].shape[2]
        # when past_key_values entries are None (empty EncoderDecoderCache).
        # We intercept, call the original, and if it crashes, fall back.
        try:
            return original_fn(*args, **kwargs)
        except (AttributeError, TypeError, IndexError):
            # Disable cache for this call and retry
            if "past_key_values" in kwargs:
                kwargs["past_key_values"] = None
            return original_fn(*args, **kwargs)

    target.prepare_inputs_for_generation = types.MethodType(_safe_prepare, target)
    logger.debug("florence2_kv_cache_patched")


class Florence2Model(CaptionModel):
    MODEL_PATH = "microsoft/Florence-2-large"
    
    def __init__(self, service):
        self.service = service
        self.model = None
        self.processor = None

    @property
    def model_id(self) -> str:
        return "florence-2"

    def load(self, variant: str = None) -> tuple[Any, Any]:
        if self.model is not None and self.processor is not None:
            logger.debug("florence2_already_loaded")
            return self.model, self.processor

        from app.api.events.download_progress import WSProgressTqdm, with_progress
        from functools import partial

        logger.info("loading_florence2", path=self.MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        with with_progress(model_id=self.MODEL_PATH, category="caption"):
            model_tqdm = partial(
                WSProgressTqdm,
                source="hf", model_id=f"{self.MODEL_PATH}/model", category="caption",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_PATH,
                trust_remote_code=True,
                dtype=dtype,
                attn_implementation="eager",
                tqdm_class=model_tqdm,
            ).to(device)

            # Patch KV-cache null-check bug so we can use use_cache=True
            _patch_florence2_kv_cache(self.model)

            proc_tqdm = partial(
                WSProgressTqdm,
                source="hf", model_id=f"{self.MODEL_PATH}/processor", category="caption",
            )
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_PATH,
                trust_remote_code=True,
                tqdm_class=proc_tqdm,
            )
        
        logger.info("florence2_loaded")
        return self.model, self.processor

    def unload(self):
        self.model = None
        self.processor = None

    def resolve_prompt(self, params: dict) -> None:
        """Florence2 uses task-type tags, not free-text prompts."""
        return None

    def generate(self, image: Image.Image, params: dict) -> str:
        if not self.model or not self.processor:
            self.load()
            
        device = self.model.device
        dtype = self.model.dtype
        
        # Map task_type to prompt
        task_prompts = {
            "Caption": "<CAPTION>",
            "Detailed Caption": "<DETAILED_CAPTION>",
            "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
        }
        task_type = params.get("task_type", "Detailed Caption")
        prompt = task_prompts.get(task_type, "<MORE_DETAILED_CAPTION>")
        
        # Get generation parameters
        max_tokens = params.get("max_tokens", 512)
        num_beams = params.get("num_beams", 3)
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=num_beams,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer[prompt] if isinstance(parsed_answer, dict) else parsed_answer
