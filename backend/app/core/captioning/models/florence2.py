import torch
from transformers import (
    AutoProcessor,
    Florence2ForConditionalGeneration,
)
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)


class Florence2Model(CaptionModel):
    # florence-community/Florence-2-large is the natively-converted repo (no
    # auto_map, no remote code). microsoft/Florence-2-large still ships the
    # legacy remote-code weight layout, which the native
    # Florence2ForConditionalGeneration class cannot load — from_pretrained
    # raises "You set 'ignore_mismatched_sizes' to False" on it.
    #
    # The converted repo's own tokenizer already carries image_token /
    # image_token_id (verified: AutoProcessor.from_pretrained(MODEL_PATH)
    # returns a working Florence2Processor with image_token="<image>",
    # image_token_id=51289, image_processor.image_seq_length=577 -- no
    # hand-assembly required), so the shim that used to register the token
    # by hand for microsoft/Florence-2-large's pre-native tokenizer is gone.
    MODEL_PATH = "florence-community/Florence-2-large"

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

        # `with_progress` emits starting / complete (or error) WS events around
        # the download. We do NOT pass `tqdm_class=` to `from_pretrained` —
        # in transformers >= 4.50 it leaks straight through `model_kwargs`
        # into the model class's `__init__` (TypeError: unexpected kwarg
        # 'tqdm_class'). Per-chunk download bar is sacrificed; the start /
        # complete events still drive the frontend download indicator.
        from app.api.events.download_progress import with_progress

        logger.info("loading_florence2", path=self.MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        with with_progress(model_id=self.MODEL_PATH, category="caption", repo_id=self.MODEL_PATH):
            # Native implementation (transformers >= 5.x) — no remote code, and
            # no KV-cache monkey-patch: the EncoderDecoderCache bug lived in the
            # hub's modeling_florence2.py, which we no longer execute.
            self.model = Florence2ForConditionalGeneration.from_pretrained(
                self.MODEL_PATH,
                dtype=dtype,
                attn_implementation="eager",
            ).to(device)

            self.processor = AutoProcessor.from_pretrained(self.MODEL_PATH)

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
