import torch
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Florence2ForConditionalGeneration,
    Florence2Processor,
)
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)

# The native Florence2Processor reads tokenizer.image_token / .image_token_id.
# microsoft/Florence-2-large predates native support and ships a RobertaTokenizer
# with neither, so we register the token ourselves.
_IMAGE_TOKEN = "<image>"

# CLIP ViT-L/14 @ 768: 576 patches + 1 CLS. Only used if the cached image
# processor config omits image_seq_length.
_FALLBACK_IMAGE_SEQ_LEN = 577


class Florence2Model(CaptionModel):
    MODEL_PATH = "microsoft/Florence-2-large"

    def __init__(self, service):
        self.service = service
        self.model = None
        self.processor = None

    @property
    def model_id(self) -> str:
        return "florence-2"

    def _build_native_processor(self) -> Florence2Processor:
        """Assemble the native processor, registering the image token.

        Built by hand rather than via AutoProcessor because the cached repo's
        tokenizer lacks image_token, which Florence2Processor.__init__ reads.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        if not hasattr(tokenizer, "image_token"):
            tokenizer.add_special_tokens({"additional_special_tokens": [_IMAGE_TOKEN]})
            tokenizer.image_token = _IMAGE_TOKEN
            # Derived, never hardcoded - the id depends on the vocab.
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)

        image_processor = AutoImageProcessor.from_pretrained(self.MODEL_PATH)
        if not hasattr(image_processor, "image_seq_length"):
            image_processor.image_seq_length = _FALLBACK_IMAGE_SEQ_LEN

        return Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

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

            self.processor = self._build_native_processor()

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
