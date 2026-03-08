import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)

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
        logger.info("loading_florence2", path=self.MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Use eager attention to avoid SDPA compatibility issues with transformers 4.57+
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATH, 
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="eager"
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_PATH, 
            trust_remote_code=True
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
        num_beams = params.get("num_beams", 5)
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

        # use_cache=False fixes past_key_values issue with transformers 4.57+
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=num_beams,
            use_cache=False
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer[prompt] if isinstance(parsed_answer, dict) else parsed_answer
