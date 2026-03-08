import torch
import sys
from unittest.mock import MagicMock

# Mock pydensecrf for Windows compatibility (custom modeling code in tencent/Youtu-VL-4B-Instruct depends on it)
try:
    import pydensecrf  # noqa: F401
except ImportError:
    mock = MagicMock()
    sys.modules["pydensecrf"] = mock
    sys.modules["pydensecrf.densecrf"] = mock
    sys.modules["pydensecrf.utils"] = mock

from transformers import AutoProcessor, AutoModelForCausalLM
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)

DEFAULT_MAX_LONG_SIDE = 768

class YoutuVLModel(CaptionModel):
    """
    Implementation of tencent/Youtu-VL-4B-Instruct model.
    """
    
    def __init__(self, service):
        self.service = service
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_id(self) -> str:
        return "youtu-vl"

    def load(self, variant: str = None) -> tuple[Any, Any]:
        model_id = "tencent/Youtu-VL-4B-Instruct"
        logger.info("loading_youtu_vl", path=model_id)
        
        # Use bfloat16 if supported, else float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if self.device == "cpu":
            dtype = torch.float32

        # Use eager attention for maximum compatibility on Windows with this custom model.
        # This avoids KeyError: 'sdpa' and ImportError: FlashAttention2.
        attn_impl = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation=attn_impl
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True
        )

        # Inject our bundled fast image processor if the loaded one is slow.
        # This caps vision tokens via max_num_patches for much faster inference.
        try:
            from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast
            if not isinstance(self.processor.image_processor, Siglip2ImageProcessorFast):
                logger.info("injecting_fast_image_processor", max_num_patches=256)
                self.processor.image_processor = Siglip2ImageProcessorFast(
                    max_num_patches=256
                )
        except Exception as e:
            logger.warning("fast_processor_injection_failed", error=str(e))
        
        
        logger.info("youtu_vl_loaded", attention_implementation=attn_impl)
        return self.model, self.processor

    def unload(self):
        self.model = None
        self.processor = None

    def generate(self, image: Image.Image, params: dict) -> str:
        if not self.model or not self.processor:
            self.load()

        # Get generation parameters - favor speed by default
        temperature = params.get("temperature", 0.0) # 0.0 for greedy = faster
        top_p = params.get("top_p", 1.0)
        max_tokens = params.get("max_tokens", 128) # 512 is too slow for captions
        repetition_penalty = params.get("repetition_penalty", 1.05)
        
        # Image path is required by Youtu-VL generate method
        image_path = params.get("image_path")
        if not image_path:
            raise ValueError("Youtu-VL model requires the original image_path for generation.")

        # Resize image if long side exceeds the configured max
        max_long_side = int(params.get("max_long_side", DEFAULT_MAX_LONG_SIDE))
        image = self._resize_for_inference(image, max_long_side)

        # Apply max_num_patches to the fast processor if available
        max_num_patches = int(params.get("max_num_patches", 256))
        try:
            from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast
            if isinstance(self.processor.image_processor, Siglip2ImageProcessorFast):
                self.processor.image_processor.max_num_patches = max_num_patches
        except Exception:
            pass

        prompt = self.resolve_prompt(params)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                max_new_tokens=max_tokens,
                img_input=image_path, # Specific to Youtu-VL
            )
        
        # Decode only new tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = [
            out_ids[input_len:] for out_ids in generated_ids
        ]
        
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return outputs[0].strip()

    def _resize_for_inference(
        self, image: Image.Image, max_long_side: int = DEFAULT_MAX_LONG_SIDE
    ) -> Image.Image:
        """Resize image so its long side does not exceed max_long_side."""
        long_side = max(image.width, image.height)
        if long_side <= max_long_side:
            return image

        scale = max_long_side / long_side
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        logger.info(
            "resizing_image_for_inference",
            original_size=f"{image.width}x{image.height}",
            new_size=f"{new_width}x{new_height}",
            max_long_side=max_long_side,
        )
        return image.resize((new_width, new_height), Image.LANCZOS)
