import torch
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)

DEFAULT_MAX_LONG_SIDE = 1280


class Qwen3VLModel(CaptionModel):
    """Qwen3-VL captioning model with multi-variant support.

    Supports Instruct and Thinking variants at 4B, 8B, and 32B sizes.
    Images exceeding 1280px on the long side are automatically resized
    in-memory before inference to stay within the model's optimal range.
    """

    VARIANTS = {
        "4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
        "4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
        "8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
        "8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
        "32B-Instruct": "Qwen/Qwen3-VL-32B-Instruct",
        "32B-Thinking": "Qwen/Qwen3-VL-32B-Thinking",
    }

    def __init__(self, service):
        self.service = service
        self.model = None
        self.processor = None
        self.loaded_variant = None

    @property
    def model_id(self) -> str:
        return "qwen3-vl"

    def load(self, variant: str = "4B-Instruct") -> tuple[Any, Any]:
        """Load model and processor for the given variant."""
        # Skip reload if same variant already loaded
        if self.model is not None and self.processor is not None and self.loaded_variant == variant:
            logger.debug("qwen3_vl_already_loaded", variant=variant)
            return self.model, self.processor

        model_id = self.VARIANTS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown Qwen3-VL variant: {variant}. "
                f"Available: {list(self.VARIANTS.keys())}"
            )

        logger.info("loading_qwen3_vl", variant=variant, path=model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use bfloat16 for 32B models if supported
        if "32B" in variant:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32

        # `with_progress` emits starting / complete (or error) WS events around
        # the download. We do NOT pass `tqdm_class=` to `from_pretrained` —
        # in transformers >= 4.50 it leaks straight through `model_kwargs`
        # into the model class's `__init__` (TypeError: unexpected kwarg
        # 'tqdm_class'). Per-chunk download bar is sacrificed; the start /
        # complete events still drive the frontend download indicator.
        from app.api.events.download_progress import with_progress

        with with_progress(model_id=model_id, category="caption", repo_id=model_id):
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

        self.loaded_variant = variant
        logger.info("qwen3_vl_loaded", variant=variant)
        return self.model, self.processor

    def unload(self):
        """Unload model and processor to free memory."""
        self.model = None
        self.processor = None
        self.loaded_variant = None

    def resolve_prompt(self, params: dict) -> str:
        """Resolve the UI 'System Prompt' for the Qwen3-VL system role.

        For Qwen3-VL the custom prompt is placed in the system role of the
        chat message list, not the user role.
        """
        return params.get("system_prompt") or (
            "You are a helpful assistant that provides accurate and detailed image descriptions."
        )

    def generate(self, image: Image.Image, params: dict) -> str:
        """Generate a caption for the given image.

        Args:
            image: PIL Image in RGB mode.
            params: Generation parameters including optional system_prompt,
                    temperature, max_tokens, top_p, num_beams,
                    repetition_penalty, and variant.

        Returns:
            The generated caption string.
        """
        variant = params.get("variant", self.loaded_variant or "4B-Instruct")

        # If we need a different variant than what's loaded,
        # normally the service would have handled unloading, but let's check here too.
        if self.loaded_variant != variant:
            self.load(variant)

        # Get generation parameters
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 512)
        top_p = params.get("top_p", 0.8)
        num_beams = params.get("num_beams", 1)
        repetition_penalty = params.get("repetition_penalty", 1.2)

        is_thinking = "Thinking" in variant

        # Resize image if long side exceeds the configured max
        max_long_side = int(params.get("max_long_side", DEFAULT_MAX_LONG_SIDE))
        image = self._resize_for_inference(image, max_long_side)

        # Build prompts — system_prompt goes to the system role
        system_prompt = self.resolve_prompt(params)

        if is_thinking:
            system_prompt = params.get("system_prompt") or (
                "You are a helpful assistant that carefully analyzes images. "
                "Think through your observations step by step before providing a detailed description."
            )
            user_prompt = params.get("user_prompt") or (
                "Please carefully analyze this image and provide a detailed description."
            )
        else:
            user_prompt = params.get("user_prompt") or "Describe this image in detail."

        # Build messages in Qwen VL chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # Apply chat template and process inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Build generation config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "repetition_penalty": repetition_penalty,
        }

        if num_beams > 1:
            # Beam search mode - do_sample must be False
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
        else:
            # Sampling mode - use temperature and top_p
            gen_kwargs["do_sample"] = temperature > 0
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # For Thinking mode, remove <think>...</think> blocks
        if is_thinking:
            output_text = self._extract_thinking_answer(output_text)

        return output_text.strip()

    def _resize_for_inference(
        self, image: Image.Image, max_long_side: int = DEFAULT_MAX_LONG_SIDE
    ) -> Image.Image:
        """Resize image so its long side does not exceed max_long_side.

        Uses LANCZOS resampling and preserves aspect ratio. The resize is
        purely in-memory — no files are written to disk.

        Args:
            image: Source PIL Image.
            max_long_side: Maximum allowed pixel length on the longest side.

        Returns:
            The original image if within bounds, or a resized copy.
        """
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

    def _extract_thinking_answer(self, text: str) -> str:
        """Extract final answer from thinking mode output, removing <think> blocks."""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip() or text.strip()

