"""External-API caption model — proxies to OpenAI-compatible providers.

One instance per provider, registered under model_id ``api-<provider>``.
No VRAM is used: load/unload are no-ops, so the batch worker's
unload-on-switch and finally-unload remain harmless.
"""

from __future__ import annotations

import io

from PIL import Image

from app.core.captioning.models.base import CaptionModel
from app.core.llm.openai_compat import chat_vision
from app.core.llm.provider_settings import resolve_provider


def _encode_jpeg(image: Image.Image, max_long_side: int) -> bytes:
    """Downscale to *max_long_side* (never upscale) and JPEG-encode."""
    img = image
    if max(img.size) > max_long_side:
        img = img.copy()
        img.thumbnail((max_long_side, max_long_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ApiCaptionModel(CaptionModel):
    """Caption via an external OpenAI-compatible chat-completions API."""

    # OpenAI-compatible chat APIs accept multiple image parts natively.
    supports_multi_image = True

    def __init__(self, service, provider: str):
        self.provider = provider

    @property
    def model_id(self) -> str:
        return f"api-{self.provider}"

    def load(self, variant: str = None):
        return None, None

    def unload(self):
        pass

    def generate(self, image: Image.Image, params: dict) -> str:
        cfg = resolve_provider(self.provider)
        model = str(params.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"No provider model selected for '{self.model_id}'. "
                "Pick one in the API captioning settings (e.g. gpt-4o).")
        max_side = int(params.get("max_long_side", 1024))
        image_jpeg = _encode_jpeg(image, max_side)
        # Control ("before") images for two-image edit captioning, encoded in
        # the same order the caller supplied (control first, target last).
        extra = params.get("extra_images") or []
        extra_jpeg = [_encode_jpeg(img, max_side) for img in extra]
        prompt = self.resolve_prompt(params)
        if extra and not params.get("system_prompt"):
            from app.core.captioning.models.base import MULTI_IMAGE_INSTRUCTION
            prompt = MULTI_IMAGE_INSTRUCTION
        return chat_vision(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=model,
            prompt=prompt,
            image_jpeg=image_jpeg,
            extra_images_jpeg=extra_jpeg,
            temperature=float(params.get("temperature", 0.7)),
            top_p=float(params.get("top_p", 1.0)),
            max_tokens=int(params.get("max_tokens", 512)),
            should_abort=params.get("_should_abort"),
        )
