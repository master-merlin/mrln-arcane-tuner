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
        image_jpeg = _encode_jpeg(image, int(params.get("max_long_side", 1024)))
        return chat_vision(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=model,
            prompt=self.resolve_prompt(params),
            image_jpeg=image_jpeg,
            temperature=float(params.get("temperature", 0.7)),
            top_p=float(params.get("top_p", 1.0)),
            max_tokens=int(params.get("max_tokens", 512)),
        )
