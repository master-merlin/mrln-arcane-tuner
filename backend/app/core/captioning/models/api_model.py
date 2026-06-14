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
                "Pick one in the API captioning settings (e.g. gpt-4o)."
            )
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

    def generate_video(self, frames: list[Image.Image], params: dict) -> str:
        """Caption a clip via the OpenAI-compatible chat API using N frames.

        Builds a single user message whose ``content`` array carries one
        ``image_url`` part per sampled frame (base64 JPEG data URLs, encoded by
        ``chat_vision``) plus a motion-aware text prompt. Frames are sent in
        chronological order — ``chat_vision`` emits ``extra_images`` first then
        the main image, so the earlier frames go in ``extra`` and the last frame
        is the main image. Cancellation (``_should_abort``) is preserved, and as
        a stateless ``api-*`` model this never touches the local unload path.
        """
        if not frames:
            raise ValueError("generate_video requires at least one frame.")
        cfg = resolve_provider(self.provider)
        model = str(params.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"No provider model selected for '{self.model_id}'. "
                "Pick one in the API captioning settings (e.g. gpt-4o)."
            )
        max_side = int(params.get("max_long_side", 1024))
        jpegs = [_encode_jpeg(f, max_side) for f in frames]
        # Chronological order: earlier frames as extras, last frame as main.
        extra_jpeg = jpegs[:-1]
        image_jpeg = jpegs[-1]

        from app.core.captioning.models.base import VIDEO_MOTION_INSTRUCTION

        prompt = (
            params.get("system_prompt")
            or params.get("user_prompt")
            or VIDEO_MOTION_INSTRUCTION
        )
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
