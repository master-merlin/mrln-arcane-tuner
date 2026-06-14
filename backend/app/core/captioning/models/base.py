from abc import ABC, abstractmethod
from PIL import Image
from typing import Any

# Default instruction used for two-image (edit) captioning when the user has
# not supplied a custom prompt. The VLM is shown the control ("before") image
# first and the target ("after") image last; the caption is an edit instruction.
MULTI_IMAGE_INSTRUCTION = (
    "The first image is the original and the second image is the edited "
    "result. Write a single concise instruction (imperative, no preamble) "
    "describing the edit applied to the original to produce the result."
)

# Default instruction for VIDEO (motion-aware) captioning when the user has not
# supplied a custom prompt. The VLM is shown N evenly-spaced frames sampled from
# the clip; the caption should read the frames as a single moving scene, not a
# slideshow. Present tense, no frame enumeration.
VIDEO_MOTION_INSTRUCTION = (
    "These frames are sampled in order from a single video clip. Write one "
    "concise caption (present tense, no preamble) describing the scene as "
    "continuous motion: the main subject, what it is doing, any camera "
    "movement (pan, zoom, tracking, static), and the setting. Do not number "
    "or enumerate the frames or mention that this is a sequence of images."
)


class CaptionModel(ABC):
    """Abstract base class for captioning model plugins."""

    # Whether this model can caption from more than one image at once (e.g.
    # control + target for edit-instruction captions). Single-image models
    # leave this False; the service rejects extra images for them.
    supports_multi_image: bool = False

    @abstractmethod
    def load(self, variant: str = None) -> tuple[Any, Any]:
        """Load the model and processor."""
        pass

    @abstractmethod
    def generate(self, image: Image.Image, params: dict) -> str:
        """Generate a caption for the given image."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model and processor to free memory."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The identifier for this model (e.g., 'florence-2')."""
        pass

    def generate_video(self, frames: list[Image.Image], params: dict) -> str:
        """Generate a caption for a video clip given *frames* sampled from it.

        Default (honest single-frame fallback): caption the MIDDLE frame via the
        single-image :meth:`generate`. Models with genuine multi-frame / video
        support (e.g. multi-image VLMs and OpenAI-compatible API providers)
        override this to pass all frames to the model.

        Args:
            frames: Ordered RGB PIL frames evenly sampled across the clip.
            params: Generation parameters (same shape as :meth:`generate`).

        Returns:
            The generated caption string.
        """
        if not frames:
            raise ValueError("generate_video requires at least one frame.")
        middle = frames[len(frames) // 2]
        return self.generate(middle, params)

    def resolve_prompt(self, params: dict) -> str | None:
        """Resolve the user-configured 'System Prompt' into this model's instruction.

        Each model overrides this to map the prompt to the correct chat role
        or format. For example, chat-based models place it in the system role,
        while instruction-following models place it in the user message.

        Args:
            params: Generation parameters dict, may contain 'system_prompt'.

        Returns:
            The resolved prompt string, or None if the model does not use
            free-text prompts (e.g. Florence2 task-type tags).
        """
        return params.get("system_prompt") or "Describe this image in detail."
