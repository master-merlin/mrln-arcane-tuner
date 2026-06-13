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
