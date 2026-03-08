from abc import ABC, abstractmethod
from PIL import Image
from typing import Any


class CaptionModel(ABC):
    """Abstract base class for captioning model plugins."""

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
