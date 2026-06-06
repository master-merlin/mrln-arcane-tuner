"""Abstract base class for scoring model plugins."""
from abc import ABC, abstractmethod

from PIL import Image


class ScoringModel(ABC):
    """
    Base class for image quality scoring models.

    Each model plugin implements load/score/unload lifecycle methods,
    following the same pattern as CaptionModel and MaskingModel.
    """

    def __init__(self, service: "ScoringService") -> None:  # noqa: F821
        self.service = service

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load the model into GPU memory."""
        ...

    @abstractmethod
    def score(self, image: Image.Image, params: dict) -> float:
        """
        Score a single image.

        Returns a float score where higher = better quality.
        The range is model-specific (HPSv2 returns raw CLIP similarity).
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        ...
