from abc import ABC, abstractmethod
from PIL import Image

class MaskingModel(ABC):
    """Abstract base class for masking model plugins."""
    
    @abstractmethod
    def load(self) -> bool:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(self, image: Image.Image, params: dict) -> Image.Image:
        """Generate a mask for the given image."""
        pass
    
    @abstractmethod
    def unload(self):
        """Unload the model to free memory."""
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """The identifier for this model (e.g., 'rembg')."""
        pass
