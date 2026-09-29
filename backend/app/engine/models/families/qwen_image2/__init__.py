"""Qwen-Image 2 model family (Qwen-Image-2.1) -- family, loader, definitions.

The trainer is imported by ``QwenImage2Family.get_trainer_class`` on use, not
here: this package is imported at registry discovery (ARCHITECTURE D1).
"""
from .family import QwenImage2Family as QwenImage2Family
from .loader import QwenImage2Loader as QwenImage2Loader
