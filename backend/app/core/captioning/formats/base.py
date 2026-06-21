from __future__ import annotations
from abc import ABC, abstractmethod


class CaptionFormat(ABC):
    id: str = "plain"
    is_structured: bool = False

    @abstractmethod
    def parse_and_normalize(self, raw: str) -> dict: ...
    @abstractmethod
    def serialize(self, data: dict) -> str: ...

    def build_generation_prompt(
        self, user_instructions: str | None = None
    ) -> str | None:
        return None  # plain → no model-specific prompt override

    def generation_overrides(self) -> dict:
        return {}

    def json_schema(self) -> dict | None:
        return None

    def detect(self, text: str) -> bool:
        return False

    def build_refine_prompt(self, target, data: dict) -> str | None:
        return None
