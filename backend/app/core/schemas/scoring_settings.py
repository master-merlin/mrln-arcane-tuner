"""Pydantic schemas for scoring settings."""
from pydantic import BaseModel, ConfigDict


class ScoringSettings(BaseModel):
    """Settings for the image quality scoring module."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "hpsv2"
    hps_version: str = "v2.1"
    use_captions: bool = True
    fallback_prompt: str = ""
    auto_score_on_scan: bool = False
