"""Smoke tests for the HiDream-O1 model family.

Validates wiring (registration, definition loading, class shapes) without
touching the actual 8B checkpoint — safe to run on CI / CPU-only boxes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry


# ── Registration ─────────────────────────────────────────────────────────

def test_family_is_discovered():
    """``hidream_o1`` registers itself via the auto-discovery scan."""
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("hidream_o1")
    assert family_cls.family_name == "hidream_o1"
