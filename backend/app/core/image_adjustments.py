"""Backward-compatibility shim — all symbols re-exported from image_processing package.

This file exists so that ``from app.core.image_adjustments import X`` still works.
New code should import from ``app.core.image_processing`` directly.
"""

from app.core.image_processing import *  # noqa: F401, F403
