"""Write runtime-config.json for the Angular frontend.

The frontend reads this file at bootstrap (via APP_INITIALIZER) to
discover the backend port dynamically.  The backend writes this file:

1. On every startup  (``main.py`` lifespan)
2. When ``application`` settings change (``settings_routes.py``)

The file is written to ``frontend/public/runtime-config.json`` so
Angular's dev server serves it as a static asset at
``/runtime-config.json``.
"""

from __future__ import annotations

import json
import os

from app.core.logger import get_logger

logger = get_logger(__name__)

# __file__      = .../backend/app/core/runtime_config.py
# dirname x2    = .../backend/app  (the app package)
# dirname x1    = .../backend      (the backend dir)
# dirname x1    = .../             (project root containing backend/ and frontend/)
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_DIR = os.path.dirname(_APP_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "frontend", "public", "runtime-config.json")


def write_runtime_config(backend_port: int, frontend_port: int) -> None:
    """Persist the runtime config to the frontend's public directory."""
    config = {
        "backendPort": backend_port,
        "frontendPort": frontend_port,
    }

    try:
        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("runtime_config_written", path=_CONFIG_PATH, config=config)
    except OSError as e:
        logger.warning("runtime_config_write_failed", error=str(e), path=_CONFIG_PATH)
