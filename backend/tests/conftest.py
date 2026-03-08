"""
Shared test fixtures for the MRLN Arcane Tuner backend.
"""
import sys
import os
import logging
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_logging():
    """Redirect all logging to tests/tests.log, keeping server.log clean.

    This fixture runs once per test session and:
    - Calls setup_logging WITHOUT the server.log file handler
    - Removes the WebSocket handler (no event loop in tests)
    - Adds a dedicated FileHandler writing to tests/tests.log
    """
    from app.core.logger import setup_logging

    # Configure structlog + standard logging without server.log or WS
    setup_logging(log_level="DEBUG", include_file_handler=False)

    root = logging.getLogger()
    # Remove any remaining handlers (console, websocket) attached by setup_logging
    root.handlers = []

    # Dedicated test log file — reset on each session
    test_log_path = os.path.join(os.path.dirname(__file__), "tests.log")
    if os.path.exists(test_log_path):
        try:
            os.remove(test_log_path)
        except OSError:
            pass

    handler = logging.FileHandler(test_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    yield

    # Cleanup after session
    root.handlers = []



@pytest.fixture
def mock_definition():
    """Provides a mock ModelDefinition for testing."""
    definition = MagicMock()
    definition.id = "test_model_id"
    definition.name = "Test Model"
    definition.family = "sdxl"
    definition.source = "/fake/model/path"
    definition.components = {}
    return definition


@pytest.fixture
def mock_config():
    """Provides a standard training config dict for testing."""
    return {
        "lora_name": "test_lora",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "output_dir": "./test_outputs",
        "datasets": [{"dataset_name": "test_dataset"}],
        "max_train_steps": 10,
        "train_batch_size": 1,
        "learning_rate": 1e-4,
        "network_rank": 16,
        "network_alpha": 8.0,
        "gradient_accumulation_steps": 1,
        "resolutions": [1024],
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary output directory that is auto-cleaned."""
    output = tmp_path / "outputs"
    output.mkdir()
    return str(output)


@pytest.fixture
def client():
    """Shared FastAPI test client — avoids duplicating in every test file."""
    from app.main import app

    return TestClient(app)
