"""The backend suite never reaches the Hugging Face hub.

``backend/conftest.py`` puts every test process in hub-offline mode before
``huggingface_hub`` is imported. These tests prove the flag is seen by the
library, refuses a real request, and reaches child processes too.
"""
import os
import subprocess
import sys

import huggingface_hub.constants
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import OfflineModeIsEnabled


def test_the_suite_runs_with_the_hub_offline():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert huggingface_hub.constants.HF_HUB_OFFLINE is True


def test_a_hub_request_is_refused_before_it_leaves():
    with pytest.raises(OfflineModeIsEnabled):
        HfApi().model_info("tencent/Youtu-VL-4B-Instruct")


def test_a_child_process_inherits_the_offline_hub():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import huggingface_hub.constants as c; print(c.HF_HUB_OFFLINE)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert result.stdout.strip() == "True"
