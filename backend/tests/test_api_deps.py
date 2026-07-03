"""Unit tests for shared FastAPI dependency helpers in ``app/api/_deps.py``."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api._deps import dataset_or_404
from app.core.dataset_manager import dataset_manager


def test_dataset_or_404_raises_when_falsy():
    with pytest.raises(HTTPException) as exc_info:
        dataset_or_404(None)
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Dataset not found"


def test_dataset_or_404_returns_dataset_when_truthy(tmp_path):
    ds_path = tmp_path / "dep_test_ds"
    ds = dataset_manager.create_dataset("dep_test_ds", path=str(ds_path))
    try:
        result = dataset_or_404(ds)
        assert result.name == "dep_test_ds"
    finally:
        dataset_manager.delete_dataset("dep_test_ds", delete_files=True)
