"""Shared FastAPI path-operation dependency helpers for API routes.

Companion to ``_path_guard.py`` (path-safety helpers) — this module holds
small, reusable pieces for building ``Depends(...)`` callables instead.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.core.dataset_manager import Dataset


def dataset_or_404(dataset: Dataset | None) -> Dataset:
    """Return *dataset* unchanged, or raise the house-standard 404.

    Centralizes the ``if not dataset: raise HTTPException(404, "Dataset not
    found")`` boilerplate duplicated across the dataset-domain routes — every
    retrofitted call site gets a byte-identical error body/status.

    Deliberately does NOT perform the ``dataset_manager.get_dataset(name)``
    lookup itself. Each route module defines its own tiny
    ``get_dataset_or_404(name)`` dependency (``return dataset_or_404(
    dataset_manager.get_dataset(name))``) that reads ITS OWN module-level
    ``dataset_manager`` name. That one extra line of indirection matters:
    the test suite's established convention (``test_control_routes.py``,
    ``test_datasets.py``, ``test_masking.py``, ``test_stats_routes.py``, ...)
    substitutes a fake manager per test via
    ``@patch("app.api.<module>.dataset_manager")`` /
    ``monkeypatch.setattr(<module>, "dataset_manager", ...)``. A dependency
    that imported and called ``dataset_manager`` directly from this shared
    module would resolve its own frozen import and never observe those
    per-module test substitutions.
    """
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset
