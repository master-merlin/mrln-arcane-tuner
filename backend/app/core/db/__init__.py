"""SQLite data layer for Arcane Tuner.

Provides the DatabaseEngine singleton, schema migrations, and repository
classes for datasets, media items, job history, templates, and metrics.
"""

from .engine import DatabaseEngine, get_db

__all__ = ["DatabaseEngine", "get_db"]
