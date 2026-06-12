"""Make ``backend/`` importable when these API tests are collected.

Mirrors ``backend/tests/conftest.py``'s sys.path insert. Without this, running
or collecting ``backend/app/api/tests`` raises ``ModuleNotFoundError: No module
named 'app'`` because the package root (``backend/``) is not on the path. No-op
when ``backend/`` is already importable.
"""
import os
import sys

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
