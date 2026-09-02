"""The warning category the CSP tests raise when they fall back to source.

It lives HERE, in an importable support module, and not in the test module
that raises it, because of pytest-xdist (LANE-63): a worker serialises every
warning as ``(message_module, message_class_name)`` and the controller
re-imports that module by name to rebuild it (``xdist/workermanage.py``
``unserialize_warning_message``, no fallback). A test module's importlib-mode
name (``api.test_csp_policy`` — ``tests/api`` has an ``__init__.py``,
``tests/`` does not) is not importable from the controller, which then reports
``ModuleNotFoundError: No module named 'api'``, marks the worker down, and the
loadgroup scheduler dies with a ``KeyError`` — the whole gate, over one
warning. ``tests.support`` resolves in the controller because the root
``backend/conftest.py`` puts ``backend/`` on ``sys.path`` in every process.
``test_xdist_warning_classes.py`` keeps warning classes out of test modules.
"""


class SourceFallbackWarning(UserWarning):
    """Raised when the CSP checks run against source instead of the built page.

    A distinct category rather than a bare ``UserWarning`` so a caller that
    wants the strict behaviour can ask for it by name —
    ``-W error::tests.support.source_fallback.SourceFallbackWarning`` in CI,
    once CI builds the frontend before the backend gate. It is deliberately
    NOT escalated in the test module: see its docstring.
    """
