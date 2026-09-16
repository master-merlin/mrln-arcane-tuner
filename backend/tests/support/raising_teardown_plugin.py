"""A plugin with one autouse fixture whose FINALISER raises.

Loaded only with ``-p tests.support.raising_teardown_plugin`` by
``tests/test_machine_lock.py`` (never by the suite itself), to reproduce the
shape of VERIFY finding 1.02: a fixture finaliser that raises makes pytest's
own ``pytest_runtest_teardown`` impl (``_pytest.runner``, which calls
``teardown_exact``) raise, and pluggy then never calls the impls ordered after
it — so a lock released in a plain ``trylast`` teardown hook is not released at
all, and the next test of the group waits out its whole bound against its own
still-live worker. The session itself stays healthy, so that next test really
does run and can show the effect.
"""
import pytest


@pytest.fixture(autouse=True)
def explode_at_teardown():
    yield
    raise RuntimeError("deliberate teardown explosion")
