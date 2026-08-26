"""The training pipeline calls the port the server is actually on.

WHY THIS IS A CORRECTNESS BUG AND NOT A TIDINESS ONE:

``prepare_data()`` builds a real HTTP call to this same process to read each
dataset. It resolved the port from the ``backend_port`` *setting* alone and
ignored ``PORT`` — the variable the container actually binds. When the two
disagree the request fails, and the handler at ``pipeline_data.py`` catches
``httpx.HTTPError`` and ``continue``s: the dataset is logged and **skipped**.

So a container started with ``PORT=9000`` does not crash. It trains on fewer
datasets than the user selected and produces a worse LoRA, reporting success.
The ``port = 8000`` fallback does not help — it covers a settings *import*
failure, not a wrong port.

HOW THIS GUARD AVOIDS BEING VACUOUS (the plan is explicit about it):

Asserting "the pipeline agrees with ``main.py``" could pass by computing both
sides through ``resolve_port()``, or by exercising an extracted helper the HTTP
client never calls. ``main.BACKEND_PORT`` is import-time while the pipeline
resolves inside ``prepare_data()``, so "they agree" is not observable.

Instead these set CONFLICTING inputs and intercept the real ``httpx`` call, so
the assertion is about the URL the client was actually handed.
"""

from __future__ import annotations

import httpx

from app.core import container_config


class TestResolverPrecedence:
    """``PORT`` outranks the saved setting, and the setting still works.

    This is the precedence the rest of the app already follows —
    ``container_config.resolve_port`` is what ``main.py`` uses — so the bug was
    that one consumer had its own private answer.
    """

    def test_port_env_wins_over_the_saved_setting(self, monkeypatch):
        monkeypatch.setenv("PORT", "9001")
        assert container_config.resolve_port(default=8123) == 9001

    def test_the_setting_is_used_when_port_is_unset(self, monkeypatch):
        """Prove the negative: PORT winning must not mean the setting is dead.

        Without this, 'PORT wins' would be satisfied by ignoring the setting
        entirely — which would break the manually-launched custom-port
        deployment that the field exists to serve.
        """
        monkeypatch.delenv("PORT", raising=False)
        assert container_config.resolve_port(default=8123) == 8123

    def test_a_junk_port_falls_back_rather_than_crashing(self, monkeypatch):
        """A bad env var must not take a training run down."""
        monkeypatch.setenv("PORT", "not-a-number")
        assert container_config.resolve_port(default=8123) == 8123


class TestTheActualConsumer:
    """Intercept the real client. The URL it receives is the contract.

    Anything short of this tests a helper rather than the behaviour: the defect
    was that ``prepare_data`` computed its own port *inline*, so only the call
    it makes can show whether that is fixed.
    """

    def _first_url(self, monkeypatch, *, env_port, setting_port) -> str:
        """Run the REAL ``prepare_data`` and return the first URL it requested.

        Not a helper, not a reimplementation: this drives the shipped code path
        and intercepts ``httpx.AsyncClient.get``, so the assertion is about the
        URL the HTTP client was actually handed. That is the only thing that
        distinguishes 'the port is resolved correctly' from 'a function that
        resolves ports exists somewhere'.

        The intercepted call raises ``ConnectError`` — the same failure a wrong
        port produces in production — so this also exercises the skip path that
        makes the bug silent.
        """
        import asyncio

        from app.core.logger import get_logger
        from app.engine.core.pipeline.pipeline_data import PipelineDataMixin
        from app.engine.models.registry import ModelRegistry

        # A REAL registered definition, not a stub: `prepare_data` runs the
        # video-contract validation, which resolves capabilities through the
        # family registry. Stubbing that would mean stubbing the seam under
        # test's own dependencies until the code no longer runs.
        ModelRegistry.initialize()
        definition = next(
            d for d in ModelRegistry._definitions.values() if d.id == "sdxl_base_1.0"
        )

        if env_port is None:
            monkeypatch.delenv("PORT", raising=False)
        else:
            monkeypatch.setenv("PORT", str(env_port))

        # The saved setting, which the pipeline used to read exclusively.
        from app.core import settings_manager as sm_mod

        monkeypatch.setattr(
            sm_mod.get_settings_manager(),
            "get_module_settings",
            lambda _module: {"backend_port": setting_port},
        )

        urls: list[str] = []

        async def _fake_get(_self, url, *args, **kwargs):  # noqa: ANN001
            urls.append(str(url))
            raise httpx.ConnectError("intercepted — no server here")

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        class _Probe(PipelineDataMixin):
            """Minimum surface `prepare_data` touches before the HTTP call."""

            def __init__(self):
                self.config = {"resolutions": [1024], "datasets": ["some-dataset"]}
                self.logger = get_logger("probe")  # structlog-style: the code logs kwargs
                self.definition = definition
                self.is_video_family = False
                self._aug_h_flip = False
                self._aug_v_flip = False

        stopped_by: BaseException | None = None
        try:
            asyncio.run(_Probe().prepare_data())
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            # Everything after the intercepted request is out of scope: with no
            # datasets fetched there is nothing to bucket. But the exception is
            # KEPT and reported below rather than discarded — a probe that
            # silently fails to reach the call under test would make every
            # assertion here vacuous, which is the exact failure mode this
            # whole task is about.
            stopped_by = exc

        assert urls, (
            "prepare_data made no HTTP request, so nothing was measured. It "
            f"stopped at: {type(stopped_by).__name__}: {stopped_by}"
        )
        return urls[0]

    def test_conflicting_inputs_resolve_to_the_env_port(self, monkeypatch):
        """Step 3 of the plan's guard: PORT=9001 beats backend_port=8123.

        THIS IS THE ASSERTION THAT FAILS AGAINST THE ORIGINAL CODE, which read
        the setting and never looked at PORT. Verified failing before the fix
        was written.
        """
        url = self._first_url(monkeypatch, env_port=9001, setting_port=8123)
        assert url.startswith("http://localhost:9001/api/"), (
            f"the pipeline called {url} while the server binds PORT=9001; every "
            "dataset fetch would fail and each one is silently skipped"
        )

    def test_without_the_env_the_setting_is_honoured(self, monkeypatch):
        """Step 4: the saved setting still drives the call when PORT is unset.

        Prove the negative — 'PORT wins' must not collapse into 'the setting is
        ignored', which would break the manually-launched custom-port
        deployment the field exists to serve.
        """
        url = self._first_url(monkeypatch, env_port=None, setting_port=8123)
        assert url.startswith("http://localhost:8123/api/"), url

    def test_the_pipeline_does_not_resolve_the_port_itself(self):
        """One producer (RULE-21).

        The defect was a second, private answer to a question the app already
        had a resolver for. Pinned structurally so it cannot grow back: the
        pipeline must delegate, not recompute.
        """
        from pathlib import Path

        src = Path(
            __import__("app.engine.core.pipeline.pipeline_data", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")

        start = src.index("def _resolve_api_port")
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        assert "resolve_port" in body, (
            "pipeline_data must delegate to container_config.resolve_port; a "
            "private port calculation here is what silently skipped datasets"
        )


class TestTheSkipIsWhatMakesItSilent:
    """Record the consequence, so the fix is not read as cosmetic."""

    def test_a_failed_dataset_fetch_is_a_continue_not_a_raise(self):
        """The line that turns a wrong port into a quieter, worse LoRA.

        Not changing that behaviour here — a partial-data run is a real
        product decision and is out of this task's scope. Pinning it so the
        cost of getting the port wrong stays visible to whoever reads this
        next.
        """
        from pathlib import Path

        src = Path(
            __import__("app.engine.core.pipeline.pipeline_data", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")

        idx = src.index("dataset_api_error")
        following = src[idx : idx + 200]
        assert "continue" in following, (
            "the skip-on-error behaviour changed; if datasets now fail loudly "
            "that is an improvement, but this test documented the silent skip "
            "and should be updated deliberately rather than deleted"
        )
