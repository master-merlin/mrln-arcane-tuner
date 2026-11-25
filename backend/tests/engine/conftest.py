"""Fixtures shared by the engine tests (imports stay inside them: this file loads for every test here)."""
import pytest


@pytest.fixture(scope="module")
def initialized_registry():
    """The model registry with every family and definition discovered."""
    from app.engine.models.registry import registry

    registry.initialize()
    return registry


@pytest.fixture(scope="session")
def radc_curriculum() -> str:
    """A RADC sampler with a curriculum from high to low noise and a resolution term."""
    return "radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0.15"


def _shell_of(definition_id: str):
    from app.engine.core.noise_map import sampling_shell
    from app.engine.models.registry import registry

    return sampling_shell(registry.get_definition(definition_id), {}, 1)


def _trainer_class_of(definition_id: str) -> type:
    return type(_shell_of(definition_id))


@pytest.fixture
def scheduler_raises(monkeypatch):
    """Make a definition's scheduler unbuildable: its trainer class's and its driver class's
    ``init_scheduler`` raise ``RuntimeError("no scheduler")``."""

    def patch(definition_id: str) -> None:
        def raising(self):
            raise RuntimeError("no scheduler")

        shell = _shell_of(definition_id)
        for cls in (type(shell), type(shell.driver)):
            monkeypatch.setattr(cls, "init_scheduler", raising)

    return patch


@pytest.fixture
def draw_raises(monkeypatch):
    """Make a definition's trainer class raise ``RuntimeError("boom")`` from ``sample_timesteps``;
    returns the patched class."""

    def patch(definition_id: str) -> type:
        def boom(self, batch_size, latents=None):
            raise RuntimeError("boom")

        cls = _trainer_class_of(definition_id)
        monkeypatch.setattr(cls, "sample_timesteps", boom)
        return cls

    return patch
