"""A gated Hugging Face repo must read as an instruction, not as a broken model.

LTX-2.5 is the first definition this project ships from a GATED repository, and
nothing in the engine handled that: the failure arrived as
``Failed to load unet from huggingface:...: 404 Client Error``, which reads as
"the model is broken" when the real answer is "click accept". This is not
LTX-specific -- it is every family's download path -- so it lives in the loader
base and is pinned here.

The delicate case is 404. The Hub returns it BOTH for a repository that does not
exist AND for one the caller is not entitled to see, deliberately, so that private
repos cannot be enumerated. A message that asserts "accept the licence" on a
mistyped repo id would send someone looking for a licence page that never existed
-- worse than the raw error. So 404 must name both causes, and that is pinned
rather than left to whoever edits the string next.
"""

from __future__ import annotations

import pytest

from app.engine.core.pipeline.loader_base import hub_access_message

REPO = "huggingface:Lightricks/LTX-2.5-Diffusers"


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class GatedRepoError(Exception):
    """Shaped like huggingface_hub's, without importing it."""

    def __init__(self, status: int = 403) -> None:
        super().__init__("Access to model is restricted")
        self.response = _Response(status)


class HfHubHTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"{status} Client Error")
        self.response = _Response(status)


class RepositoryNotFoundError(HfHubHTTPError):
    pass


def test_a_gated_repo_names_the_licence_and_the_repo():
    msg = hub_access_message(GatedRepoError(), "unet", REPO)
    assert msg is not None
    assert "huggingface.co/Lightricks/LTX-2.5-Diffusers" in msg
    assert "licence" in msg
    assert "unet" in msg


def test_a_401_asks_for_a_token_not_a_licence():
    msg = hub_access_message(HfHubHTTPError(401), "text_encoder", REPO)
    assert msg is not None
    assert "token" in msg
    assert "accept the model licence" not in msg, (
        "401 means the request was unauthenticated, not that a licence is unsigned; "
        "sending the user to a licence page does not fix a missing token"
    )


def test_a_404_refuses_to_guess_between_missing_and_gated():
    """The whole point of the ambiguity handling."""
    msg = hub_access_message(RepositoryNotFoundError(404), "vae", REPO)
    assert msg is not None
    assert "spelling" in msg, "404 must admit the id might simply be wrong"
    assert "accept the model licence" in msg, "404 must also admit it might be gated"


@pytest.mark.parametrize(
    "exc",
    [
        OSError("No such file or directory: 'config.json'"),
        ValueError("Unrecognized configuration class"),
        RuntimeError("CUDA out of memory"),
        HfHubHTTPError(500),
    ],
    ids=["missing-file", "bad-config", "oom", "server-error"],
)
def test_it_stays_silent_on_failures_that_are_not_about_access(exc):
    """Returning None keeps the ORIGINAL error, which is the right one here.

    A helpful-sounding message on an out-of-memory crash is not a smaller bug
    than an unhelpful one on a gated repo -- it is a larger one, because it sends
    the reader to Hugging Face to fix their GPU.
    """
    assert hub_access_message(exc, "unet", REPO) is None


class _HostileResponse:
    """A response whose status_code getter raises, like a proxy or a half-built one."""

    @property
    def status_code(self):
        raise RuntimeError("connection object is not initialised")


class _HostileError(Exception):
    def __init__(self) -> None:
        super().__init__("something went wrong downloading")
        self.response = _HostileResponse()


def test_it_never_raises_out_of_an_exception_handler():
    """Found by adversarial review, and the severity is in WHERE this runs.

    ``hub_access_message`` is called from inside an ``except`` block on every
    family's component-load path. ``getattr(obj, name, default)`` swallows
    AttributeError and nothing else, so a ``status_code`` property that raises
    propagates out of the error reporter and replaces the real component failure
    with a traceback about the reporter itself -- the actual cause is then gone.
    Its own docstring promised this could not happen; that promise is now a test.
    """
    assert hub_access_message(_HostileError(), "unet", REPO) is None


def test_a_non_integer_status_cannot_reach_the_message():
    """A Mock or a stringly-typed status must not be formatted into user text."""

    class _Odd(Exception):
        def __init__(self) -> None:
            super().__init__("odd")
            self.response = type("R", (), {"status_code": "403"})()

    # Not an int -> not a recognised access failure -> original error preserved.
    assert hub_access_message(_Odd(), "unet", REPO) is None


def test_the_message_never_blames_the_user_s_data():
    """The failure arrives mid-job, right after dataset work, and reads like its fault."""
    for exc in (GatedRepoError(), HfHubHTTPError(401), RepositoryNotFoundError(404)):
        assert "Nothing is wrong with your dataset" in hub_access_message(exc, "unet", REPO)
