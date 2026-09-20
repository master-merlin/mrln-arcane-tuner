"""POSITIVE CONTROL for ``test_minimax_h3_convention.py::test_sampler_source_has_no_autocast``.

NOT production code; never imported by the engine. Carries the shapes the
scoped autocast guard must classify: a BARE ``torch.autocast(`` call (an
offender), one entered with ``enabled=False`` (allowed), a ``torch.amp``
spelling (an offender), and prose that names ``torch.autocast(`` in a comment
and a docstring (ignored). Plan row 2.10 (memory ``autocast-sampler-collapse-
gotcha``: an autocast wrapper around the DiT forward collapsed sampling to the
conditional mean once already).
"""

from __future__ import annotations

import torch


def bare_autocast() -> None:
    # torch.autocast( mentioned in a comment is NOT a site.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pass


def disabled_autocast() -> None:
    """Prose mentioning torch.autocast( is not a site either."""
    with torch.autocast("cuda", enabled=False):
        pass


def amp_spelling() -> None:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pass
