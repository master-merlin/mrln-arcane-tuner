"""Values for pip's environment variables that survive a path with a space.

pip reads its list-valued settings from the environment (``PIP_FIND_LINKS``,
``PIP_EXTRA_INDEX_URL``, ``PIP_TRUSTED_HOST`` ...) by splitting the value at
whitespace. A bare directory whose path has a space so arrives as two locations,
each ignored with ``Location ... is ignored``, and a test that serves its wheels
from that directory sees pip find nothing. A ``file:`` URI escapes the space as
``%20`` and reaches pip as one location.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["find_links_value"]


def find_links_value(directory: Path) -> str:
    """The value for ``PIP_FIND_LINKS`` that names ``directory`` as one location: its file URI."""
    return Path(directory).resolve().as_uri()
