"""Metadata every saver must write identically, in one place.

Each family saver builds its own ``ss_*`` map. That is fine for the keys a
family genuinely owns, but a value with ONE correct spelling across all of
them drifts the moment a saver is copied — ltx2 shipped without the trigger
word at all because its copy predated that line. Anything in here is imported,
never re-typed.
"""

from __future__ import annotations

from typing import Any

#: Kohya's conventional home for the trigger word. CivitAI and most LoRA
#: browsers read this one, so it stays even though it is a free-text comment
#: field being used to carry a specific value.
TRIGGER_COMMENT_KEY = "ss_training_comment"

#: The field the Stability ModelSpec defines for exactly this, mirrored by
#: kohya's own ``library/sai_model_spec.py``. A tool that follows the spec
#: looks here and finds nothing if only the comment is written.
TRIGGER_PHRASE_KEY = "modelspec.trigger_phrase"


def trigger_metadata(config: Any) -> dict[str, str]:
    """Both keys a downstream tool might read the trigger word from.

    Returns an EMPTY dict when the run has no trigger word. Writing an empty
    string would advertise a trigger phrase that does not exist, and a reader
    cannot tell "" apart from "the author left it blank on purpose" — an absent
    key is unambiguous.
    """
    if not isinstance(config, dict):
        return {}
    trigger = config.get("global_triggerword")
    if trigger is None:
        return {}
    trigger = str(trigger).strip()
    if not trigger:
        return {}
    return {TRIGGER_COMMENT_KEY: trigger, TRIGGER_PHRASE_KEY: trigger}
