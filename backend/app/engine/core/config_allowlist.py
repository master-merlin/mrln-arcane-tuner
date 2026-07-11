"""Backend capability allowlist — the single server-side choke point that
silently drops top-level config keys a family's descriptor gates OFF.

Motivation
----------
A training config travels ``API route -> DB -> trainer`` as a plain
``dict[str, Any]`` with no server-side capability enforcement. Capability
gating today lives only in ``_FIELD_RULES`` (see
:mod:`app.engine.core.archetypes`) -> ``field_visibility``, which the *frontend*
reads to strip unsupported fields before submit. A direct API caller, a
template import, or an old DB row can therefore carry a field the target family
does not support (e.g. ``num_frames`` on an image model).

This module mirrors :mod:`app.engine.core.video_contract` / ``edit_validation``
(pure logic; the caller wires it at the ``job_manager`` choke point) but with a
deliberately *permissive* contract: **silently drop, never reject**. A 400 here
would break template-import permissiveness and refuse old configs on edit — so
unsupported keys are popped, forward-compat/vendor keys are left alone, and the
drop is logged (by the caller) rather than raised.

Source of truth
---------------
The descriptor consulted is EXACTLY the one the frontend reads —
``resolve_capabilities(defn)["field_visibility"]`` — a FLAT ``{field_name:
{"supported": bool, "reason"?: str}}`` map. A key is dropped iff it appears in
that map AND is marked unsupported. Consequently:

* runtime/system keys and unknown vendor keys are never in the map -> untouched
  (forward-compat + vendor keys survive; see requirement "unknown keys"),
* the ``EXEMPT_KEYS`` set below is a belt-and-suspenders guard: it short-circuits
  the runtime keys the create/update/resume paths inject *before* the map is
  even consulted, so a future rule that happened to name one of them could never
  strip it.

Nesting
-------
``_FIELD_RULES`` is a flat field-name list with **no path syntax** — a rule
names ``masking_enabled``, not ``datasets[].masking_enabled``. The descriptor
therefore does not "describe nested fields", so this allowlist gates TOP-LEVEL
keys only and never reaches into ``datasets`` (which is exempt). The nested
per-dataset edit-masking gate is owned by ``validate_edit_config`` (a hard
reject at trainer run start); backend and frontend agree on the flat descriptor.
"""

from __future__ import annotations

from typing import Any

from app.engine.core.archetypes import resolve_capabilities

# Runtime/system keys the job pipeline injects or routes on that are NOT gated
# schema fields and must never be dropped. None currently appear in
# ``_FIELD_RULES`` (so the field_visibility lookup would already skip them), but
# exempting them explicitly documents intent and survives a future rule/rename
# collision. Sourced from the create/update/resume paths in job_manager:
#   * job_id                 — injected by create_job / update_job_config
#   * definition_id          — target-model routing key (also a DB column)
#   * plugin_id              — legacy plugin routing key
#   * project_id, lora_name  — DB columns set from the config at create
#   * output_dir             — run output folder
#   * datasets               — nested dataset list (never gated here; see module doc)
#   * resume_from_checkpoint — injected by resume_from_checkpoint()
#   * use_cached_latents /
#     use_cached_embeddings  — injected on resume for cache reuse
EXEMPT_KEYS: frozenset[str] = frozenset(
    {
        "job_id",
        "definition_id",
        "plugin_id",
        "project_id",
        "lora_name",
        "output_dir",
        "datasets",
        "resume_from_checkpoint",
        "use_cached_latents",
        "use_cached_embeddings",
    }
)


def compute_disallowed_keys(config: dict[str, Any], definition) -> list[str]:
    """Return the top-level keys of ``config`` the ``definition``'s descriptor
    gates OFF. Read-only — does not mutate ``config``.

    A key is disallowed iff it is NOT exempt, appears in the family's
    ``field_visibility`` map, and is marked ``supported: False`` there.
    """
    field_visibility = resolve_capabilities(definition)["field_visibility"]
    disallowed: list[str] = []
    for key in list(config.keys()):
        if key in EXEMPT_KEYS:
            continue
        rule = field_visibility.get(key)
        if rule is not None and not rule.get("supported", True):
            disallowed.append(key)
    return disallowed


def apply_capability_allowlist(config: dict[str, Any], definition) -> list[str]:
    """Pop every gated-off top-level key from ``config`` in place.

    Returns the list of dropped keys (empty when nothing was dropped) so the
    caller can log a single INFO line. Never raises on a normal config and never
    rejects — silent-drop only.
    """
    dropped = compute_disallowed_keys(config, definition)
    for key in dropped:
        config.pop(key, None)
    return dropped
