"""Render the training form's "?" help text into `docs/training-guide.md`.

The user asked for the "?" texts specifically. Those live in
`frontend/public/config_help.json` (a static asset, keyed exactly the way
`ConfigHelpService.getConfigHelp()` / `getHelpTip(key)` key it —
`training-dynamic-config.ts:972-974`) — a separate, hand-authored file, NOT
generated from the schema's own `description=` (see the LANE-72 report: the
first version of this tool rendered the schema description alone before this
was found). Each field now renders:

1. the config_help **tip**, verbatim, when one exists for that key;
2. the schema **description**, appended only when it adds information the
   tip does not already say (a plain substring check);
3. **default** and **range/choices**, always from the schema — config_help
   carries none of that.

A field with no config_help entry falls back to the schema description alone
and is listed by `--report` (so the gap is visible, not silently absorbed).
`--report` also lists fields where the two texts *might* disagree — a cheap
heuristic (distinct numbers mentioned in both) that flags candidates for a
human to judge.

Renders one block per schema group (the same eight groups + labels
`training-dynamic-config.ts::formatGroupName` shows in the form) into
`docs/training-guide.md`, between machine-owned markers:

    <!-- schema-explainers:<GROUP> start -->
    ...generated...
    <!-- schema-explainers:<GROUP> end -->

`--write` renders and replaces every marked block (creating the containing
"## Field reference" section and its per-group markers on first run).
`--check` renders in memory and exits 1 if the file's blocks differ from a
fresh render of EITHER source (schema or config_help.json) — the guard: the
guide drifts from either source, the tool says so.

Exit codes: 0 clean / written, 1 findings (stale blocks, missing anchor),
3 could not check (not in a git checkout, an input missing, the backend not
importable).

The checkout is `git rev-parse --show-toplevel` of the cwd -- the one the tool
RUNS in, so a worktree stays a worktree (LANE-72: a root derived from a
junction-followed `__file__` once wrote into the primary tree).
Imports `BaseTrainingConfig` directly, the same class the
`/plugins/{id}/schema` route calls `.model_json_schema()` on
(`backend/app/api/training/plugin_routes.py`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

NAME = "docs_schema_explainers"
EXIT_UNVERIFIED = 3


class Unverified(Exception):
    """The check could not run: say why, never report a finding about it."""


def _repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Unverified(f"not inside a git checkout: {exc}") from None
    return Path(out)


try:
    _REPO_ROOT = _repo_root()
except Unverified as _exc:
    if __name__ != "__main__":
        raise
    print(f"{NAME}: UNVERIFIED: {_exc}")
    sys.exit(EXIT_UNVERIFIED)
_BACKEND = _REPO_ROOT / "backend"
_GUIDE = _REPO_ROOT / "docs" / "training-guide.md"
_CONFIG_HELP = _REPO_ROOT / "frontend" / "public" / "config_help.json"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Group codes, restricted to the groups the user asked for by name
# (LANE-72) — MODEL_SELECTION, CONCEPTS and the audio-only catch-all are
# documented in prose elsewhere in the guide, not as field-reference blocks.
# This is NOT the form's own TOC order (General, Training Dynamics, Video,
# LoRA, Optimizer, Expert, Sampling, Advanced Engine —
# `training-dynamic-config.ts:1591`); within each group the render is
# alphabetical by key, not the form's field order either. Both facts are
# stated in `_SECTION_INTRO` so the guide never implies it mirrors the
# screen's layout. Labels mirror `training-dynamic-config.ts::formatGroupName()`
# verbatim; if that map ever adds/renames a group this list goes stale
# silently.
_GROUP_ORDER = [
    "BASE",
    "STRATEGY",
    "NETWORK",
    "OPTIMIZER",
    "OPTIMIZER_EXPERT",
    "ENGINE",
    "VIDEO",
    "SAMPLING",
]
_GROUP_LABELS = {
    "BASE": "General Settings",
    "STRATEGY": "Training Dynamics",
    "NETWORK": "LoRA Parameters",
    "OPTIMIZER": "Optimizer Settings",
    "OPTIMIZER_EXPERT": "Expert Features",
    "ENGINE": "Advanced Engine",
    "VIDEO": "Video Settings",
    "SAMPLING": "Sampling",
}

_SECTION_HEADING = "## Field reference (generated from the schema)"
_SECTION_INTRO = (
    "Generated from the training schema and the in-app help texts, and "
    "regenerated whenever either changes. Where a field has a `?` icon in "
    "the form, the text below is that icon's tip, verbatim. Fields without "
    "one (currently 25, mostly Video Settings) are marked \"From the "
    "schema:\" and show the schema's own description instead, which the "
    "form does not display anywhere. Headings below are the form's own "
    "group names; fields within a group are alphabetical, not the form's "
    "own field order."
)

_WRITE_ADVICE = "Run: python backend/scripts/docs_schema_explainers.py --write"

_MARKER_RE_TEMPLATE = (
    r"<!-- schema-explainers:{group} start -->\n(.*?)"
    r"<!-- schema-explainers:{group} end -->\n"
)


def _load_schema() -> dict:
    # imported at the call site: the backend absent / not importable = UNVERIFIED, named
    try:
        base = importlib.import_module("app.engine.models.base")
    except ImportError as exc:
        raise Unverified(f"cannot import app.engine.models.base: {exc}") from None
    return base.BaseTrainingConfig.model_json_schema()


def _load_config_help() -> dict:
    """The `?` tooltip source, keyed exactly the way `getHelpTip(key)` keys it.

    Dotted keys (`adaptive_targeting.warmup_pct`, …) address fields inside a
    sub-config namespace (`AdaptiveTargetingConfig`) that is not itself a
    property of `BaseTrainingConfig` — its host field is the opaque
    `adaptive_targeting_config` dict (`ui_type: adaptive_targeting`, hidden
    from the plain field list). Those dotted entries have nothing to attach
    to in this tool's per-group rendering and are left out of both the
    rendered blocks and the `--report` gap list on purpose, not by oversight.
    """
    # an absent tooltip file is not "no tips" (every field would be reported tip-less): UNVERIFIED
    if not _CONFIG_HELP.is_file():
        raise Unverified(f"config_help.json missing: {_CONFIG_HELP.as_posix()}")
    try:
        return json.loads(_CONFIG_HELP.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Unverified(f"damaged config {_CONFIG_HELP.as_posix()}: {exc}") from None


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_disagree(tip: str, description: str) -> bool:
    """Cheap heuristic: both texts state a number, and the sets don't overlap.

    Flags a CANDIDATE contradiction for `--report` — a human judges it, this
    only surfaces it. E.g. two texts naming different defaults or units for
    the same field would trip this; two texts that simply mention different
    UNRELATED numbers (rare, given field descriptions are one sentence) would
    also trip it, which is why this is a report line, not an error.
    """
    tnums = set(_NUM_RE.findall(tip))
    dnums = set(_NUM_RE.findall(description))
    return bool(tnums) and bool(dnums) and tnums.isdisjoint(dnums)


def _format_choices(prop: dict) -> str | None:
    enum = prop.get("enum")
    if not enum:
        return None
    return "choices: " + ", ".join(f"`{v}`" for v in enum)


def _format_range(prop: dict) -> str | None:
    lo, hi, step = prop.get("min"), prop.get("max"), prop.get("step")
    if lo is None and hi is None:
        return None
    parts = []
    if lo is not None and hi is not None:
        # "to", not an en dash: `-1.0–-0.1` reads as a typo of a single
        # negative number, not a range (adversary LANE-72 round 1, MINOR 12c).
        parts.append(f"range {lo} to {hi}")
    elif lo is not None:
        parts.append(f"min {lo}")
    elif hi is not None:
        parts.append(f"max {hi}")
    if step is not None:
        parts.append(f"step {step}")
    return ", ".join(parts)


def _format_default(prop: dict, key: str, required: set[str]) -> str:
    if "default" not in prop:
        return "required, no default" if key in required else "none"
    default = prop["default"]
    if default == "":
        return "`\"\"` (empty)"
    if isinstance(default, dict) and not default:
        return "`{}`"
    return f"`{default}`"


def render_group(
    group_code: str, schema: dict, config_help: dict
) -> tuple[str, list[str], list[str]]:
    """Render one group's marker-wrapped block body (no marker lines).

    Returns (body, keys_without_config_help, keys_with_candidate_contradiction).
    """
    props: dict = schema.get("properties", {})
    required = set(schema.get("required", []))
    rows: list[tuple[str, str]] = []
    no_tip: list[str] = []
    contradictions: list[str] = []
    for key, prop in props.items():
        if prop.get("group") != group_code:
            continue
        if prop.get("hidden"):
            continue
        title = prop.get("title", key.replace("_", " ").title())
        description = (prop.get("description") or "").strip()
        default = _format_default(prop, key, required)
        extras = [e for e in (_format_range(prop), _format_choices(prop)) if e]

        tip = (config_help.get(key, {}) or {}).get("tip", "").strip()
        if tip:
            if _numbers_disagree(tip, description):
                contradictions.append(key)
            body = tip
            # The schema description earns its place only when it says
            # something the tip does not already say (plain substring check
            # — cheap, and false negatives just mean an occasional harmless
            # repeat, never a dropped fact).
            if description and description.lower() not in tip.lower():
                body += f" *(schema: {description})*"
        else:
            # No `?` icon exists for this field (`hasHelp()` is false in the
            # form) — say so plainly rather than implying an icon it doesn't
            # have (adversary LANE-72 round 1, MAJOR 3).
            no_tip.append(key)
            body = f"From the schema: {description}" if description else "From the schema: (no description)"

        # Only append a terminator when the body doesn't already end with
        # one — config_help tips are free text and some already end in a
        # period, question mark or similar (adversary MINOR 12b).
        if body and body[-1] not in ".?!":
            body += "."
        line = f"- **{title}** (`{key}`) — {body} Default {default}"
        if extras:
            line += " (" + "; ".join(extras) + ")"
        line += "."
        rows.append((key, line))
    rows.sort(key=lambda kv: kv[0])
    body_text = (
        "_No fields in this group._\n"
        if not rows
        else "\n".join(line for _, line in rows) + "\n"
    )
    return body_text, no_tip, contradictions


def render_all(
    schema: dict, config_help: dict
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    blocks: dict[str, str] = {}
    no_tip: dict[str, list[str]] = {}
    contradictions: dict[str, list[str]] = {}
    for group in _GROUP_ORDER:
        body, gaps, conflicts = render_group(group, schema, config_help)
        blocks[group] = body
        if gaps:
            no_tip[group] = gaps
        if conflicts:
            contradictions[group] = conflicts
    return blocks, no_tip, contradictions


_INTRO_KEY = "INTRO"


def _with_intro(blocks: dict[str, str]) -> dict[str, str]:
    """`blocks` plus the static intro paragraph, keyed like a group so the
    same replace/check loop covers it — a constant-text change (like the
    LANE-72 round-1 rewrite of `_SECTION_INTRO`) must also trip `--check`,
    not just a schema/config_help drift."""
    combined = dict(blocks)
    combined[_INTRO_KEY] = _SECTION_INTRO + "\n"
    return combined


def _build_section(blocks: dict[str, str]) -> str:
    parts = [
        _SECTION_HEADING,
        "",
        f"<!-- schema-explainers:{_INTRO_KEY} start -->",
        _SECTION_INTRO,
        f"<!-- schema-explainers:{_INTRO_KEY} end -->",
        "",
    ]
    for group in _GROUP_ORDER:
        parts.append(f"### {_GROUP_LABELS[group]}")
        parts.append("")
        parts.append(f"<!-- schema-explainers:{group} start -->")
        parts.append(blocks[group].rstrip("\n"))
        parts.append(f"<!-- schema-explainers:{group} end -->")
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


def apply(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Replace (or insert) every group's marked block. Returns (new_text, changed_groups)."""
    changed: list[str] = []
    out = text
    intro_marker = f"<!-- schema-explainers:{_INTRO_KEY} start -->"
    if _SECTION_HEADING not in out or intro_marker not in out:
        # Full (re)build: either there is no section yet, or there is one
        # from before the intro itself was marker-wrapped (a constant-text
        # rewrite that predates this migration) — replace the whole thing
        # rather than trying to patch an unmarked paragraph.
        section_re = re.compile(
            re.escape(_SECTION_HEADING) + r".*?(?=\n## |\Z)", re.DOTALL
        )
        insertion = _build_section(blocks)
        if section_re.search(out):
            out = section_re.sub(insertion.rstrip("\n") + "\n", out, count=1)
        else:
            anchor = "## Where things live"
            if anchor in out:
                out = out.replace(anchor, insertion + "\n" + anchor, 1)
            else:
                out = out.rstrip("\n") + "\n\n" + insertion
        changed.extend([_INTRO_KEY, *_GROUP_ORDER])
        return out, changed

    all_blocks = _with_intro(blocks)
    for group in [_INTRO_KEY, *_GROUP_ORDER]:
        pattern = re.compile(
            _MARKER_RE_TEMPLATE.format(group=re.escape(group)), re.DOTALL
        )
        replacement = (
            f"<!-- schema-explainers:{group} start -->\n"
            f"{all_blocks[group]}"
            f"<!-- schema-explainers:{group} end -->\n"
        )
        new_out, n = pattern.subn(replacement, out)
        if n == 0:
            raise SystemExit(
                f"docs_schema_explainers: markers for group {group!r} not found "
                f"in {_GUIDE} — the '## Field reference' heading exists but this "
                f"group's marker pair is missing or malformed."
            )
        if new_out != out:
            changed.append(group)
        out = new_out
    return out, changed


def check(text: str, blocks: dict[str, str]) -> list[str]:
    """Return the list of groups (plus INTRO) whose rendered block differs
    from the file's."""
    stale: list[str] = []
    intro_marker = f"<!-- schema-explainers:{_INTRO_KEY} start -->"
    if _SECTION_HEADING not in text or intro_marker not in text:
        return [_INTRO_KEY, *_GROUP_ORDER]
    all_blocks = _with_intro(blocks)
    for group in [_INTRO_KEY, *_GROUP_ORDER]:
        pattern = re.compile(
            _MARKER_RE_TEMPLATE.format(group=re.escape(group)), re.DOTALL
        )
        m = pattern.search(text)
        if not m or m.group(1) != all_blocks[group]:
            stale.append(group)
    return stale


def _print_report(no_tip: dict[str, list[str]], contradictions: dict[str, list[str]]) -> None:
    total_gaps = sum(len(v) for v in no_tip.values())
    print(f"docs_schema_explainers --report: {total_gaps} field(s) with no config_help entry")
    for group in _GROUP_ORDER:
        keys = no_tip.get(group)
        if keys:
            print(f"  {group}: {', '.join(keys)}")
    total_conflicts = sum(len(v) for v in contradictions.values())
    print(f"docs_schema_explainers --report: {total_conflicts} candidate contradiction(s)")
    for group in _GROUP_ORDER:
        keys = contradictions.get(group)
        if keys:
            print(f"  {group}: {', '.join(keys)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="render and write the file")
    mode.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    mode.add_argument(
        "--report", action="store_true",
        help="list fields with no config_help entry and candidate contradictions",
    )
    args = parser.parse_args()
    try:
        return body(args)
    except Unverified as exc:
        print(f"{NAME}: UNVERIFIED: {exc}")
        return EXIT_UNVERIFIED


def body(args: argparse.Namespace) -> int:
    # inputs first (cheap, named when missing), the backend import last
    if not args.report and not _GUIDE.is_file():
        raise Unverified(f"training guide missing: {_GUIDE.as_posix()}")
    config_help = _load_config_help()
    schema = _load_schema()
    blocks, no_tip, contradictions = render_all(schema, config_help)

    if args.report:
        _print_report(no_tip, contradictions)
        return 0

    text = _GUIDE.read_text(encoding="utf-8")

    if args.check:
        stale = check(text, blocks)
        if stale:
            # Distinguish "I could not FIND the section" from "the blocks are
            # stale". They are different facts and they have different fixes,
            # and conflating them cost a session on 2026-09-06: a docs pass
            # renamed the heading — reasonably, on a sentence-case styling
            # instruction — and this tool reported all NINE groups stale. The
            # blocks were byte-identical. The advice it printed (`--write`)
            # would have rebuilt the section rather than restoring one line.
            # A checker that cannot locate its subject must say THAT, not
            # report a finding about the subject it never found.
            if _SECTION_HEADING not in text:
                print(
                    f"docs_schema_explainers --check: section heading not found in "
                    f"{_GUIDE.name}. Expected this line, verbatim:\n"
                    f"    {_SECTION_HEADING}\n"
                    "The generated blocks below it may be perfectly current — this "
                    "heading is a MACHINE ANCHOR, not prose. Either restore the "
                    "wording, or change _SECTION_HEADING in this tool in the same "
                    "commit. Do not run --write to 'fix' this."
                )
                return 1
            print(
                "docs_schema_explainers --check: stale/missing blocks for groups: "
                + ", ".join(stale)
            )
            print(_WRITE_ADVICE)
            return 1
        print("docs_schema_explainers --check: clean")
        return 0

    new_text, changed = apply(text, blocks)
    if new_text != text:
        _GUIDE.write_text(new_text, encoding="utf-8")
        print(
            "docs_schema_explainers --write: updated groups: "
            + (", ".join(changed) if changed else "(none)")
        )
    else:
        print("docs_schema_explainers --write: already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
