"""Cutlist parsing — LosslessCut ``.llc`` (JSON5), CSV, and TSV.

``parse_cutlist(data, filename, source_duration_s)`` turns an uploaded cutlist
into a list of :class:`Segment` (start/end seconds + optional label) plus a list
of human-readable warnings for rows that were clamped or dropped. It NEVER
raises on malformed rows — a bad row becomes a warning and is skipped — so the
synchronous ``/cutlist/parse`` endpoint always returns a usable result.

Return shape: :class:`CutlistResult` ``{segments, warnings, format}``.

Format detection is by extension:
  * ``.llc``              → JSON5 (LosslessCut project file)
  * ``.csv`` / ``.tsv``   → delimited rows ``start[,end][,label]``
  * anything else         → best-effort: try JSON5, else delimited.

LosslessCut stores its cuts under a ``cutSegments`` array of
``{start, end, name}``. Key names and the presence of ``end`` have drifted
across LosslessCut versions, so parsing is defensive: a missing/empty ``end``
falls back to ``source_duration_s``; a missing ``name`` becomes ``None``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single clip span within the source video (seconds)."""

    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    label: str | None = None


class CutlistResult(BaseModel):
    """Result of parsing a cutlist: usable segments + collected warnings."""

    segments: list[Segment]
    warnings: list[str]
    format: str


# Tolerated key spellings across LosslessCut / variant exporters.
_START_KEYS = ("start", "start_s", "startTime", "from")
_END_KEYS = ("end", "end_s", "endTime", "to")
_NAME_KEYS = ("name", "label", "tag")
_SEGMENTS_KEYS = ("cutSegments", "segments", "cuts")


def _coerce_float(value, default: float | None = None) -> float | None:
    """Best-effort float coercion. Empty/``None``/unparseable → ``default``."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_llc(data: bytes) -> list[dict]:
    """Parse an ``.llc`` body into a list of raw ``{start,end,name}`` dicts.

    Tries strict ``json.loads`` first (most ``.llc`` files are valid JSON), then
    falls back to ``json5.loads`` for the relaxed dialect (trailing commas,
    comments, unquoted keys) some exports use.
    """
    text = data.decode("utf-8-sig", errors="replace")
    obj = None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            import json5

            obj = json5.loads(text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not parse .llc as JSON/JSON5: {exc}") from exc

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        raw = _first_key(obj, _SEGMENTS_KEYS)
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
    return []


def _parse_delimited(data: bytes, delimiter: str | None) -> list[list[str]]:
    """Parse CSV/TSV into a list of string rows.

    ``delimiter=None`` sniffs comma vs tab from the body (defaults to comma).
    A header row whose first cell is non-numeric is skipped.
    """
    text = data.decode("utf-8-sig", errors="replace")
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(text[:2048], delimiters=",\t;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[list[str]] = []
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        rows.append(row)

    # Drop a header row (first cell not a number).
    if rows and _coerce_float(rows[0][0]) is None:
        rows = rows[1:]
    return rows


def _build_segments(
    raw_rows: list,
    source_duration_s: float,
    *,
    is_llc: bool,
) -> tuple[list[Segment], list[str]]:
    """Clamp + validate raw rows into Segments, collecting warnings."""
    segments: list[Segment] = []
    warnings: list[str] = []
    dur = (
        float(source_duration_s)
        if source_duration_s and source_duration_s > 0
        else None
    )

    for idx, row in enumerate(raw_rows):
        if is_llc:
            start = _coerce_float(_first_key(row, _START_KEYS), 0.0)
            end = _coerce_float(_first_key(row, _END_KEYS), None)
            name = _first_key(row, _NAME_KEYS)
            label = str(name) if name not in (None, "") else None
        else:
            start = _coerce_float(row[0], None) if len(row) >= 1 else None
            end = _coerce_float(row[1], None) if len(row) >= 2 else None
            label = None
            if len(row) >= 3 and str(row[2]).strip():
                label = str(row[2]).strip()

        if start is None:
            warnings.append(f"row {idx + 1}: missing/invalid start — skipped")
            continue
        # Missing end → run to source duration (best-effort; needs a known dur).
        if end is None:
            if dur is None:
                warnings.append(
                    f"row {idx + 1}: missing end and source duration unknown — skipped"
                )
                continue
            end = dur

        # Clamp to [0, duration].
        start = max(0.0, start)
        end = max(0.0, end)
        if dur is not None:
            start = min(start, dur)
            end = min(end, dur)

        if end - start <= 0:
            warnings.append(
                f"row {idx + 1}: zero/negative length "
                f"(start={start:.3f}, end={end:.3f}) — skipped"
            )
            continue

        segments.append(Segment(start_s=start, end_s=end, label=label))

    return segments, warnings


def parse_cutlist(
    data: bytes, filename: str, source_duration_s: float
) -> CutlistResult:
    """Parse an uploaded cutlist body into segments + warnings.

    Args:
        data: Raw file bytes.
        filename: Original filename — its extension selects the parser.
        source_duration_s: Clip duration, used to clamp ranges and fill in a
            missing segment end. ``0``/unknown disables clamping but then any
            row with no explicit end is dropped (with a warning).

    Returns:
        :class:`CutlistResult` with ``segments`` (clamped, positive-length),
        ``warnings`` (one per dropped/clamped row), and ``format`` (``"llc"``,
        ``"csv"``, ``"tsv"``, or ``"unknown"``).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".llc":
        fmt = "llc"
        try:
            raw_rows = _parse_llc(data)
        except ValueError as exc:
            return CutlistResult(segments=[], warnings=[str(exc)], format=fmt)
        segments, warnings = _build_segments(raw_rows, source_duration_s, is_llc=True)
        return CutlistResult(segments=segments, warnings=warnings, format=fmt)

    if ext in (".csv", ".tsv"):
        fmt = "csv" if ext == ".csv" else "tsv"
        delimiter = "," if ext == ".csv" else "\t"
        try:
            raw_rows = _parse_delimited(data, delimiter)
        except Exception as exc:  # noqa: BLE001
            return CutlistResult(
                segments=[], warnings=[f"could not parse {fmt}: {exc}"], format=fmt
            )
        segments, warnings = _build_segments(raw_rows, source_duration_s, is_llc=False)
        return CutlistResult(segments=segments, warnings=warnings, format=fmt)

    # Unknown extension — try JSON5 first, then delimited.
    try:
        raw_rows = _parse_llc(data)
        if raw_rows:
            segments, warnings = _build_segments(
                raw_rows, source_duration_s, is_llc=True
            )
            return CutlistResult(segments=segments, warnings=warnings, format="llc")
    except ValueError:
        pass

    raw_rows = _parse_delimited(data, None)
    segments, warnings = _build_segments(raw_rows, source_duration_s, is_llc=False)
    return CutlistResult(segments=segments, warnings=warnings, format="unknown")
