#!/usr/bin/env bash
# Update this MRLN Arcane Tuner checkout and its installed dependencies.
#
#   ./update.sh              pull, then install whatever is out of date
#   ./update.sh --check      report what is out of date, change nothing
#   ./update.sh --no-pull    skip git, just sync deps to this checkout
#   ./update.sh --build      also run a production frontend build
#
# This file only finds a Python and hands over to update.py, where all
# the logic lives. Deliberately thin: this repo already needs five install
# paths to agree with each other, and a launcher that re-implements any of the
# work is a sixth thing to keep in step.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/update.py"

if [ ! -f "$DRIVER" ]; then
    echo "update: cannot find $DRIVER" >&2
    exit 1
fi

# Prefer the project's own venv: it is the interpreter the backend runs on, so
# if it is broken the update should surface that rather than paper over it with
# a system Python that happens to work.
for candidate in \
    "$HERE/backend/venv/bin/python" \
    "$HERE/backend/venv/Scripts/python.exe" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"
do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        exec "$candidate" "$DRIVER" "$@"
    fi
done

echo "update: no Python interpreter found (tried backend/venv, python3, python)." >&2
exit 1
