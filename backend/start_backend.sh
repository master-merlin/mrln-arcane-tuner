#!/usr/bin/env bash
# Start the MRLN Arcane Tuner backend server
cd "$(dirname "$0")"
source venv/bin/activate
# Invoke uvicorn through the interpreter, not the `uvicorn` console-script
# shim. Those shims hard-code the absolute path of the interpreter that created
# them, so they die if the venv directory is ever renamed or moved (the
# interpreter itself is fine — it derives sys.prefix from its own location).
#
# Binds to THIS MACHINE ONLY by default. It used to be 0.0.0.0, which put an
# unauthenticated server holding your datasets, models and GPU on every network
# you join — including hotel and cafe wifi. To reach it from another machine,
# set a token AND widen the bind:
#   MRLN_AUTH_TOKEN=<long random string> MRLN_BIND_HOST=0.0.0.0 ./start_backend.sh
# Widening without a token is refused at startup, by design.
# The port has ONE producer: port_resolver.py, which every launcher calls. This
# script does NOT parse settings.json — four launchers in three languages each
# reading the same file is four chances to disagree with the app about which
# port it is on, which is the bug this replaced. The shell only carries the
# answer across on --port; the app reads it back out of its own argv.
#
# A failure here is a REFUSAL, not a fallback. Defaulting to 8000 when the
# resolver cannot answer would start a server on a port the settings screen
# denies — the same silent disagreement, restored by the error path.
if ! bindPort="$(venv/bin/python port_resolver.py)"; then
    echo "start_backend: refusing to start — the backend port could not be determined." >&2
    echo "start_backend: the reason is above; fix or delete backend/settings.json." >&2
    exit 1
fi
venv/bin/python -m uvicorn app.main:app \
    --host "${MRLN_BIND_HOST:-127.0.0.1}" --port "$bindPort"
