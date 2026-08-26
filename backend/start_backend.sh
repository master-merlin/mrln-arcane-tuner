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
venv/bin/python -m uvicorn app.main:app \
    --host "${MRLN_BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}"
