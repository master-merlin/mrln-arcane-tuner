#!/usr/bin/env bash
# Start the MRLN Arcane Tuner backend server
cd "$(dirname "$0")"
source venv/bin/activate
# Invoke uvicorn through the interpreter, not the `uvicorn` console-script
# shim. Those shims hard-code the absolute path of the interpreter that created
# them, so they die if the venv directory is ever renamed or moved (the
# interpreter itself is fine — it derives sys.prefix from its own location).
venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
