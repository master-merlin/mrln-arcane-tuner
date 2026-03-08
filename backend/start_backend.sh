#!/usr/bin/env bash
# Start the MRLN Arcane Tuner backend server
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
