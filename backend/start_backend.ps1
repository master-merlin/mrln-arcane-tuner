# Start the MRLN Arcane Tuner backend server
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
# Invoke uvicorn through the interpreter, not the `uvicorn.exe` console-script
# shim. Those shims hard-code the absolute path of the interpreter that created
# them, so they die silently if the venv directory is ever renamed or moved
# (`python.exe` itself is fine — it derives sys.prefix from its own location).
#
# Binds to THIS MACHINE ONLY by default. It used to be 0.0.0.0, which put an
# unauthenticated server holding your datasets, models and GPU on every network
# you join — including hotel and cafe wifi. To reach it from another machine,
# set a token AND widen the bind:
#   $env:MRLN_AUTH_TOKEN = "<long random string>"
#   $env:MRLN_BIND_HOST  = "0.0.0.0"
# Widening without a token is refused at startup, by design.
$bindHost = if ($env:MRLN_BIND_HOST) { $env:MRLN_BIND_HOST } else { "127.0.0.1" }
$bindPort = if ($env:PORT) { $env:PORT } else { "8000" }
& .\venv\Scripts\python.exe -m uvicorn app.main:app --host $bindHost --port $bindPort
