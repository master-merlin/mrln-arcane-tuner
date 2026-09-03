# Start the MRLN Arcane Tuner backend server
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
# Invoke uvicorn through the interpreter, not the `uvicorn.exe` console-script
# shim. Those shims hard-code the absolute path of the interpreter that created
# them, so they die silently if the venv directory is ever renamed or moved
# (`python.exe` itself is fine - it derives sys.prefix from its own location).
#
# Binds to THIS MACHINE ONLY by default. It used to be 0.0.0.0, which put an
# unauthenticated server holding your datasets, models and GPU on every network
# you join - including hotel and cafe wifi. To reach it from another machine,
# set a token AND widen the bind:
#   $env:MRLN_AUTH_TOKEN = "<long random string>"
#   $env:MRLN_BIND_HOST  = "0.0.0.0"
# Widening without a token is refused at startup, by design.
$bindHost = if ($env:MRLN_BIND_HOST) { $env:MRLN_BIND_HOST } else { "127.0.0.1" }

# The port has ONE producer: port_resolver.py, which every launcher calls. This
# script does NOT parse settings.json - four launchers in three languages each
# reading the same file is four chances to disagree with the app about which
# port it is on, which is the bug this replaced. The shell only carries the
# answer across on --port; the app reads it back out of its own argv.
#
# A failure here is a REFUSAL, not a fallback. Defaulting to 8000 when the
# resolver cannot answer would start a server on a port the settings screen
# denies - the same silent disagreement, restored by the error path.
# No `| Select-Object -First 1` here, though one line is all that is expected:
# Select-Object stops the upstream pipeline, which kills the native command and
# leaves $LASTEXITCODE reporting that rather than the resolver's own verdict -
# the launcher then refuses on a perfectly good settings file. Found by a test
# that runs this script, not by reading it.
$bindPort = (& .\venv\Scripts\python.exe port_resolver.py)
if ($LASTEXITCODE -ne 0) {
    Write-Host "start_backend: refusing to start - the backend port could not be determined."
    Write-Host "start_backend: the reason is above; fix or delete backend\settings.json."
    exit 1
}
$bindPort = "$bindPort".Trim()
if (-not $bindPort) {
    Write-Host "start_backend: refusing to start - the port resolver returned nothing."
    exit 1
}
& .\venv\Scripts\python.exe -m uvicorn app.main:app --host $bindHost --port $bindPort
