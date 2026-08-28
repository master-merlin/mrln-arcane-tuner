# Update this MRLN Arcane Tuner checkout and its installed dependencies.
#
#   .\update.ps1              pull, then install whatever is out of date
#   .\update.ps1 --check      report what is out of date, change nothing
#   .\update.ps1 --no-pull    skip git, just sync deps to this checkout
#   .\update.ps1 --build      also run a production frontend build
#
# This file only finds a Python and hands over to update.py, where all
# the logic lives. Deliberately thin: this repo already needs five install
# paths to agree with each other, and a launcher that re-implements any of the
# work is a sixth thing to keep in step.

$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Driver = Join-Path $Here 'update.py'

if (-not (Test-Path $Driver)) {
    Write-Error "update: cannot find $Driver"
    exit 1
}

# Prefer the project's own venv: it is the interpreter the backend runs on, so
# if it is broken the update should surface that rather than paper over it with
# a system Python that happens to work.
$Candidates = @(
    (Join-Path $Here 'backend\venv\Scripts\python.exe'),
    (Join-Path $Here 'backend\venv\bin\python')
)
$Python = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Python = $cmd.Source }
}
if (-not $Python) {
    Write-Error "update: no Python interpreter found (tried backend\venv, python)."
    exit 1
}

& $Python $Driver @args
exit $LASTEXITCODE
