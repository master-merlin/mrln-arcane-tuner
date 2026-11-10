# --------------------------------------------------------------------------
# MRLN Arcane Tuner - Windows install script
#
# Optionally creates a virtual environment, installs the torch stack this
# machine can run (torch_stack.py reads the driver and every GPU and picks a
# set from torch-stack.cfg), then installs all remaining dependencies. PEP 508
# markers in requirements.txt automatically handle platform-specific packages.
#
#   install.ps1 [-VenvDir <dir>] [-Yes]
#
# -VenvDir  the venv to create or reuse (default: venv next to this script).
# -Yes      no questions: reuse or create the venv.
# MRLN_TORCH_INDEX=<set> chooses the torch set (cu132, cu126, cpu); it is
# still refused where that set cannot run.
# --------------------------------------------------------------------------
param(
    [string]$VenvDir = "",
    [switch]$Yes
)
$ErrorActionPreference = "Stop"

# Anchored on this script, never on the current directory.
$Here = $PSScriptRoot
$Req = Join-Path $Here "requirements.txt"
$Stack = Join-Path $Here "torch_stack.py"
if (-not $VenvDir) { $VenvDir = Join-Path $Here "venv" }

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Confirm-Step([string]$Question) {
    if ($Yes) { return $true }
    $answer = Read-Host "$Question [Y/n]"
    return -not ($answer -match '^[Nn]$')
}

# -- Virtual environment --------------------------------------------------
# Every install below runs this venv's own interpreter by path. Nothing is
# activated, so an already-active venv cannot redirect the install.

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "[OK] Virtual environment '$VenvDir' already exists." -ForegroundColor Green
    if (-not (Confirm-Step "   Install into it?")) {
        Write-Host "Aborted."
        exit 0
    }
    $Py = $VenvPython
} elseif (Confirm-Step "Create a virtual environment in '$VenvDir'?") {
    Write-Host "[*] Creating virtual environment ..." -ForegroundColor Cyan
    Invoke-Checked "python" @("-m", "venv", $VenvDir)
    $Py = $VenvPython
    Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[!] Skipping venv - installing into the current Python environment." -ForegroundColor Yellow
    $Py = (Get-Command python).Source
}

# -- The torch stack --------------------------------------------------------
# torch_stack.py chooses the set, refuses (exit 2, with the table) before
# installing anything where no set fits, and installs torch, torchvision,
# torchaudio, torchao and triton from it.

Write-Host ""
Invoke-Checked $Py @($Stack, "install", "--python", $Py)

# -- Remaining dependencies -----------------------------------------------
# The bulk install never sees the torch stack (torch_stack.py owns it) nor
# scenedetect/sam3/hpsv2 (installed --no-deps below; see install-deps.sh for
# why each needs it).

Write-Host ""
Write-Host "[+] Installing remaining dependencies ..." -ForegroundColor Cyan
$TmpReq = [System.IO.Path]::GetTempFileName()
try {
    Invoke-Checked $Py @($Stack, "filter", $Req, $TmpReq, "--also", "scenedetect", "sam3", "hpsv2")
    Invoke-Checked $Py @("-m", "pip", "install", "-r", $TmpReq)
} finally {
    Remove-Item $TmpReq -Force -ErrorAction SilentlyContinue
}

function Get-Pin([string]$Name) {
    Get-Content $Req |
        Where-Object { $_ -match "^\s*$Name\s*==" } |
        ForEach-Object { ($_ -replace '#.*$', '') -replace '\s', '' } |
        Select-Object -First 1
}

# scenedetect's declared dependency is the GUI build `opencv-python`, which
# collides with the pinned `opencv-python-headless` (both ship the `cv2`
# module) - install it separately, without its deps.
# sam3's declared ceilings would block the resolve against our pins; its
# import works (test_sam3_imports_cleanly_despite_declared_hub_pin).
# hpsv2 declares its dev deps (`pytest ==7.2.0`, `pytest-split ==0.8.0`) as
# install requirements; it never imports them
# (test_hpsv2_works_under_a_runner_its_metadata_forbids).
foreach ($name in @("scenedetect", "sam3", "hpsv2")) {
    $pin = Get-Pin $name
    if ($pin) {
        Write-Host "[+] Installing $pin (--no-deps) ..." -ForegroundColor Cyan
        Invoke-Checked $Py @("-m", "pip", "install", "--no-deps", $pin)
    }
}

# -- The stack is still the one chosen --------------------------------------

Write-Host ""
Invoke-Checked $Py @($Stack, "verify", "--python", $Py)

Write-Host ""
Write-Host "[OK] Done - all dependencies installed." -ForegroundColor Green
