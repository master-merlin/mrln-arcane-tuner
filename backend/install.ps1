# --------------------------------------------------------------------------
# MRLN Arcane Tuner - Windows install script
#
# Optionally creates a virtual environment, installs PyTorch with CUDA 13.0
# from the official PyTorch wheel index, then installs all remaining
# dependencies. PEP 508 markers in requirements.txt automatically handle
# platform-specific packages.
# --------------------------------------------------------------------------

$VenvDir = "venv"

# -- Virtual environment --------------------------------------------------

if (Test-Path $VenvDir) {
    Write-Host "[OK] Virtual environment '$VenvDir' already exists." -ForegroundColor Green
    $answer = Read-Host "   Activate it and continue? [Y/n]"
    if ($answer -eq '' ) { $answer = 'Y' }
    if ($answer -match '^[Nn]$') {
        Write-Host "Aborted."
        exit 0
    }
    & "$VenvDir\Scripts\Activate.ps1"
} else {
    $answer = Read-Host "Create a virtual environment in '.\$VenvDir'? [Y/n]"
    if ($answer -eq '' ) { $answer = 'Y' }
    if ($answer -match '^[Nn]$') {
        Write-Host "[!] Skipping venv - installing into current Python environment." -ForegroundColor Yellow
    } else {
        Write-Host "[*] Creating virtual environment ..." -ForegroundColor Cyan
        python -m venv $VenvDir
        & "$VenvDir\Scripts\Activate.ps1"
        pip install --upgrade pip
        Write-Host "[OK] Virtual environment created and activated." -ForegroundColor Green
    }
}

# -- PyTorch (CUDA 13.0, split stack) -------------------------------------

Write-Host ""
Write-Host "[*] Installing PyTorch 2.12.1 + torchvision 0.27.1 (CUDA 13.0) ..." -ForegroundColor Cyan
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130

# torchaudio has no 2.12-series wheel yet (maintenance mode) and its own
# metadata pins torch==2.11.0, so it MUST be installed --no-deps or pip would
# downgrade torch back to 2.11.0.
Write-Host "[*] Installing torchaudio 2.11.0 (--no-deps; declares torch==2.11.0) ..." -ForegroundColor Cyan
pip install torchaudio==2.11.0 --no-deps --index-url https://download.pytorch.org/whl/cu130

# -- Remaining dependencies -----------------------------------------------
# torch/torchvision/torchaudio (installed above) and scenedetect/sam3 (need
# --no-deps below) are excluded from this bulk install - see install-deps.sh
# for the full rationale (this mirrors its filter minus triton/triton-windows
# - local venvs need those from requirements; only the container filters them
# to protect its baked 2.11-matched copy).

Write-Host ""
Write-Host "[+] Installing remaining dependencies ..." -ForegroundColor Cyan
$TmpReq = [System.IO.Path]::GetTempFileName()
Get-Content requirements.txt |
    Where-Object { $_ -notmatch '^\s*(scenedetect|sam3|hpsv2|torch|torchvision|torchaudio)([\s=<>!~#]|$)' } |
    Set-Content $TmpReq
pip install -r $TmpReq
Remove-Item $TmpReq -Force

# scenedetect's declared dependency is the GUI build `opencv-python`, which
# collides with the pinned `opencv-python-headless` (both ship the `cv2`
# module) - install it separately, without its deps.
$SD = Get-Content requirements.txt |
    Where-Object { $_ -match '^\s*scenedetect\s*==' } |
    ForEach-Object { ($_ -replace '#.*$', '') -replace '\s', '' } |
    Select-Object -First 1
if ($SD) {
    Write-Host "[+] Installing $SD (--no-deps) ..." -ForegroundColor Cyan
    pip install --no-deps $SD
}

# sam3 declares `huggingface-hub<1.0,>=0.30.0`, but this repo pins
# huggingface-hub==1.27.0 (transformers 5.x requires it). The import works
# fine under hub 1.x - its declared ceiling is just stale (see
# test_sam3_imports_cleanly_despite_declared_hub_pin) - so install it
# separately, without its deps, rather than letting it block the resolve.
$S3 = Get-Content requirements.txt |
    Where-Object { $_ -match '^\s*sam3\s*==' } |
    ForEach-Object { ($_ -replace '#.*$', '') -replace '\s', '' } |
    Select-Object -First 1
if ($S3) {
    Write-Host "[+] Installing $S3 (--no-deps) ..." -ForegroundColor Cyan
    pip install --no-deps $S3
}

# hpsv2 declares `pytest ==7.2.0` and `pytest-split ==0.8.0` as INSTALL
# requirements - its dev deps leaked into its metadata. It never imports either
# and registers no pytest11 hook, so the pins are inert (see
# test_hpsv2_works_under_a_runner_its_metadata_forbids), but left in the bulk
# resolve they abort it against our own pytest pin. All 18 of its real deps are
# already in requirements.txt, so nothing is lost by skipping its metadata.
$HP = Get-Content requirements.txt |
    Where-Object { $_ -match '^\s*hpsv2\s*==' } |
    ForEach-Object { ($_ -replace '#.*$', '') -replace '\s', '' } |
    Select-Object -First 1
if ($HP) {
    Write-Host "[+] Installing $HP (--no-deps) ..." -ForegroundColor Cyan
    pip install --no-deps $HP
}

Write-Host ""
Write-Host "[OK] Done - all dependencies installed." -ForegroundColor Green
