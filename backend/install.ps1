# ──────────────────────────────────────────────────────────────────────────
# MRLN Arcane Tuner — Windows install script
#
# Optionally creates a virtual environment, installs PyTorch with CUDA 13.0
# from the official PyTorch wheel index, then installs all remaining
# dependencies. PEP 508 markers in requirements.txt automatically handle
# platform-specific packages.
# ──────────────────────────────────────────────────────────────────────────

$VenvDir = "venv"

# ── Virtual environment ──────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host "✅ Virtual environment '$VenvDir' already exists." -ForegroundColor Green
    $answer = Read-Host "   Activate it and continue? [Y/n]"
    if ($answer -eq '' ) { $answer = 'Y' }
    if ($answer -match '^[Nn]$') {
        Write-Host "Aborted."
        exit 0
    }
    & "$VenvDir\Scripts\Activate.ps1"
} else {
    $answer = Read-Host "🐍 Create a virtual environment in '.\$VenvDir'? [Y/n]"
    if ($answer -eq '' ) { $answer = 'Y' }
    if ($answer -match '^[Nn]$') {
        Write-Host "⚠️  Skipping venv — installing into current Python environment." -ForegroundColor Yellow
    } else {
        Write-Host "🔧 Creating virtual environment ..." -ForegroundColor Cyan
        python -m venv $VenvDir
        & "$VenvDir\Scripts\Activate.ps1"
        pip install --upgrade pip
        Write-Host "✅ Virtual environment created and activated." -ForegroundColor Green
    }
}

# ── PyTorch (CUDA 13.0) ─────────────────────────────────────────────────

Write-Host ""
Write-Host "🔧 Installing PyTorch 2.10.0 + CUDA 12.6 ..." -ForegroundColor Cyan
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# ── Remaining dependencies ───────────────────────────────────────────────

Write-Host ""
Write-Host "📦 Installing remaining dependencies ..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "✅ Done — all dependencies installed." -ForegroundColor Green
