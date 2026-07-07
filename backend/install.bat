@echo off
REM ──────────────────────────────────────────────────────────────────────────
REM MRLN Arcane Tuner — Windows install script (CMD)
REM
REM Optionally creates a virtual environment, installs PyTorch with CUDA 13.0
REM from the official PyTorch wheel index, then installs all remaining
REM dependencies. PEP 508 markers in requirements.txt automatically handle
REM platform-specific packages.
REM ──────────────────────────────────────────────────────────────────────────

set VENV_DIR=venv

REM ── Virtual environment ────────────────────────────────────────────────

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [OK] Virtual environment '%VENV_DIR%' already exists.
    set /p ANSWER="   Activate it and continue? [Y/n] "
    if /i "%ANSWER%"=="" set ANSWER=Y
    if /i "%ANSWER%"=="n" (
        echo Aborted.
        exit /b 0
    )
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    set /p ANSWER="Create a virtual environment in '.\%VENV_DIR%'? [Y/n] "
    if /i "%ANSWER%"=="" set ANSWER=Y
    if /i "%ANSWER%"=="n" (
        echo [WARN] Skipping venv — installing into current Python environment.
    ) else (
        echo Creating virtual environment ...
        python -m venv %VENV_DIR%
        call "%VENV_DIR%\Scripts\activate.bat"
        pip install --upgrade pip
        echo [OK] Virtual environment created and activated.
    )
)

REM ── PyTorch (CUDA 13.0, split stack) ───────────────────────────────────

echo.
echo Installing PyTorch 2.12.1 + torchvision 0.27.1 (CUDA 13.0) ...
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130

REM torchaudio has no 2.12-series wheel yet (maintenance mode) and its own
REM metadata pins torch==2.11.0, so it MUST be installed --no-deps or pip
REM would downgrade torch back to 2.11.0.
echo Installing torchaudio 2.11.0 (--no-deps; declares torch==2.11.0) ...
pip install torchaudio==2.11.0 --no-deps --index-url https://download.pytorch.org/whl/cu130

REM ── Remaining dependencies ─────────────────────────────────────────────
REM torch/torchvision/torchaudio (installed above) and scenedetect (needs
REM --no-deps below) are excluded from this bulk install — see
REM install-deps.sh for the full rationale (this mirrors its filter).

echo.
echo Installing remaining dependencies ...
findstr /V /R /I "^scenedetect== ^torch== ^torchvision== ^torchaudio==" requirements.txt > "%TEMP%\mrln_requirements_filtered.txt"
pip install -r "%TEMP%\mrln_requirements_filtered.txt"
del "%TEMP%\mrln_requirements_filtered.txt"

REM scenedetect's declared dependency is the GUI build `opencv-python`, which
REM collides with the pinned `opencv-python-headless` (both ship the `cv2`
REM module) — install it separately, without its deps.
set "SD="
for /f "usebackq tokens=1 delims= #" %%A in (`findstr /R /I "^scenedetect==" requirements.txt`) do (
    if not defined SD set "SD=%%A"
)
if defined SD (
    echo Installing %SD% (--no-deps) ...
    pip install --no-deps %SD%
)

echo.
echo [OK] Done — all dependencies installed.
pause
