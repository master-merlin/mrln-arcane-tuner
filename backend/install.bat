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

REM ── PyTorch (CUDA 12.6) ───────────────────────────────────────────────

echo.
echo Installing PyTorch 2.10.0 + CUDA 12.6 ...
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126

REM ── Remaining dependencies ─────────────────────────────────────────────

echo.
echo Installing remaining dependencies ...
pip install -r requirements.txt

echo.
echo [OK] Done — all dependencies installed.
pause
