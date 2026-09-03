@echo off
REM --------------------------------------------------------------------------
REM MRLN Arcane Tuner - Windows install script (CMD)
REM
REM Optionally creates a virtual environment, installs PyTorch with CUDA 13.0
REM from the official PyTorch wheel index, then installs all remaining
REM dependencies. PEP 508 markers in requirements.txt automatically handle
REM platform-specific packages.
REM --------------------------------------------------------------------------

set VENV_DIR=venv

REM -- Virtual environment ------------------------------------------------

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
        echo [WARN] Skipping venv - installing into current Python environment.
    ) else (
        echo Creating virtual environment ...
        python -m venv %VENV_DIR%
        call "%VENV_DIR%\Scripts\activate.bat"
        pip install --upgrade pip
        echo [OK] Virtual environment created and activated.
    )
)

REM -- PyTorch (CUDA 13.0, split stack) -----------------------------------

echo.
echo Installing PyTorch 2.12.1 + torchvision 0.27.1 (CUDA 13.0) ...
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130

REM torchaudio has no 2.12-series wheel yet (maintenance mode) and its own
REM metadata pins torch==2.11.0, so it MUST be installed --no-deps or pip
REM would downgrade torch back to 2.11.0.
echo Installing torchaudio 2.11.0 (--no-deps; declares torch==2.11.0) ...
pip install torchaudio==2.11.0 --no-deps --index-url https://download.pytorch.org/whl/cu130

REM -- Remaining dependencies ---------------------------------------------
REM torch/torchvision/torchaudio (installed above) and scenedetect/sam3 (need
REM --no-deps below) are excluded from this bulk install - see
REM install-deps.sh for the full rationale (this mirrors its filter minus
REM triton/triton-windows - local venvs need those from requirements; only
REM the container filters them to protect its baked 2.11-matched copy).

echo.
echo Installing remaining dependencies ...
findstr /V /R /I "^scenedetect== ^sam3== ^hpsv2== ^torch== ^torchvision== ^torchaudio==" requirements.txt > "%TEMP%\mrln_requirements_filtered.txt"
pip install -r "%TEMP%\mrln_requirements_filtered.txt"
del "%TEMP%\mrln_requirements_filtered.txt"

REM scenedetect's declared dependency is the GUI build `opencv-python`, which
REM collides with the pinned `opencv-python-headless` (both ship the `cv2`
REM module) - install it separately, without its deps.
set "SD="
for /f "usebackq tokens=1 delims= #" %%A in (`findstr /R /I "^scenedetect==" requirements.txt`) do (
    if not defined SD set "SD=%%A"
)
if defined SD (
    echo Installing %SD% ^(--no-deps^) ...
    pip install --no-deps %SD%
)

REM sam3 declares `huggingface-hub<1.0,>=0.30.0`, but this repo pins
REM huggingface-hub==1.27.0 (transformers 5.x requires it). The import works
REM fine under hub 1.x - its declared ceiling is just stale (see
REM test_sam3_imports_cleanly_despite_declared_hub_pin) - so install it
REM separately, without its deps, rather than letting it block the resolve.
set "S3="
for /f "usebackq tokens=1 delims= #" %%A in (`findstr /R /I "^sam3==" requirements.txt`) do (
    if not defined S3 set "S3=%%A"
)
if defined S3 (
    echo Installing %S3% ^(--no-deps^) ...
    pip install --no-deps %S3%
)

REM hpsv2 declares `pytest ==7.2.0` and `pytest-split ==0.8.0` as INSTALL
REM requirements - its dev deps leaked into its metadata. It never imports
REM either and registers no pytest11 hook, so the pins are inert (see
REM test_hpsv2_works_under_a_runner_its_metadata_forbids), but left in the bulk
REM resolve they abort it against our own pytest pin. All 18 of its real deps
REM are already in requirements.txt, so nothing is lost by skipping its metadata.
set "HP="
for /f "usebackq tokens=1 delims= #" %%A in (`findstr /R /I "^hpsv2==" requirements.txt`) do (
    if not defined HP set "HP=%%A"
)
if defined HP (
    echo Installing %HP% ^(--no-deps^) ...
    pip install --no-deps %HP%
)

echo.
echo [OK] Done - all dependencies installed.
pause
