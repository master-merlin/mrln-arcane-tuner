@echo off
REM --------------------------------------------------------------------------
REM MRLN Arcane Tuner - Windows install script (CMD)
REM
REM Optionally creates a virtual environment, installs the torch stack this
REM machine can run (torch_stack.py reads the driver and every GPU and picks a
REM set from torch-stack.cfg), then installs all remaining dependencies. PEP 508
REM markers in requirements.txt automatically handle platform-specific packages.
REM
REM   install.bat [--venv-dir <dir>] [--yes]
REM
REM --venv-dir  the venv to create or reuse (default: venv next to this script).
REM --yes       no questions: reuse or create the venv.
REM MRLN_TORCH_INDEX=<set> chooses the torch set (cu132, cu126, cpu); it is
REM still refused where that set cannot run.
REM --------------------------------------------------------------------------
setlocal

REM Anchored on this script, never on the current directory.
set "HERE=%~dp0"
set "REQ=%HERE%requirements.txt"
set "STACK=%HERE%torch_stack.py"
set "VENV_DIR=%HERE%venv"
set "YES=0"

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--venv-dir" (
    set "VENV_DIR=%~2"
    shift
    shift
    goto args
)
if /i "%~1"=="--yes" (
    set "YES=1"
    shift
    goto args
)
echo unknown argument: %~1
exit /b 64
:args_done

REM -- Virtual environment ------------------------------------------------
REM Every install below runs this venv's own interpreter by path. Nothing is
REM activated, so an already-active venv cannot redirect the install.
REM Each prompt is read OUTSIDE a parenthesised block: inside one, %ANSWER%
REM would be expanded when the block is parsed, before set /p ran.

set "PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%PY%" goto no_venv
echo [OK] Virtual environment '%VENV_DIR%' already exists.
if "%YES%"=="1" goto stack
set "ANSWER=Y"
set /p ANSWER="   Install into it? [Y/n] "
if /i "%ANSWER%"=="n" goto aborted
goto stack

:no_venv
if "%YES%"=="1" goto create_venv
set "ANSWER=Y"
set /p ANSWER="Create a virtual environment in '%VENV_DIR%'? [Y/n] "
if /i not "%ANSWER%"=="n" goto create_venv
echo [WARN] Skipping venv - installing into the current Python environment.
set "PY=python"
goto stack

:create_venv
echo Creating virtual environment ...
python -m venv "%VENV_DIR%" || goto failed
"%PY%" -m pip install --upgrade pip || goto failed
echo [OK] Virtual environment created.

REM -- The torch stack ------------------------------------------------------
REM torch_stack.py chooses the set, refuses (exit 2, with the table) before
REM installing anything where no set fits, and installs torch, torchvision,
REM torchaudio, torchao and triton from it.

:stack
echo.
"%PY%" "%STACK%" install --python "%PY%" || goto failed

REM -- Remaining dependencies ---------------------------------------------
REM The bulk install never sees the torch stack (torch_stack.py owns it) nor
REM scenedetect/sam3/hpsv2 (installed --no-deps below; see install-deps.sh for
REM why each needs it).

echo.
echo Installing remaining dependencies ...
set "TMP_REQ=%TEMP%\mrln_requirements_filtered.txt"
"%PY%" "%STACK%" filter "%REQ%" "%TMP_REQ%" --also scenedetect sam3 hpsv2 || goto failed
"%PY%" -m pip install -r "%TMP_REQ%"
set "RC=%ERRORLEVEL%"
del "%TMP_REQ%"
if not "%RC%"=="0" goto failed

REM scenedetect's declared dependency is the GUI build `opencv-python`, which
REM collides with the pinned `opencv-python-headless` (both ship the `cv2`
REM module) - install it separately, without its deps.
REM sam3's declared ceilings would block the resolve against our pins; its
REM import works (test_sam3_imports_cleanly_despite_declared_hub_pin).
REM hpsv2 declares its dev deps (`pytest ==7.2.0`, `pytest-split ==0.8.0`) as
REM install requirements; it never imports them
REM (test_hpsv2_works_under_a_runner_its_metadata_forbids).
for %%N in (scenedetect sam3 hpsv2) do call :no_deps %%N || goto failed

REM -- The stack is still the one chosen ----------------------------------

echo.
"%PY%" "%STACK%" verify --python "%PY%" || goto failed

echo.
echo [OK] Done - all dependencies installed.
if not "%YES%"=="1" pause
exit /b 0

:no_deps
set "PIN="
for /f "usebackq tokens=1 delims= #" %%A in (`findstr /R /I /C:"^%1==" "%REQ%"`) do (
    if not defined PIN set "PIN=%%A"
)
if not defined PIN exit /b 0
echo Installing %PIN% (--no-deps) ...
"%PY%" -m pip install --no-deps %PIN%
exit /b %ERRORLEVEL%

:aborted
echo Aborted.
exit /b 0

:failed
echo [FAILED] the install stopped; see the message above.
if not "%YES%"=="1" pause
exit /b 1
