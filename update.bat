@echo off
REM Update this MRLN Arcane Tuner checkout and its installed dependencies.
REM
REM   update.bat              pull, then install whatever is out of date
REM   update.bat --check      report what is out of date, change nothing
REM   update.bat --no-pull    skip git, just sync deps to this checkout
REM   update.bat --build      also run a production frontend build
REM
REM This file only finds a Python and hands over to update.py, where
REM all the logic lives. Deliberately thin: this repo already needs five
REM install paths to agree with each other, and a launcher that re-implements
REM any of the work is a sixth thing to keep in step.
REM
REM Exists alongside update.ps1 so a double-click works without touching
REM PowerShell's execution policy.

setlocal
set "HERE=%~dp0"
set "DRIVER=%HERE%update.py"

if not exist "%DRIVER%" (
    echo update: cannot find "%DRIVER%" 1>&2
    exit /b 1
)

REM Prefer the project's own venv: it is the interpreter the backend runs on,
REM so if it is broken the update should surface that rather than paper over it
REM with a system Python that happens to work.
set "PY=%HERE%backend\venv\Scripts\python.exe"
if not exist "%PY%" (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY_FALLBACK set "PY_FALLBACK=%%P"
    )
    set "PY=%PY_FALLBACK%"
)

if not defined PY (
    echo update: no Python interpreter found ^(tried backend\venv, python^). 1>&2
    exit /b 1
)
if not exist "%PY%" (
    echo update: no Python interpreter found ^(tried backend\venv, python^). 1>&2
    exit /b 1
)

"%PY%" "%DRIVER%" %*
exit /b %ERRORLEVEL%
