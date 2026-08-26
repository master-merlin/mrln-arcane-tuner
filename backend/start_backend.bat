@echo off
REM Start the MRLN Arcane Tuner backend server
cd /d "%~dp0"
call venv\Scripts\activate.bat
REM Invoke uvicorn through the interpreter, not the `uvicorn.exe` console-script
REM shim. Those shims hard-code the absolute path of the interpreter that created
REM them, so they die silently if the venv directory is ever renamed or moved
REM (python.exe itself is fine - it derives sys.prefix from its own location).
REM
REM Binds to THIS MACHINE ONLY by default. It used to be 0.0.0.0, which put an
REM unauthenticated server holding your datasets, models and GPU on every
REM network you join - including hotel and cafe wifi. To reach it from another
REM machine, set a token AND widen the bind:
REM   set MRLN_AUTH_TOKEN=<long random string>
REM   set MRLN_BIND_HOST=0.0.0.0
REM Widening without a token is refused at startup, by design.
if "%MRLN_BIND_HOST%"=="" set MRLN_BIND_HOST=127.0.0.1
if "%PORT%"=="" set PORT=8000
venv\Scripts\python.exe -m uvicorn app.main:app --host %MRLN_BIND_HOST% --port %PORT%
