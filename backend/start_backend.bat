@echo off
REM Start the MRLN Arcane Tuner backend server
cd /d "%~dp0"
call venv\Scripts\activate.bat
REM Invoke uvicorn through the interpreter, not the `uvicorn.exe` console-script
REM shim. Those shims hard-code the absolute path of the interpreter that created
REM them, so they die silently if the venv directory is ever renamed or moved
REM (python.exe itself is fine - it derives sys.prefix from its own location).
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
