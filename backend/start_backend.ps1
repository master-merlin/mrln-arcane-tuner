# Start the MRLN Arcane Tuner backend server
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
# Invoke uvicorn through the interpreter, not the `uvicorn.exe` console-script
# shim. Those shims hard-code the absolute path of the interpreter that created
# them, so they die silently if the venv directory is ever renamed or moved
# (`python.exe` itself is fine — it derives sys.prefix from its own location).
& .\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
