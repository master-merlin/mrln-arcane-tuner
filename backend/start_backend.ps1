# Start the MRLN Arcane Tuner backend server
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000
