@echo off
setlocal
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

REM This script is the server's SUPERVISOR (the restart contract lives in
REM backend\app\core\restart_contract.py, the one producer of the exit code):
REM the server is told it is supervised, and when it exits with code 75 -
REM "relaunch me", what a restart from the UI or the self-updater asks for -
REM this script starts it again IN THIS TERMINAL, with MRLN_RESTART=1 and the
REM port resolved afresh. Any other exit code is this script's exit code too: a
REM crash must not loop. The relaunch belongs to the owner of the console: a
REM replacement spawned by the server that is dying is an orphan by construction,
REM invisible in the window you started from - which is what every earlier
REM restart fix made observable without curing.
REM
REM `setlocal` above keeps MRLN_SUPERVISED (and everything else set here) OUT of
REM the shell that called this script: a bare `uvicorn` started later from the
REM same window must NOT inherit it, or its restart would exit 75 with nothing
REM listening for it.
set MRLN_SUPERVISED=1

:launch
REM The port has ONE producer: port_resolver.py, which every launcher calls.
REM This script does NOT parse settings.json - four launchers in three languages
REM each reading the same file is four chances to disagree with the app about
REM which port it is on, which is the bug this replaced. The shell only carries
REM the answer across on --port; the app reads it back out of its own argv.
REM
REM A failure here is a REFUSAL, not a fallback. Defaulting to 8000 when the
REM resolver cannot answer would start a server on a port the settings screen
REM denies - the same silent disagreement, restored by the error path.
REM
REM `for /f` cannot see the child's exit code, so the refusal is detected by an
REM EMPTY variable instead. That is sound because the resolver prints nothing on
REM stdout when it refuses (its reason goes to stderr, which passes straight
REM through to this console) - a property pinned by a test, precisely so this
REM line can depend on it.
set MRLN_RESOLVED_PORT=
for /f "usebackq delims=" %%p in (`venv\Scripts\python.exe port_resolver.py`) do set MRLN_RESOLVED_PORT=%%p
if "%MRLN_RESOLVED_PORT%"=="" (
    echo start_backend: refusing to start - the backend port could not be determined.
    echo start_backend: the reason is above; fix or delete backend\settings.json.
    exit /b 1
)
venv\Scripts\python.exe -m uvicorn app.main:app --host %MRLN_BIND_HOST% --port %MRLN_RESOLVED_PORT%
set MRLN_EXIT=%errorlevel%
REM Captured first, compared second: %errorlevel% inside a parenthesised block
REM expands when the block is PARSED, so it would be the value from before.
if "%MRLN_EXIT%"=="75" (
    echo start_backend: restart requested - starting again in this terminal
    set MRLN_RESTART=1
    goto launch
)
if "%MRLN_EXIT%"=="3" if "%MRLN_RESTART%"=="1" (
    echo start_backend: the relaunched server could not bind - a child of the old server may still hold the port; see the output above, backend\server.log and the task list
)
exit /b %MRLN_EXIT%
