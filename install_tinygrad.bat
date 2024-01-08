@echo off
SETLOCAL

REM Change directory
cd C:\Users\bunni\OneDrive\Desktop\tinygrad_dev\

REM Check if the .venv directory exists
if not exist .venv (
    REM Create a virtual environment
    python -m venv .venv
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment.
        exit /b
    )
)

REM Check if the virtual environment is active
IF DEFINED VIRTUAL_ENV (
    echo Virtual environment is active.
) ELSE (
    echo Virtual environment is not active.
    REM Activate the virtual environment
    call .venv\Scripts\activate
)

REM Set user and repo
SET user=tinygrad
SET repo=tinygrad

REM Remove local directory if it already exists
IF EXIST %repo% (
    rmdir /S /Q %repo%
)

REM Clone the repository
git clone https://github.com/%user%/%repo%.git

REM Install the package
python -m pip install -e ./%repo%

REM Add the pythonpath
set PYTHONPATH=%PYTHONPATH%;%CD%\tinygrad\tinygrad
echo %PYTHONPATH%

ENDLOCAL