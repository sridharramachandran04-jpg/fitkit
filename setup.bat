@echo off
REM FitKit setup — builds an isolated virtual environment with the exact
REM dependency versions this project needs, so nothing else on your machine
REM (a global mediapipe install, another Python version, etc.) can override
REM them. Run this once, then always launch the app with run.bat.

set PY=C:\Users\chand\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PY%" (
    echo Could not find Python at %PY%
    echo Edit setup.bat and set PY= to your actual python.exe path.
    pause
    exit /b 1
)

echo Using: %PY%
echo.

echo [1/3] Creating virtual environment in .\venv ...
"%PY%" -m venv venv
if errorlevel 1 goto :error

echo [2/3] Installing pinned dependencies (this can take a few minutes) ...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/3] Verifying mediapipe...
python -c "import mediapipe as mp; assert mp.__version__ == '0.10.21', mp.__version__; assert hasattr(mp, 'solutions'); print('mediapipe', mp.__version__, 'OK — mp.solutions is available')"
if errorlevel 1 goto :error

echo.
echo Setup complete. Launch the app with: run.bat
pause
exit /b 0

:error
echo.
echo Setup failed — see the error above.
pause
exit /b 1
