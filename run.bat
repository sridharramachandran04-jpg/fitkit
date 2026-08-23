@echo off
REM Always run FitKit through the isolated venv built by setup.bat, so the
REM pinned mediapipe==0.10.21 (and matching numpy/opencv/streamlit) are the
REM ones actually used — never whatever's on your global Python.

if not exist "venv\Scripts\activate.bat" (
    echo venv not found — run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
streamlit run app.py
pause
