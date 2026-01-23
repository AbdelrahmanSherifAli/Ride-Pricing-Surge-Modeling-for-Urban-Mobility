@echo off
echo ========================================
echo   Ride Price & Surge Prediction App
echo ========================================
echo.
echo Starting Streamlit...
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit is not installed. Installing...
    pip install streamlit
    echo.
)

REM Run streamlit
streamlit run streamlit_app.py

pause
