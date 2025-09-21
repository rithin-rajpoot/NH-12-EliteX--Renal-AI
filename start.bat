@echo off
title RenalAI - Kidney Stone Detector
color 0B

echo.
echo     ╔══════════════════════════════════════════════════════════════╗
echo     ║                       🏥 RenalAI 🏥                         ║
echo     ║                 AI-Powered Medical Imaging                   ║
echo     ║                   Quick Start Launcher                       ║
echo     ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 Starting Kidney Stone Detector...
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo 💡 Please install Python 3.8 or higher and try again
    echo.
    pause
    exit /b 1
)

:: Check if we're in the correct directory
if not exist "streamlit_app\app.py" (
    echo ❌ Error: Please run this script from the kidney_stone_detector directory
    echo 💡 Current directory should contain streamlit_app folder
    echo.
    pause
    exit /b 1
)

:: Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Python found
echo ✅ Dependencies ready
echo ✅ Starting application...
echo.

:: Start the Streamlit application
echo 🌐 Opening Kidney Stone Detector...
echo 💡 The application will open in your default web browser
echo 🛑 Press Ctrl+C to stop the application
echo.

:: Run the application with optimal settings
python -m streamlit run streamlit_app/app.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false

:: If the application stops, show exit message
echo.
echo 🛑 Application stopped.
echo 👋 Thank you for using RenalAI!
echo.
pause