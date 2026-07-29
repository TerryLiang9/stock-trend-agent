@echo off
cd /d "%~dp0"
echo ================================
echo   Stock Agent Server
echo ================================
echo.
echo Starting server on http://0.0.0.0:8000
echo Press Ctrl+C to stop
echo.
call .venv\Scripts\activate.bat
python main.py --serve-only
pause
