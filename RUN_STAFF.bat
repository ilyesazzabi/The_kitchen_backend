@echo off
cd /d "%~dp0"
echo === The Kitchen Backend ===
echo.

REM Essayer le venv local, sinon utiliser python du PATH
if exist "venv\Scripts\python.exe" (
    echo Utilisation du venv local
    "venv\Scripts\python.exe" -u video_to_dashboard.py %*
) else if exist "venv_new\Scripts\python.exe" (
    echo Utilisation de venv_new
    "venv_new\Scripts\python.exe" -u video_to_dashboard.py %*
) else (
    echo Utilisation de python du PATH
    python -u video_to_dashboard.py %*
)
pause
