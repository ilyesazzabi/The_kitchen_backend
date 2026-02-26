@echo off
cd /d "%~dp0"
REM Lancer le serveur WebSocket + la détection
"..\staff detection images\venv_new\Scripts\python.exe" ws_dashboard_server.py
pause
