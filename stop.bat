@echo off
rem Stop AI Reel Maker, wherever it was started from.
setlocal
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" scripts\stop_app.py
if errorlevel 1 (
  echo Could not stop AI Reel Maker. See the messages above.
  pause
  exit /b 1
)
rem Leave the message on screen for a moment before the window closes.
timeout /t 3 /nobreak >nul 2>nul
exit /b 0
