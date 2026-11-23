@echo off
rem Start AI Reel Maker and open it in the browser. Double-click again to just open the browser.
rem Stop it with Ctrl+C in this window, or with stop.bat.
setlocal
cd /d "%~dp0"
title AI Reel Maker

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from python.org, tick "Add python.exe to PATH",
  echo then run: pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "frontend\dist\index.html" (
  call :build_ui
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

"%PY%" run.py --open-browser
if errorlevel 1 (
  echo.
  echo AI Reel Maker stopped with an error. See the messages above.
  pause
  exit /b 1
)
exit /b 0

:build_ui
where npm >nul 2>nul
if errorlevel 1 (
  echo The app's screens are not built yet and Node.js was not found.
  echo Install Node.js from nodejs.org, then run start.bat again.
  exit /b 1
)
echo Building the app's screens. This happens only the first time...
pushd frontend
if not exist "node_modules" (
  call npm install
  if errorlevel 1 goto build_failed
)
call npm run build
if errorlevel 1 goto build_failed
popd
exit /b 0

:build_failed
popd
echo Building the app's screens failed. See the messages above.
exit /b 1
