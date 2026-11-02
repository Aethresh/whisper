@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
if "%~1"=="" (
  echo Drag a folder or text files onto this icon.
  pause
  exit /b 1
)
D:\anaconda3\envs\whisper\python.exe D:\whisper\wordfreq.py "%~1"
pause
