@echo off
chcp 65001 >nul
set "PATH=D:\anaconda3\envs\whisper\Lib\site-packages\nvidia\cublas\bin;D:\anaconda3\envs\whisper\Lib\site-packages\nvidia\cudnn\bin;%PATH%"
set "HF_HOME=D:\huggingface"
set "HF_ENDPOINT=https://hf-mirror.com"
set "PYTHONIOENCODING=utf-8"
if "%~1"=="" (
  echo Drag an audio/video file onto this icon to transcribe it.
  pause
  exit /b 1
)
D:\anaconda3\envs\whisper\python.exe D:\whisper\transcribe.py "%~1"
pause
