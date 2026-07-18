@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Pastikan Anda sudah membuat venv di folder proyek.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
