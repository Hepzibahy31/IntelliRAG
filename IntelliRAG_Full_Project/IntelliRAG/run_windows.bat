@echo off
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist .env copy .env.example .env
start http://localhost:8000
python server.py
