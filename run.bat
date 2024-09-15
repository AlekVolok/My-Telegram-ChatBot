@echo off
REM Set PYTHONPATH to the project root directory
set PYTHONPATH=%~dp0

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the bot
python main.py

REM Deactivate the virtual environment (optional)
call venv\Scripts\deactivate

pause
