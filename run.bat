@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Run the bot
python bot\main.py

REM Deactivate the virtual environment (optional)
call venv\Scripts\deactivate

pause
