@echo off
REM Setup script for AI Prediction Market Bot (Windows)

echo Setting up AI Prediction Market Bot environment...

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "backtest_results" mkdir backtest_results

REM Create .env file from example if it doesn't exist
if not exist ".env" (
    echo Creating .env file from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo Please edit .env file with your configuration
    ) else (
        echo Warning: .env.example not found
    )
) else (
    echo .env file already exists
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
