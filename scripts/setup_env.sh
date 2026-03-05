#!/bin/bash
# Setup script for AI Prediction Market Bot

echo "Setting up AI Prediction Market Bot environment..."

# Create necessary directories
mkdir -p logs data models backtest_results

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please edit .env file with your configuration"
    else
        echo "Warning: .env.example not found"
    fi
else
    echo ".env file already exists"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
