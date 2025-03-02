#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Upgrade pip using venv's Python
venv/bin/python -m pip install --upgrade pip

# Install requirements using venv's Python
venv/bin/python -m pip install -r requirements.txt

echo "Virtual environment setup complete! ✅"
echo "To activate, run: source venv/bin/activate"
