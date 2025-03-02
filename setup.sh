#!/usr/bin/env bash

# Create a virtual environment 
python3 -m venv venv

# Activate the virtual environment
. venv/bin/activate

# Install the required packages
pip install -r requirements.txt