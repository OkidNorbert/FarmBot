#!/bin/bash
# AI Tomato Sorter Startup Script

echo "🌐 Starting AI Tomato Sorter"
echo "================================"

# Activate virtual environment
source tomato_sorter_env/bin/activate

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check Python version
python --version

# Check required packages
echo "🔍 Checking dependencies..."
python -c "import torch, cv2, flask, yaml; print('✅ All dependencies available')" || {
    echo "❌ Missing dependencies, please run setup.sh again"
    exit 1
}

# Start the web interface
echo "Starting AI Tomato Sorter Web Interface..."
exec "$VENV_DIR/bin/python" pi_web_interface.py
