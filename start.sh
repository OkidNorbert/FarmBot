#!/bin/bash
# AI Tomato Sorter Startup Script

echo "🌐 Starting AI Tomato Sorter"
echo "================================"

# Activate virtual environment
source farmbot_env/bin/activate

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
python -c "import torch, cv2, flask, yaml, flask_socketio; print('✅ All dependencies available')" || {
    echo "❌ Missing dependencies, please run setup.sh again"
    exit 1
}

# Check and free port 5000 if in use
echo "🔍 Checking port 5000..."
if lsof -ti :5000 > /dev/null 2>&1; then
    echo "⚠️  Port 5000 is in use. Cleaning up..."
    lsof -ti :5000 | xargs kill -9 2>/dev/null
    sleep 1
    # Also kill any web_interface processes
    pkill -9 -f "web_interface.py" 2>/dev/null
    sleep 1
    if lsof -ti :5000 > /dev/null 2>&1; then
        echo "❌ Could not free port 5000. Please manually kill the process using it."
        exit 1
    else
        echo "✅ Port 5000 is now free"
    fi
else
    echo "✅ Port 5000 is available"
fi

# Start the unified web interface
echo "Starting AI Tomato Sorter Unified Web Interface..."
exec python web_interface.py
