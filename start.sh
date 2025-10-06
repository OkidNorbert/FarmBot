#!/bin/bash
# AI Tomato Sorter Startup Script

echo "🌐 Starting AI Tomato Sorter"
echo "================================"

# Check if virtual environment exists
if [ ! -d "tomato_sorter_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to install the system"
    exit 1
fi

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
echo "🐍 Python version: $(python --version)"

# Check required packages
echo "🔍 Checking dependencies..."
python -c "
import sys
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except ImportError as e:
    print('❌ PyTorch not found:', e)
    sys.exit(1)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV not found:', e)
    sys.exit(1)

try:
    import flask
    print('✅ Flask:', flask.__version__)
except ImportError as e:
    print('❌ Flask not found:', e)
    sys.exit(1)

try:
    import numpy
    print('✅ NumPy:', numpy.__version__)
except ImportError as e:
    print('❌ NumPy not found:', e)
    sys.exit(1)

print('✅ All dependencies available')
"

if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies, please run ./setup.sh to install"
    exit 1
fi

# Check if web_interface.py exists
if [ ! -f "web_interface.py" ]; then
    echo "❌ web_interface.py not found!"
    echo "Please ensure you're in the correct directory"
    exit 1
fi

# Check camera availability
echo "📹 Checking camera..."
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera available at index 0')
    cap.release()
else:
    print('⚠️  Camera not available (this is normal if no camera is connected)')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p temp
mkdir -p models/tomato
mkdir -p datasets/tomato
mkdir -p learning_data
mkdir -p logs

# Set permissions
chmod 755 temp models datasets learning_data logs

# Start web interface
echo "🚀 Starting web interface..."
echo "🌐 Web Interface: http://localhost:5001"
echo "📁 Upload folder: datasets/"
echo "💾 Models folder: models/"
echo "🔧 Using virtual environment for PyTorch & OpenCV"
echo "📁 Temp directory: temp/"
echo "🧠 Continuous Learning: ENABLED"
print("📹 Live Camera Feed: ENABLED")
print("🍅 Real-Time Object Detection: ENABLED")
print("🤖 Production-Ready: Single-Tomato Classification")
print("🍅 Dataset: 7,224 images, 4 classes")
echo "============================================================"
echo "Press Ctrl+C to stop the server"
echo "============================================================"

# Start the Flask application
python web_interface.py
