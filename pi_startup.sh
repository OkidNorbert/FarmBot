#!/bin/bash
# Raspberry Pi Auto-Startup Script for AI Tomato Sorter
# This script runs automatically on boot

echo "🍅 AI Tomato Sorter - Auto Startup"
echo "=================================="

# Wait for system to fully boot
sleep 10

# Change to project directory
cd /home/$USER/tomato_sorter

# Activate virtual environment
source tomato_sorter_env/bin/activate

# Check system status
echo "🔍 Checking system status..."

# Check camera
if vcgencmd get_camera | grep -q "detected=1"; then
    echo "✅ Camera detected"
else
    echo "⚠️  Camera not detected"
fi

# Check Arduino
if ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM* 2>/dev/null; then
    echo "✅ Arduino detected"
else
    echo "⚠️  Arduino not detected"
fi

# Check network
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "✅ Network connected"
    PI_IP=$(hostname -I | awk '{print $1}')
    echo "🌐 Pi IP: $PI_IP"
else
    echo "⚠️  Network not connected"
fi

# Start the main controller
echo "🚀 Starting AI Tomato Sorter..."
python pi_controller.py
