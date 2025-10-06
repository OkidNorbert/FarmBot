#!/bin/bash
# Tomato Classification GUI Launcher
# Easy script to start the GUI application

echo "🍅 AI Tomato Sorter - GUI Launcher"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "tomato_sorter_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup script first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source tomato_sorter_env/bin/activate

# Check if GUI files exist
if [ ! -f "tomato_gui.py" ]; then
    echo "❌ GUI files not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

# Launch GUI
echo "🚀 Starting GUI application..."
python launch_gui.py

echo "👋 GUI application closed. Goodbye!"
