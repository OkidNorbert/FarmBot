#!/bin/bash
# Upload Arduino sketch using Arduino CLI
# This script handles permission issues

export PATH="$HOME/.local/bin:$PATH"

SKETCH_DIR="arduino/test_all_components"
BOARD="arduino:renesas_uno:unor4wifi"
PORT="/dev/ttyACM0"

echo "=========================================="
echo "  Arduino Sketch Uploader"
echo "=========================================="
echo ""

# Check if Arduino CLI is installed
if ! command -v arduino-cli &> /dev/null; then
    echo "ERROR: Arduino CLI not found!"
    echo "Please add ~/.local/bin to your PATH"
    exit 1
fi

# Check if port exists
if [ ! -e "$PORT" ]; then
    echo "ERROR: Port $PORT not found!"
    echo "Please check your Arduino connection"
    exit 1
fi

# Check permissions
if [ ! -r "$PORT" ] || [ ! -w "$PORT" ]; then
    echo "WARNING: Permission denied on $PORT"
    echo ""
    echo "Fixing permissions temporarily..."
    sudo chmod 666 "$PORT" || {
        echo "ERROR: Could not fix permissions"
        echo ""
        echo "Please run one of these commands:"
        echo "  1. sudo usermod -a -G dialout \$USER  (then log out/in)"
        echo "  2. sudo chmod 666 $PORT"
        exit 1
    }
fi

# Change to project directory
cd "$(dirname "$0")/.." || exit 1

echo "Compiling sketch..."
arduino-cli compile --fqbn "$BOARD" "$SKETCH_DIR" || {
    echo "ERROR: Compilation failed!"
    exit 1
}

echo ""
echo "Uploading to Arduino R4 on $PORT..."
arduino-cli upload -p "$PORT" --fqbn "$BOARD" "$SKETCH_DIR" || {
    echo "ERROR: Upload failed!"
    exit 1
}

echo ""
echo "=========================================="
echo "  Upload Complete!"
echo "=========================================="
echo ""
echo "Open Serial Monitor (115200 baud) to see output"
echo "Try: arduino-cli monitor -p $PORT -c baudrate=115200"

