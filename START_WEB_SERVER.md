# 🚀 Starting the Web Interface

## Quick Start

### Option 1: Direct Start (if dependencies installed)
```bash
cd /home/okidi6/Documents/GitHub/emebeded
python3 web_interface.py
```

### Option 2: Using Start Script
```bash
cd /home/okidi6/Documents/GitHub/emebeded
python3 start_web_interface.py
```

## Access the Web Interface

Once started, open your web browser and go to:
- **http://localhost:5000**
- **http://127.0.0.1:5000**

## Test Arduino BLE Connection

Before starting the web interface, test the Arduino connection:

```bash
cd /home/okidi6/Documents/GitHub/emebeded
python3 test_arduino_ble.py
```

This will:
1. Scan for Arduino BLE device named "FarmBot"
2. Connect to it
3. Send a test command (HOME)
4. Verify communication works

## Install Missing Dependencies

If you get import errors, install dependencies:

```bash
# Install Bleak for BLE support
pip3 install bleak --break-system-packages

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install bleak flask flask-socketio opencv-python
```

## Arduino Status Check

Make sure your Arduino:
1. ✅ Is powered on
2. ✅ Has firmware uploaded (main_firmware.ino)
3. ✅ Is advertising BLE as "FarmBot"
4. ✅ Shows "BLE FarmBot Ready" in Serial Monitor

## Web Interface Features

Once running, you can:
- View live camera feed
- Control robotic arm manually
- Run tomato detection
- Send commands to Arduino via BLE
- Monitor system status

## Troubleshooting

### "ModuleNotFoundError: No module named 'bleak'"
```bash
pip3 install bleak --break-system-packages
```

### "Arduino not found" in BLE scan
- Check Arduino Serial Monitor - should show "BLE FarmBot Ready"
- Make sure Bluetooth is enabled on your computer
- Try resetting the Arduino

### Web interface won't start
- Check if port 5000 is already in use: `lsof -i :5000`
- Install missing Python packages
- Check Python version: `python3 --version` (needs 3.7+)

