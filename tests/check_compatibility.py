#!/usr/bin/env python3
"""
Quick compatibility check for the system
"""

import os
import sys

print("=" * 60)
print("System Compatibility Check")
print("=" * 60)
print()

# Check Arduino firmware
arduino_main = "arduino/src/main.ino"
if os.path.exists(arduino_main):
    print("✅ New Arduino firmware found: arduino/src/main.ino")
    with open(arduino_main, 'r') as f:
        content = f.read()
        if "WebSocket" in content or "commClient" in content:
            print("   → Uses WebSocket communication (NEW SYSTEM)")
        elif "Serial" in content and "PICK" in content:
            print("   → Uses Serial G-code (OLD SYSTEM)")
else:
    print("❌ Arduino firmware not found")

print()

# Check web interface
web_interface = "web_interface.py"
if os.path.exists(web_interface):
    print("✅ Web interface found")
    with open(web_interface, 'r') as f:
        content = f.read()
        if "socketio.emit('command'" in content:
            print("   → Has WebSocket Arduino handlers (NEW SYSTEM)")
        if "HardwareController" in content:
            print("   → Has hardware_controller integration (LEGACY)")
else:
    print("❌ Web interface not found")

print()

# Check hardware controller
hw_controller = "hardware_controller.py"
if os.path.exists(hw_controller):
    print("✅ Hardware controller found")
    with open(hw_controller, 'r') as f:
        content = f.read()
        if "send_command" in content and "PICK" in content:
            print("   → Uses Serial/Bluetooth G-code (OLD SYSTEM)")
        if "BLE" in content:
            print("   → Has Bluetooth support")
else:
    print("❌ Hardware controller not found")

print()

# Check YOLO service
yolo_service = "yolo_service.py"
if os.path.exists(yolo_service):
    print("✅ YOLO service found (NEW FEATURE)")
else:
    print("❌ YOLO service not found")

print()

# Check calibration
calibration_wizard = "calibration/pixel_to_servo_wizard.py"
if os.path.exists(calibration_wizard):
    print("✅ Calibration wizard found (NEW FEATURE)")
else:
    print("❌ Calibration wizard not found")

print()
print("=" * 60)
print("Recommendation:")
print("=" * 60)
print()
print("If you want to use the NEW system:")
print("  1. Upload arduino/src/main.ino to Arduino")
print("  2. Configure WiFi in arduino/src/config.h")
print("  3. Start web interface: python web_interface.py")
print("  4. Arduino will connect via WebSocket automatically")
print()
print("The hardware_controller.py will still work for camera/AI,")
print("but won't connect to Arduino (that's OK - WebSocket handles it)")
print()

