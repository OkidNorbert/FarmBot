# Tomato Sorter System - Quick Fix Guide

## What Was Fixed

### 1. **Camera Feed** ✅
- **Problem**: Duplicate route definition preventing proper streaming
- **Solution**: Removed duplicate route and properly structured the camera feed endpoint
- **Test**: Visit `http://<pi-ip>:5000/control` and check if camera feed displays

### 2. **Arduino Bluetooth Connection** ✅
- **Problem**: Firmware had BLE but web couldn't connect
- **Solution**: Added Bluetooth API endpoints and UI in control panel
- **Test**: Click "Scan for Devices" button in Control panel

### 3. **Servo Control (Wild Movements)** ✅
- **Problem**: No safety limits causing erratic servo behavior
- **Solution**: 
  - Added servo position validation
  - Limited max movement to 90° per command
  - Added emergency stop checks in movement loops
  - Double-checked bounds before writing to servos
- **Test**: Send manual arm commands and verify smooth, controlled movement

### 4. **Backend Integration** ✅
- **Problem**: Missing methods and command mismatches
- **Solution**:
  - Added `update_calibration()` method to hardware controller
  - Added PICK command handler in Arduino firmware
  - Added DISTANCE command response
  - Added GRIPPER alternative command

## How to Use

### Starting the System

```bash
cd /home/okidi6/Documents/GitHub/emebeded

# Start the web interface
python3 pi_web_interface.py
```

The web interface will be available at: `http://0.0.0.0:5000`

### Uploading Arduino Firmware

1. Open Arduino IDE
2. Load `/home/okidi6/Documents/GitHub/emebeded/arduino/tomato_sorter_arduino.ino`
3. Install required libraries:
   - ArduinoBLE
   - Adafruit_VL53L0X
   - Servo
4. Select your Arduino board and port
5. Upload the firmware

### Connecting via Bluetooth

1. Go to Control panel: `http://<pi-ip>:5000/control`
2. Click "Scan for Devices"
3. Click on "FarmBot" when it appears
4. Connection status will show "Connected"

### Testing Camera Feed

1. Go to Monitor panel: `http://<pi-ip>:5000/monitor`
2. Camera feed should display with timestamp
3. If no camera: placeholder image will show "NO CAMERA"

### Testing Servo Control

1. Go to Control panel
2. Use the axis control buttons:
   - **X Axis**: Base rotation
   - **Y Axis**: Forward/backward reach
   - **Z Axis**: Up/down height
3. Click "Home Position" to return to safe position
4. Monitor command log for feedback

### Emergency Stop

If servos behave erratically:
1. Send `STOP` command via serial monitor
2. Or click emergency stop in web interface (if implemented)
3. Send `HOME` command to reset

## Arduino Commands Reference

### Serial/BLE Commands

```
HOME                    - Move to home position
MOVE X Y CLASS          - Move to coordinates and sort
PICK X Y Z CLASS        - Pick from coordinates with depth
ANGLE A1 A2 A3 A4 A5 A6 - Set servo angles directly
GRIP OPEN               - Open gripper
GRIP CLOSE              - Close gripper
GRIPPER OPEN            - Alternative gripper command
GRIPPER CLOSE           - Alternative gripper command
STATUS                  - Get current status
DISTANCE                - Get distance sensor reading
STOP                    - Emergency stop
```

### Example Commands

```
HOME
MOVE 100 150 1
PICK 100 150 50 1
ANGLE 90 90 90 90 90 30
GRIP OPEN
DISTANCE
```

## Servo Safety Features

### New Safety Mechanisms

1. **Position Validation**: All angles constrained to min/max limits
2. **Movement Limiting**: Max 90° movement per command
3. **Emergency Stop**: Checked in every movement loop
4. **Bounds Checking**: Double-checked before writing to servo
5. **Index Validation**: Prevents invalid servo index access

### Servo Limits

```
Servo 1 (Base):         0° - 180°
Servo 2 (Shoulder):    10° - 170°
Servo 3 (Elbow):        0° - 180°
Servo 4 (Wrist Yaw):    0° - 180°
Servo 5 (Wrist Pitch):  0° - 180°
Servo 6 (Gripper):     20° - 160°
```

## Troubleshooting

### Camera Not Working
- Check if camera is connected: `ls /dev/video*`
- Test camera: `python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- Check permissions: `sudo usermod -a -G video $USER`

### Arduino Not Connecting
- Check serial port: `ls /dev/ttyUSB* /dev/ttyACM*`
- Check permissions: `sudo usermod -a -G dialout $USER`
- Verify baud rate: 115200

### Bluetooth Not Working
- Install BLE libraries: `sudo apt-get install python3-bluez`
- Enable Bluetooth: `sudo systemctl start bluetooth`
- Check BLE status: `bluetoothctl`

### Servos Moving Erratically
- Check power supply (servos need 5-6V, sufficient current)
- Verify wiring connections
- Send HOME command to reset
- Check for loose connections
- Reduce MAX_MOVEMENT_SPEED in firmware if needed

### Web Interface Not Loading
- Check if Flask is running: `ps aux | grep python`
- Check port 5000: `sudo netstat -tlnp | grep 5000`
- Check firewall: `sudo ufw allow 5000`

## Next Steps

1. **Test Camera Feed**: Verify video streaming works
2. **Test Arduino Commands**: Send commands via serial monitor
3. **Test Servo Movements**: Use web interface to control arm
4. **Calibrate System**: Use calibration page to map coordinates
5. **Test Auto Mode**: Enable auto mode for autonomous sorting

## Files Modified

- `pi_web_interface.py` - Fixed camera route, added Bluetooth endpoints
- `hardware_controller.py` - Added calibration method
- `arduino/tomato_sorter_arduino.ino` - Added safety features, PICK command
- `templates/pi_control.html` - Added Bluetooth UI

## Support

If issues persist, check:
- System logs: `tail -f pi_controller.log`
- Arduino serial output
- Browser console for JavaScript errors
- Network connectivity between Pi and browser
