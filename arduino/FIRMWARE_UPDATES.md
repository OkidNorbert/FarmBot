# Arduino Firmware Updates - Web-Only Control

## ✅ Changes Made

### 1. **Removed Automatic Arm Movement on Startup**
- **Before**: Arm automatically moved to home position when powered on
- **After**: Arm remains stationary until commanded via web interface
- **Location**: Removed `moveToHome()` call from `setup()`
- **Servo Pin Control**: All servo pins set to INPUT mode on startup to prevent any signal
- **Note**: If servos still move on power-up, it's a hardware issue:
  - External power supply may be turned on before Arduino boots
  - Servos may physically move when they first receive power (normal behavior)
  - **Solution**: Power on Arduino first, then external servo power supply, OR use a relay to control servo power

### 2. **Enhanced Serial Monitor Output**
- **TOF Distance Readings**: Displayed every 1 second
  - Format: `[TOF] Distance: XXXmm (XX.Xcm)`
- **Status Reports**: Displayed every 5 seconds
  - Shows servo angles, emergency stop status, BLE connection
- **Command Echo**: All commands (BLE and Serial) are echoed to monitor

### 3. **Bluetooth Support**
- ✅ Already implemented with ArduinoBLE library
- ✅ BLE Service UUID: `19B10000-E8F2-537E-4F6C-D104768A1214`
- ✅ BLE Characteristic UUID: `19B10001-E8F2-537E-4F6C-D104768A1214`
- ✅ Device advertises as "FarmBot" (can be changed to "Arduino")

### 4. **TOF Sensor Monitoring**
- ✅ VL53L0X sensor readings displayed on serial monitor
- ✅ Sensor availability tracked and reported
- ✅ Distance readings in both mm and cm

## 📊 Serial Monitor Output Examples

### On Startup:
```
Tomato Sorter Arduino - Ready
VL53L0X TOF sensor initialized successfully
TOF Test Reading: 15.3cm
BLE FarmBot Ready
STATUS: Waiting for commands from web interface
NOTE: Arm will NOT move automatically - control via web only
Servos attached - ready for commands
```

### Periodic Output (Every 1 second):
```
[TOF] Distance: 153mm (15.3cm)
```

### Periodic Status (Every 5 seconds):
```
=== STATUS REPORT ===
Emergency Stop: INACTIVE
Servo Angles: Base=90°, Shoulder=90°, Elbow=90°, WristY=90°, WristP=90°, Gripper=30°
BLE Connected: YES
===================
```

### Command Response:
```
BLE Command: HOME
Moving to home position
OK: HOME
```

## 🔧 Supported Commands

### From Web Interface (via BLE or Serial):

| Command | Format | Description |
|---------|--------|-------------|
| `HOME` | `HOME` | Move arm to home position |
| `MOVE` | `MOVE X Y CLASS` | Move to coordinates and sort |
| `PICK` | `PICK X Y Z CLASS` | Pick from coordinates with depth |
| `ANGLE` | `ANGLE A1 A2 A3 A4 A5 A6` | Set servo angles directly |
| `GRIPPER` | `GRIPPER OPEN/CLOSE` | Control gripper |
| `STATUS` | `STATUS` | Get full status report |
| `DISTANCE` | `DISTANCE` | Get TOF sensor reading |
| `STOP` | `STOP` | Emergency stop |

## 🚀 Upload Instructions

1. **Open Arduino IDE**
2. **Select Board**: Tools → Board → Arduino UNO R4 WiFi
3. **Select Port**: Tools → Port → (your Arduino port)
4. **Install Libraries** (if not already installed):
   - `Servo` (built-in)
   - `Wire` (built-in)
   - `ArduinoBLE` (via Library Manager)
   - `Adafruit VL53L0X` (via Library Manager)
5. **Upload** the sketch
6. **Open Serial Monitor** (115200 baud) to see output

## 📝 Notes

- **No Auto-Movement**: Arm will NOT move on power-up
- **Web Control Only**: All movement must be initiated via web interface
- **TOF Readings**: Displayed continuously on serial monitor
- **BLE Ready**: Firmware supports both Serial and Bluetooth connections
- **Status Monitoring**: Continuous status updates help with debugging

## 🔍 Troubleshooting

### TOF Sensor Not Working:
- Check I2C connections (SDA→A4, SCL→A5)
- Verify 3.3V power to sensor
- Check serial monitor for initialization messages

### BLE Not Connecting:
- Verify Arduino R4 WiFi has BLE capability
- Check device name matches in web interface ("FarmBot" or "Arduino")
- Ensure BLE is enabled on your computer

### No Serial Output:
- Check baud rate is set to 115200
- Verify USB cable connection
- Try different USB port

---

**Updated**: Firmware now supports web-only control with enhanced monitoring

