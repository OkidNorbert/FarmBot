# Bluetooth Setup Guide for Arduino R4 WiFi

## Overview

The new firmware supports **both WiFi/WebSocket AND Bluetooth Low Energy (BLE)** communication. You can choose which method to use, or let it automatically fall back from WiFi to BLE.

## Configuration

In `arduino/src/config.h`, set the communication mode:

```cpp
// Options: "WIFI", "BLE", or "AUTO"
#define COMM_MODE "AUTO"  // Tries WiFi first, falls back to BLE
```

### Communication Modes

1. **"WIFI"**: Only use WiFi/WebSocket connection
2. **"BLE"**: Only use Bluetooth Low Energy
3. **"AUTO"**: Try WiFi first, automatically fall back to BLE if WiFi fails

## BLE Connection Flow

### Arduino Side (Peripheral)
- Arduino advertises as "FarmBot" (configurable in `config.h`)
- Web server connects to Arduino via BLE
- Commands sent via BLE characteristic
- Telemetry received via BLE characteristic

### Web Server Side (Central)
The `hardware_controller.py` already has BLE support built-in:

```python
# In web_interface.py, hardware_controller is initialized with:
hw_controller = HardwareController(connection_type='bluetooth', ble_device_name="FarmBot")
```

## BLE Characteristics

- **Command Characteristic** (UUID: `19B10001-E8F2-537E-4F6C-D104768A1214`)
  - Write: Web server sends commands to Arduino
  - Read/Notify: Arduino can send responses

- **Telemetry Characteristic** (UUID: `19B10002-E8F2-537E-4F6C-D104768A1214`)
  - Read/Notify: Arduino sends telemetry data

## Message Format

### Commands (Web → Arduino)
JSON format:
```json
{
  "cmd": "pick",
  "id": "det123",
  "x": 320,
  "y": 240,
  "class": "ripe",
  "confidence": 0.92
}
```

### Telemetry (Arduino → Web)
JSON format:
```json
{
  "battery_voltage": 12.4,
  "status": "IDLE",
  "last_action": "HOME"
}
```

## Setup Steps

### 1. Upload Firmware
1. Open `arduino/src/main.ino` in Arduino IDE
2. Install **ArduinoBLE** library:
   - Tools → Manage Libraries
   - Search "ArduinoBLE"
   - Install by Arduino
3. Set `COMM_MODE` in `config.h`:
   - `"BLE"` for Bluetooth only
   - `"AUTO"` for WiFi with BLE fallback
4. Upload to Arduino

### 2. Configure Web Server

The web interface will automatically use BLE if:
- `hardware_controller.py` is initialized with `connection_type='bluetooth'`
- OR `connection_type='auto'` and Serial connection fails

### 3. Connect

1. **Arduino**: Powers on and starts BLE advertising
2. **Web Server**: Scans for "FarmBot" BLE device
3. **Connection**: Established automatically
4. **Status**: Check `/pi/status` endpoint to verify connection

## Troubleshooting

### Arduino Not Advertising
- Check Serial monitor for BLE initialization messages
- Verify ArduinoBLE library is installed
- Check if BLE is enabled in `config.h` (`USE_BLE` defined)

### Web Server Can't Find Arduino
- Verify Arduino is powered on and advertising
- Check device name matches `BLE_DEVICE_NAME` in `config.h`
- Use `bluetoothctl` to scan:
  ```bash
  bluetoothctl
  scan on
  # Look for "FarmBot" device
  ```

### Connection Drops
- BLE has limited range (~10m)
- Check for interference
- Verify both devices have Bluetooth enabled

### Commands Not Working
- Check Serial monitor on Arduino for received commands
- Verify JSON format is correct
- Check BLE characteristic is writable

## Advantages of BLE

✅ **No WiFi Required**: Works without network infrastructure
✅ **Low Power**: More efficient than WiFi
✅ **Direct Connection**: Point-to-point communication
✅ **Simple Setup**: No IP addresses or network configuration

## Advantages of WiFi/WebSocket

✅ **Long Range**: Works across network
✅ **Multiple Clients**: Multiple devices can connect
✅ **Standard Protocol**: Uses standard WebSocket/Socket.IO
✅ **Remote Access**: Can access from anywhere on network

## Recommendation

Use **"AUTO"** mode for best flexibility:
- Tries WiFi first (if available and configured)
- Automatically falls back to BLE if WiFi fails
- Best of both worlds!

