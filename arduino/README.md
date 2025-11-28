# Arduino Firmware: Hybrid Tomato-Picking System

## Overview

Complete firmware for Arduino UNO R4 WiFi to control a 6-DOF robotic arm for automated tomato picking and sorting.

## Hardware Requirements

- **Arduino UNO R4 WiFi**
- **6 Servos**:
  - 3x SG90 (Claw, Pitch, Elbow) on D2-D4
  - 3x MG99x/MG996R (Forearm, Shoulder, Base) on D5-D7
- **VL53L0X** Time-of-Flight sensor (I2C)
- **External Power Supply**: 5-6V @ 2-5A for servos
- **Emergency Stop Switch** (optional, D8)

## Pin Configuration

| Pin | Component | Type |
|-----|-----------|------|
| D2 | Claw Servo | SG90 |
| D3 | Pitch Servo | SG90 |
| D4 | Elbow Servo | SG90 |
| D5 | Forearm Servo | MG99x |
| D6 | Shoulder Servo | MG99x |
| D7 | Base Servo | MG99x |
| D8 | Emergency Stop | Digital Input |
| SDA/SCL | VL53L0X | I2C |

## Software Structure

```
arduino/src/
├── main.ino              # Main program loop
├── config.h              # Pin definitions and configuration
├── servo_manager.h/cpp   # Servo control with safety limits
├── tof_vl53.h/cpp        # VL53L0X sensor management
├── comm_client.h/cpp     # WebSocket/WiFi communication
├── motion_planner.h/cpp  # Pick sequence state machine
└── calibration.h/cpp     # EEPROM calibration storage
```

## Features

### ✅ Implemented

1. **Modular Architecture**: Clean separation of concerns
2. **Safety Limits**: Per-joint angle limits enforced
3. **Smooth Motion**: Speed-controlled servo movement
4. **ToF Integration**: Distance-based fine positioning
5. **WebSocket Communication**: Real-time command/telemetry
6. **Pick Sequence**: Complete state machine (Approach → Grasp → Lift → Bin → Home)
7. **Calibration Storage**: EEPROM-based calibration persistence
8. **Emergency Stop**: Hardware and software stop
9. **Homing Sequence**: All servos to 90° on startup

### Safety Features

- **Joint Limits**: Enforced per servo type
- **Speed Control**: Prevents sudden movements
- **Emergency Stop**: Immediate halt on D8 or "stop" command
- **Timeout Protection**: 10-second state timeout
- **Range Validation**: ToF distance checks before approach

## Configuration

Edit `config.h` to configure:

- WiFi credentials (`WIFI_SSID`, `WIFI_PASS`)
- WebSocket server (`WS_HOST`, `WS_PORT`)
- Servo limits (if different from defaults)
- Motion parameters (speed, approach distance, etc.)

## Upload Instructions

1. **Install Arduino IDE** (latest version)
2. **Install Board Support**: 
   - Tools → Board → Boards Manager
   - Search "Arduino UNO R4 WiFi" and install
3. **Install Libraries**:
   - Tools → Manage Libraries
   - Install: `Servo`, `Adafruit_VL53L0X`, `ArduinoWebsockets`, `ArduinoJson`
4. **Open Project**:
   - File → Open → `arduino/src/main.ino`
5. **Configure**:
   - Edit `config.h` with your WiFi credentials and server IP
6. **Upload**:
   - Select board: Tools → Board → Arduino UNO R4 WiFi
   - Select port: Tools → Port → (your port)
   - Click Upload

## First Run

1. **Power On**: Arduino boots, servos home to 90°
2. **WiFi Connection**: Arduino connects to WiFi network
3. **WebSocket Connection**: Arduino connects to web server
4. **Ready**: System waits for commands

## Commands

### Via WebSocket

**Pick Command**:
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

**Manual Move**:
```json
{
  "cmd": "move_joints",
  "base": 90,
  "shoulder": 45,
  "forearm": 90,
  "elbow": 90,
  "pitch": 90,
  "claw": 0
}
```

**System Commands**:
```json
{"cmd": "home"}
{"cmd": "stop"}
{"cmd": "set_mode", "mode": "AUTO"}
```

## Calibration

### Servo Trim Calibration

1. Physically position each servo to 90°
2. Send calibration command via web interface
3. Trims saved to EEPROM

### Bin Position Calibration

1. Manually move arm to right bin (ripe) position
2. Record servo angles
3. Repeat for left bin (unripe)
4. Save to EEPROM

### Pixel-to-Robot Mapping

Use the calibration wizard (`calibration/pixel_to_servo_wizard.py`) to:
1. Capture calibration points (pixel + ToF + servo angles)
2. Generate distance-based lookup table
3. Export for use in motion planner

## Troubleshooting

### Servos Don't Move
- Check power supply (external 5-6V)
- Verify ground connections
- Check pin assignments in `config.h`

### WiFi Not Connecting
- Verify SSID and password in `config.h`
- Check network availability
- Verify WiFi antenna (R4 WiFi has built-in)

### WebSocket Not Connecting
- Verify web server IP in `config.h`
- Check web server is running
- Verify firewall allows connections

### Pick Sequence Fails
- Check ToF sensor readings
- Verify joint limits allow movement
- Check bin positions are reachable

## Safety Warnings

⚠️ **IMPORTANT**:
- Always test in MANUAL mode first
- Verify emergency stop works before automation
- Start with slow speeds and small movements
- Keep hands clear during automatic operation
- Power on Arduino BEFORE enabling servo power supply

## Documentation

- **Wiring Diagram**: See `WIRING_DIAGRAM.md`
- **Commissioning**: See `COMMISSIONING_CHECKLIST.md`
- **API Contract**: See `web/api_contract.md`
