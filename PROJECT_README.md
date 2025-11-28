# Hybrid Tomato-Picking System - Complete Implementation

## Project Overview

A complete hybrid automatic/manual tomato-picking system using:
- **Arduino UNO R4 WiFi** as field controller
- **6-DOF robotic arm** with 6 servos (3x SG90 + 3x MG99x)
- **VL53L0X ToF sensor** for distance measurement
- **YOLO-based vision** for tomato detection/classification
- **Web interface** for control and monitoring

## Project Structure

```
emebeded/
├── arduino/                    # Arduino firmware
│   ├── src/                   # Source code
│   │   ├── main.ino          # Main program
│   │   ├── servo_manager.*   # Servo control
│   │   ├── tof_vl53.*        # ToF sensor
│   │   ├── comm_client.*     # WebSocket/WiFi
│   │   ├── motion_planner.*  # Pick sequence
│   │   ├── calibration.*     # EEPROM storage
│   │   └── config.h          # Configuration
│   ├── README.md             # Arduino documentation
│   └── WIRING_DIAGRAM.md     # Wiring guide
│
├── web/                      # Web backend
│   ├── api_contract.md       # API specification
│   └── (web_interface.py)    # Main Flask app
│
├── calibration/              # Calibration tools
│   └── pixel_to_servo_wizard.py  # Calibration wizard
│
├── yolo_service.py           # YOLO detection service
├── hardware_controller.py     # Hardware abstraction
├── web_interface.py          # Flask web server
│
├── COMMISSIONING_CHECKLIST.md # Setup checklist
└── PROJECT_README.md         # This file
```

## Quick Start

### 1. Arduino Setup

1. **Install Arduino IDE** and UNO R4 WiFi board support
2. **Install Libraries**:
   - Servo
   - Adafruit_VL53L0X
   - ArduinoWebsockets
   - ArduinoJson
3. **Configure** `arduino/src/config.h`:
   ```cpp
   #define WIFI_SSID "your_wifi"
   #define WIFI_PASS "your_password"
   #define WS_HOST "192.168.1.100"  // Web server IP
   #define WS_PORT 5000
   ```
4. **Upload** `arduino/src/main.ino` to Arduino

### 2. Web Server Setup

1. **Activate virtual environment**:
   ```bash
   source tomato_sorter_env/bin/activate
   ```

2. **Start web interface**:
   ```bash
   ./start.sh
   # OR
   python web_interface.py
   ```

3. **Access web interface**: http://localhost:5000

### 3. YOLO Service (Optional)

Run YOLO detection service separately:
```bash
python yolo_service.py --model models/tomato/best_model.pth --camera 0
```

### 4. Calibration

Run calibration wizard:
```bash
python calibration/pixel_to_servo_wizard.py
```

## System Modes

### Automatic Mode
- YOLO detects tomatoes in camera feed
- Detections sent to web backend
- Web backend sends pick commands to Arduino
- Arduino executes complete pick sequence
- Tomatoes sorted to appropriate bin (ripe/unripe)

### Manual Mode
- Operator controls arm via web interface
- Individual joint control
- Direct servo angle commands
- Useful for calibration and testing

## Communication Flow

```
YOLO Service → Web Backend → Arduino (via WebSocket)
                ↓
            Web Interface (monitoring/control)
                ↓
            Arduino → Telemetry → Web Backend
```

## Key Features

### ✅ Implemented

1. **Modular Arduino Firmware**
   - Servo manager with safety limits
   - ToF sensor integration
   - WebSocket communication
   - Motion planner with state machine
   - EEPROM calibration storage

2. **Complete Pick Sequence**
   - Approach pose calculation
   - ToF-based fine positioning
   - Grasp and lift
   - Bin routing (ripe/unripe)
   - Return to home

3. **Safety Features**
   - Joint angle limits
   - Emergency stop (hardware + software)
   - Timeout protection
   - Range validation

4. **Web Integration**
   - REST API endpoints
   - WebSocket real-time communication
   - YOLO detection integration
   - Manual control interface

5. **Calibration Tools**
   - Servo trim calibration
   - Bin position calibration
   - Pixel-to-robot mapping wizard

## API Endpoints

### WebSocket (Arduino ↔ Web)

**Commands to Arduino**:
- `{"cmd": "pick", "id": "...", "x": 320, "y": 240, "class": "ripe", "confidence": 0.92}`
- `{"cmd": "move_joints", "base": 90, ...}`
- `{"cmd": "home"}`
- `{"cmd": "stop"}`
- `{"cmd": "set_mode", "mode": "AUTO"}`

**Telemetry from Arduino**:
- `{"battery_voltage": 12.4, "status": "IDLE", ...}`
- `{"id": "...", "status": "SUCCESS", "result": "ripe", "duration_ms": 4500}`

### REST API

- `POST /api/control/mode` - Set system mode
- `POST /api/control/emergency_stop` - Emergency stop
- `POST /api/manual/move` - Manual joint control
- `POST /api/auto/start` - Start auto mode
- `POST /api/auto/stop` - Stop auto mode
- `POST /api/vision/detection` - YOLO detection endpoint

## Hardware Configuration

### Servo Mapping

| Pin | Servo | Type | Range |
|-----|-------|------|-------|
| D2 | Claw | SG90 | 0-90° |
| D3 | Pitch | SG90 | 20-160° |
| D4 | Elbow | SG90 | 15-165° |
| D5 | Forearm | MG99x | 10-170° |
| D6 | Shoulder | MG99x | 15-165° |
| D7 | Base | MG99x | 0-180° |

### Power Requirements

- **Arduino**: USB 5V @ 500mA OR 7-12V via barrel jack
- **Servos**: External 5-6V @ 2-5A (separate supply)
- **VL53L0X**: 3.3V or 5V (check module)

⚠️ **CRITICAL**: All grounds must be connected together!

## Safety Warnings

1. **Power Sequencing**: Arduino first, then servo supply
2. **Emergency Stop**: Test before automation
3. **Manual Testing**: Always test in MANUAL mode first
4. **Clear Workspace**: Keep hands clear during operation
5. **Speed Limits**: Start with slow speeds

## Troubleshooting

### Arduino Issues
- **Servos don't move**: Check power supply and ground connections
- **WiFi not connecting**: Verify credentials in `config.h`
- **WebSocket fails**: Check web server IP and firewall

### Web Interface Issues
- **Port 5000 in use**: Kill existing process or change port
- **Camera not working**: Check camera permissions and index
- **AI Model not loaded**: Verify model file path

### Pick Sequence Issues
- **Approach fails**: Check ToF sensor readings
- **Grasp fails**: Verify claw servo range (0-90°)
- **Bin routing fails**: Check bin positions are reachable

## Documentation

- **Arduino README**: `arduino/README.md`
- **Wiring Diagram**: `arduino/WIRING_DIAGRAM.md`
- **Commissioning**: `COMMISSIONING_CHECKLIST.md`
- **API Contract**: `web/api_contract.md`

## Next Steps

1. **Hardware Assembly**: Follow wiring diagram
2. **Calibration**: Run calibration wizard
3. **Testing**: Use commissioning checklist
4. **Deployment**: Start with manual mode, then enable auto

## Support

For issues or questions:
1. Check troubleshooting section
2. Review commissioning checklist
3. Verify all connections per wiring diagram
4. Check serial monitor for Arduino debug output

---

**System Status**: ✅ Fully Implemented
**Last Updated**: 2024
**Version**: 1.0

