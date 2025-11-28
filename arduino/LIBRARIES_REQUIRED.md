# Required Libraries for Arduino Firmware

## Complete Library List

To upload `arduino/src/main.ino` to your Arduino UNO R4 WiFi, you need to install the following libraries:

### 1. **ArduinoJson** (by Benoit Blanchon)
   - **Version**: 6.x or 7.x (recommended: latest)
   - **Purpose**: JSON parsing for WebSocket/BLE commands
   - **Install**: Tools → Manage Libraries → Search "ArduinoJson"
   - **Required for**: Command parsing, telemetry formatting

### 2. **Servo** (Built-in)
   - **Version**: Included with Arduino IDE
   - **Purpose**: Servo motor control
   - **Install**: Already included, no installation needed
   - **Required for**: All servo operations

### 3. **Adafruit_VL53L0X** (by Adafruit)
   - **Version**: Latest
   - **Purpose**: VL53L0X Time-of-Flight sensor driver
   - **Install**: Tools → Manage Libraries → Search "Adafruit VL53L0X"
   - **Note**: Also install "Adafruit Unified Sensor" (dependency)
   - **Required for**: Distance measurement

### 4. **ArduinoWebsockets** (by Links2004)
   - **Version**: Latest
   - **Purpose**: WebSocket client for WiFi communication
   - **Install**: Tools → Manage Libraries → Search "ArduinoWebsockets"
   - **Required for**: WiFi/WebSocket communication (if using WiFi mode)

### 5. **WiFiS3** (Built-in for R4 WiFi)
   - **Version**: Included with Arduino R4 WiFi board package
   - **Purpose**: WiFi connectivity
   - **Install**: Automatically included when you install Arduino UNO R4 WiFi board support
   - **Required for**: WiFi communication

### 6. **ArduinoBLE** (by Arduino)
   - **Version**: Latest
   - **Purpose**: Bluetooth Low Energy support
   - **Install**: Tools → Manage Libraries → Search "ArduinoBLE"
   - **Required for**: Bluetooth communication (if using BLE mode)

### 7. **Wire** (Built-in)
   - **Version**: Included with Arduino IDE
   - **Purpose**: I2C communication for VL53L0X
   - **Install**: Already included, no installation needed
   - **Required for**: ToF sensor communication

### 8. **EEPROM** (Built-in)
   - **Version**: Included with Arduino IDE
   - **Purpose**: Non-volatile storage for calibration data
   - **Install**: Already included, no installation needed
   - **Required for**: Calibration persistence

## Installation Steps

### Step 1: Install Arduino IDE
1. Download Arduino IDE 2.x or 1.8.x from [arduino.cc](https://www.arduino.cc/en/software)
2. Install the IDE

### Step 2: Install Board Support
1. Open Arduino IDE
2. Go to **Tools → Board → Boards Manager**
3. Search for **"Arduino UNO R4 WiFi"**
4. Install the board package (includes WiFiS3 library)

### Step 3: Install Required Libraries
1. Go to **Tools → Manage Libraries**
2. Install each library in this order:

   ```
   1. ArduinoJson (by Benoit Blanchon)
   2. Adafruit Unified Sensor (dependency for VL53L0X)
   3. Adafruit VL53L0X (by Adafruit)
   4. ArduinoWebsockets (by Links2004)
   5. ArduinoBLE (by Arduino)
   ```

### Step 4: Verify Installation
1. Open `arduino/src/main.ino`
2. Go to **Sketch → Verify/Compile**
3. If compilation succeeds, all libraries are installed correctly
4. If errors occur, check which library is missing and install it

## Library Installation via Library Manager

### Quick Install Commands (if using CLI)
```bash
# Not directly available, but you can use Arduino CLI:
arduino-cli lib install "ArduinoJson"
arduino-cli lib install "Adafruit Unified Sensor"
arduino-cli lib install "Adafruit VL53L0X"
arduino-cli lib install "ArduinoWebsockets"
arduino-cli lib install "ArduinoBLE"
```

## Optional Libraries (for advanced features)

### **SD** (Built-in)
- For logging to SD card (if you add SD card support later)

### **SPI** (Built-in)
- For SPI communication (if you add SPI devices later)

## Library Versions Compatibility

| Library | Minimum Version | Recommended Version |
|---------|----------------|---------------------|
| ArduinoJson | 6.0.0 | 7.x (latest) |
| Adafruit_VL53L0X | 1.0.0 | Latest |
| ArduinoWebsockets | 0.5.0 | Latest |
| ArduinoBLE | 1.3.0 | Latest |

## Troubleshooting

### Error: "No such file or directory: ArduinoJson.h"
- **Solution**: Install ArduinoJson library via Library Manager

### Error: "WiFiS3.h: No such file or directory"
- **Solution**: Install Arduino UNO R4 WiFi board package

### Error: "Adafruit_VL53L0X.h: No such file or directory"
- **Solution**: Install both "Adafruit Unified Sensor" and "Adafruit VL53L0X"

### Error: "ArduinoBLE.h: No such file or directory"
- **Solution**: Install ArduinoBLE library (only needed if using BLE mode)

### Error: "ArduinoWebsockets.h: No such file or directory"
- **Solution**: Install ArduinoWebsockets library (only needed if using WiFi mode)

## Conditional Compilation

The firmware uses conditional compilation, so:
- If `USE_WIFI` is defined: WiFi libraries are required
- If `USE_BLE` is defined: BLE library is required
- If both are defined: Both sets of libraries are required

By default, both are enabled in `config.h`.

## Summary Checklist

Before uploading, ensure you have:

- [ ] Arduino IDE installed
- [ ] Arduino UNO R4 WiFi board package installed
- [ ] ArduinoJson library installed
- [ ] Adafruit Unified Sensor library installed
- [ ] Adafruit VL53L0X library installed
- [ ] ArduinoWebsockets library installed (for WiFi)
- [ ] ArduinoBLE library installed (for BLE)
- [ ] All libraries verified via Sketch → Verify/Compile

## Quick Reference

**Minimum Required** (if using BLE only):
- ArduinoJson
- ArduinoBLE
- Servo (built-in)
- Wire (built-in)

**Full Required** (WiFi + BLE):
- ArduinoJson
- ArduinoBLE
- ArduinoWebsockets
- WiFiS3 (from board package)
- Adafruit VL53L0X
- Adafruit Unified Sensor
- Servo (built-in)
- Wire (built-in)
- EEPROM (built-in)

