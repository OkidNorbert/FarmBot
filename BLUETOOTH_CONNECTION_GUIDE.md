# 📡 Bluetooth Connection Guide - Arduino R4 WiFi

Complete guide for connecting your Linux laptop to Arduino R4 WiFi via Bluetooth from the web interface.

## 🎯 Overview

This guide will help you:
- Set up Bluetooth on your Linux laptop
- Connect to Arduino R4 WiFi via Bluetooth Low Energy (BLE)
- Control the robotic arm through the web interface
- Troubleshoot connection issues

## 📋 Prerequisites

### Hardware Requirements:
- ✅ Arduino R4 WiFi board
- ✅ Linux laptop/computer with Bluetooth support
- ✅ USB cable (for initial Arduino setup)

### Software Requirements:
- ✅ Python 3.8+
- ✅ Bluetooth stack installed on Linux
- ✅ Bleak library (Python BLE library)
- ✅ Web application running on localhost

## 🔧 Step 1: Install Bluetooth Dependencies

### On Linux (Ubuntu/Debian):
```bash
# Install Bluetooth stack
sudo apt-get update
sudo apt-get install -y bluetooth bluez libbluetooth-dev

# Start Bluetooth service
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# Check Bluetooth status
bluetoothctl --version
```

### On Linux (Arch/Manjaro):
```bash
# Install Bluetooth stack
sudo pacman -S bluez bluez-utils

# Start Bluetooth service
sudo systemctl start bluetooth
sudo systemctl enable bluetooth
```

### Install Python BLE Library:
```bash
# Activate your virtual environment
source tomato_sorter_env/bin/activate

# Install Bleak (already in requirements.txt)
# For Python 3.13+, use bleak>=2.0.0
pip install bleak>=2.0.0

# Or install from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "from bleak import BleakScanner; print('Bleak installed successfully')"
```

## 🔧 Step 2: Prepare Arduino R4 WiFi

### Upload BLE Firmware to Arduino:

1. **Open Arduino IDE** and create a new sketch
2. **Install ArduinoBLE library** (if not already installed):
   - Go to `Tools` → `Manage Libraries`
   - Search for "ArduinoBLE"
   - Install the library by Arduino

3. **Upload this BLE sketch** to your Arduino R4 WiFi:

```cpp
#include <ArduinoBLE.h>

// BLE Service UUID
BLEService armService("19B10000-E8F2-537E-4F6C-D104768A1214");

// BLE Characteristic UUID for commands
BLEStringCharacteristic commandChar("19B10001-E8F2-537E-4F6C-D104768A1214", 
                                    BLERead | BLEWrite, 128);

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }
  
  // Set local name
  BLE.setLocalName("Arduino");
  BLE.setAdvertisedService(armService);
  
  // Add characteristic to service
  armService.addCharacteristic(commandChar);
  
  // Add service
  BLE.addService(armService);
  
  // Start advertising
  BLE.advertise();
  
  Serial.println("BLE device is now advertising as 'Arduino'");
  Serial.println("Waiting for connections...");
}

void loop() {
  // Wait for BLE central to connect
  BLEDevice central = BLE.central();
  
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());
    
    while (central.connected()) {
      // Check if command received
      if (commandChar.written()) {
        String command = commandChar.value();
        Serial.print("Received command: ");
        Serial.println(command);
        
        // Process command here
        // Example: parse and execute arm movements
        processCommand(command);
      }
    }
    
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}

void processCommand(String cmd) {
  // Parse and execute commands
  // Example commands:
  // HOME - Home the arm
  // MOVE x y z - Move to coordinates
  // GRIPPER OPEN/CLOSE - Control gripper
  
  if (cmd.startsWith("HOME")) {
    // Home arm code here
    Serial.println("Executing: HOME");
  } else if (cmd.startsWith("MOVE")) {
    // Parse coordinates and move
    Serial.print("Executing: MOVE ");
    Serial.println(cmd.substring(5));
  } else if (cmd.startsWith("GRIPPER")) {
    // Control gripper
    Serial.print("Executing: GRIPPER ");
    Serial.println(cmd.substring(8));
  }
}
```

4. **Upload the sketch** to your Arduino R4 WiFi
5. **Open Serial Monitor** (115200 baud) to verify BLE is advertising

## 🌐 Step 3: Connect via Web Interface

### Method 1: Using the Web Interface (Recommended)

1. **Start the web application**:
   ```bash
   cd /home/okidi6/Documents/GitHub/emebeded
   source tomato_sorter_env/bin/activate
   ./start.sh
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Navigate to the Control Panel** (or use the API endpoints directly)

4. **Scan for Bluetooth devices**:
   - Use the `/api/bluetooth/scan` endpoint or the UI button
   - Look for device named "Arduino" or your custom name

5. **Connect to the device**:
   - Click on the device in the list, or
   - Use the `/api/bluetooth/connect` endpoint with the device address

### Method 2: Using API Endpoints Directly

#### Scan for Devices:
```bash
curl -X POST http://localhost:5000/api/bluetooth/scan
```

Response:
```json
{
  "success": true,
  "devices": [
    {
      "name": "Arduino",
      "address": "AA:BB:CC:DD:EE:FF",
      "rssi": -45
    }
  ]
}
```

#### Connect to Device:
```bash
curl -X POST http://localhost:5000/api/bluetooth/connect \
  -H "Content-Type: application/json" \
  -d '{
    "address": "AA:BB:CC:DD:EE:FF",
    "name": "Arduino"
  }'
```

#### Check Connection Status:
```bash
curl http://localhost:5000/api/bluetooth/status
```

#### Disconnect:
```bash
curl -X POST http://localhost:5000/api/bluetooth/disconnect
```

## 🔍 Step 4: Verify Connection

### Check Connection Status:
```bash
# Via API
curl http://localhost:5000/api/bluetooth/status

# Expected response:
{
  "success": true,
  "connected": true,
  "connection_type": "bluetooth",
  "ble_connected": true,
  "ble_address": "AA:BB:CC:DD:EE:FF"
}
```

### Test Arm Control:
```bash
# Home the arm
curl -X POST http://localhost:5000/api/arm/home

# Move arm
curl -X POST http://localhost:5000/api/arm/move \
  -H "Content-Type: application/json" \
  -d '{"x": 100, "y": 150, "z": 50}'
```

## 🛠️ Step 5: Configure for Automatic Connection

### Option 1: Modify Hardware Controller Initialization

Edit `hardware_controller.py` or pass connection type when initializing:

```python
# In web_interface.py, modify the initialization:
hw_controller = HardwareController(
    connection_type='bluetooth',  # or 'auto' to try both
    ble_device_name='Arduino'     # Name of your BLE device
)
```

### Option 2: Use Environment Variables

Create a `.env` file:
```bash
ARDUINO_CONNECTION_TYPE=bluetooth
BLE_DEVICE_NAME=Arduino
```

## 🐛 Troubleshooting

### Issue: "Bleak library not available"
**Solution:**
```bash
source tomato_sorter_env/bin/activate
# For Python 3.13+, use bleak>=2.0.0
pip install bleak>=2.0.0
# Or for older Python versions (<3.13), use:
# pip install bleak==0.21.1
```

### Issue: "Bluetooth device not found"
**Solutions:**
1. **Check Bluetooth is enabled**:
   ```bash
   bluetoothctl
   power on
   scan on
   ```

2. **Check Arduino is advertising**:
   - Open Serial Monitor on Arduino IDE
   - Should see "BLE device is now advertising"

3. **Check device name matches**:
   - Default name is "Arduino"
   - Make sure Arduino sketch uses same name

4. **Check permissions**:
   ```bash
   # Add user to bluetooth group
   sudo usermod -aG bluetooth $USER
   # Log out and back in
   ```

### Issue: "Connection timeout"
**Solutions:**
1. **Move devices closer** (BLE range is ~10 meters)
2. **Check for interference** (WiFi, other Bluetooth devices)
3. **Restart Bluetooth service**:
   ```bash
   sudo systemctl restart bluetooth
   ```

### Issue: "Permission denied"
**Solution:**
```bash
# Check if user is in bluetooth group
groups | grep bluetooth

# If not, add user:
sudo usermod -aG bluetooth $USER
# Then log out and back in
```

### Issue: "Device connects but commands don't work"
**Solutions:**
1. **Check Arduino Serial Monitor** for received commands
2. **Verify command format** matches Arduino sketch expectations
3. **Check BLE characteristic UUIDs** match in both Arduino and Python code

## 📊 Monitoring Connection

### Check System Logs:
```bash
# View hardware controller logs
tail -f /var/log/syslog | grep -i bluetooth

# Or check application output
# (if running in foreground)
```

### Test BLE Connection Manually:
```python
# Python test script
from bleak import BleakScanner
import asyncio

async def scan():
    devices = await BleakScanner.discover(timeout=10.0)
    for device in devices:
        print(f"Found: {device.name} - {device.address}")

asyncio.run(scan())
```

## 🔐 Security Notes

- **BLE is relatively secure** but not encrypted by default
- **Use pairing** for production deployments
- **Limit BLE range** if security is a concern
- **Consider WiFi** for more secure connections in production

## 📝 Command Reference

### Available API Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bluetooth/scan` | POST | Scan for BLE devices |
| `/api/bluetooth/connect` | POST | Connect to device |
| `/api/bluetooth/disconnect` | POST | Disconnect from device |
| `/api/bluetooth/status` | GET | Get connection status |
| `/api/arm/move` | POST | Move arm to coordinates |
| `/api/arm/home` | POST | Home the arm |
| `/api/arm/gripper` | POST | Control gripper |

### Example Commands for Arduino:

```
HOME              - Return arm to home position
MOVE 100 150 50   - Move to coordinates (x, y, z)
GRIPPER OPEN      - Open gripper
GRIPPER CLOSE     - Close gripper
STATUS            - Get arm status
```

## 🎉 Success Checklist

- [ ] Bluetooth stack installed and running
- [ ] Bleak library installed
- [ ] Arduino R4 WiFi BLE sketch uploaded
- [ ] Arduino is advertising (visible in scan)
- [ ] Web application can scan and find device
- [ ] Connection established successfully
- [ ] Commands can be sent and received
- [ ] Arm responds to commands

## 📚 Additional Resources

- **Bleak Documentation**: https://bleak.readthedocs.io/
- **ArduinoBLE Library**: https://www.arduino.cc/reference/en/libraries/arduinoble/
- **Arduino R4 WiFi Docs**: https://docs.arduino.cc/hardware/uno-r4-wifi

## 💡 Tips

1. **Keep devices close** during initial connection
2. **Use consistent device names** for easier identification
3. **Monitor Serial Monitor** on Arduino for debugging
4. **Test connection** before running automated sequences
5. **Keep Arduino powered** during operation

---

**Need Help?** Check the troubleshooting section or review the application logs for detailed error messages.

