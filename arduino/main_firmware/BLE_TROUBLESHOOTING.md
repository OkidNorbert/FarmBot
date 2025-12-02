# BLE Connection Troubleshooting Guide

## Quick Checklist

1. ✅ **Board Selection**: Arduino UNO R4 WiFi (NOT Arduino Nano R4 or classic Uno)
2. ✅ **Arduino is advertising**: Check Serial Monitor for "BLE device advertising as: FarmBot"
3. ✅ **Python client can scan**: Run `python test_arduino_ble.py` to scan for devices
4. ✅ **Bluetooth enabled**: Check `bluetoothctl` or system settings
5. ✅ **Permissions**: User in bluetooth group

## Step-by-Step Debugging

### Step 1: Verify Arduino BLE is Working

1. **Upload firmware** to Arduino UNO R4 WiFi
2. **Open Serial Monitor** (115200 baud)
3. **Look for these messages**:
   ```
   ========================================
   Initializing BLE...
   ========================================
   ✅ BLE hardware initialized
   ✅ Device name set to: FarmBot
   ✅ Service created: 19B10000-E8F2-537E-4F6C-D104768A1214
   ✅ Command characteristic created: 19B10001-E8F2-537E-4F6C-D104768A1214
   ✅ Telemetry characteristic created
   ✅ Characteristics added to service
   ✅ Service added to BLE
   ========================================
   📡 BLE device advertising as: FarmBot
   ...
   ⏳ Waiting for BLE central to connect...
   ```

4. **If you see errors**:
   - `"BLE initialization failed!"` → Wrong board selected (must be Arduino UNO R4 WiFi)
   - `"Unsupported board selected!"` → Wrong board architecture (must be Renesas, not AVR)

### Step 2: Test BLE Scanning from Python

Run the test script:
```bash
cd /home/okidi6/Documents/GitHub/emebeded
source tomato_sorter_env/bin/activate
python test_arduino_ble.py
```

**Expected output if working**:
```
🔍 Scanning for Arduino BLE device...
Looking for device named: FarmBot
------------------------------------------------------------

📡 Found X BLE device(s):
------------------------------------------------------------
  • FarmBot (AA:BB:CC:DD:EE:FF)
  ✅ MATCH! This is your Arduino!
------------------------------------------------------------

✅ Arduino found: FarmBot
   Address: AA:BB:CC:DD:EE:FF
```

**If Arduino not found**:
- Check Arduino Serial Monitor - is it advertising?
- Check Bluetooth is enabled: `bluetoothctl power on`
- Try moving devices closer together
- Check for interference (WiFi, other Bluetooth devices)

### Step 3: Test Connection

The test script will automatically try to connect:
```
🔌 Attempting to connect to FarmBot...
✅ Connected to Arduino!
   Service UUID: 19B10000-E8F2-537E-4F6C-D104768A1214
   Characteristic UUID: 19B10001-E8F2-537E-4F6C-D104768A1214

📋 Available services: 1
   Service: 19b10000-e8f2-537e-4f6c-d104768a1214
     Characteristic: 19b10001-e8f2-537e-4f6c-d104768a1214

📤 Testing command send...
✅ Command sent successfully!
```

**If connection fails**:
- Check Arduino Serial Monitor for connection messages
- Verify UUIDs match between Arduino and Python
- Try resetting Arduino
- Check if another device is already connected

### Step 4: Check Web Interface Connection

1. **Start web interface**:
   ```bash
   ./start.sh
   ```

2. **Check connection status**:
   ```bash
   curl http://localhost:5000/api/bluetooth/status
   ```

3. **Scan for devices**:
   ```bash
   curl -X POST http://localhost:5000/api/bluetooth/scan
   ```

4. **Connect manually**:
   ```bash
   curl -X POST http://localhost:5000/api/bluetooth/connect \
     -H "Content-Type: application/json" \
     -d '{"address": "AA:BB:CC:DD:EE:FF", "name": "FarmBot"}'
   ```

## Common Issues and Solutions

### Issue 1: "BLE initialization failed!"

**Cause**: Wrong board selected in Arduino IDE

**Solution**:
1. Go to **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi**
2. NOT "Arduino Nano R4" or "Arduino UNO R4" (without WiFi)
3. Re-upload firmware

### Issue 2: "Unsupported board selected!" Compilation Error

**Cause**: Trying to compile for AVR architecture (classic Arduino Uno/Nano)

**Solution**:
- ArduinoBLE library only supports: samd, megaavr, mbed, esp32, renesas, etc.
- Must use Arduino UNO R4 WiFi (Renesas architecture)
- If you must use AVR board, disable BLE in config.h

### Issue 3: Arduino Not Found in Scan

**Possible Causes**:
1. **Arduino not advertising**:
   - Check Serial Monitor for "BLE device advertising" message
   - If not advertising, BLE.begin() may have failed

2. **Bluetooth disabled**:
   ```bash
   bluetoothctl
   power on
   scan on
   ```

3. **Permissions issue**:
   ```bash
   sudo usermod -aG bluetooth $USER
   # Log out and back in
   ```

4. **Device name mismatch**:
   - Arduino advertises as "FarmBot" (from config.h)
   - Python looks for "FarmBot"
   - Make sure they match

5. **Range/Interference**:
   - BLE range is ~10 meters
   - Move devices closer
   - Check for WiFi interference

### Issue 4: Connection Timeout

**Possible Causes**:
1. **Another device connected**:
   - BLE allows only one connection at a time
   - Disconnect other devices first

2. **Arduino not responding**:
   - Reset Arduino
   - Check Serial Monitor for errors

3. **UUID mismatch**:
   - Verify SERVICE_UUID and CHAR_UUID match in:
     - Arduino: `config.h`
     - Python: `hardware_controller.py` or `test_arduino_ble.py`

### Issue 5: Connected But Commands Don't Work

**Possible Causes**:
1. **Callback not set**:
   - Check Arduino Serial Monitor for "⚠️ Warning: No message callback set!"
   - This means `commClient.onMessage()` wasn't called

2. **Command format wrong**:
   - Check Arduino Serial Monitor for received commands
   - Verify command format matches what Arduino expects

3. **Characteristic not writable**:
   - Verify CHAR_UUID has BLEWrite permission
   - Check Arduino code: `BLERead | BLEWrite | BLENotify`

## Debugging Commands

### Check Bluetooth Status (Linux)
```bash
# Check if Bluetooth is running
sudo systemctl status bluetooth

# Start Bluetooth
sudo systemctl start bluetooth

# Enable Bluetooth
bluetoothctl power on

# Scan for devices
bluetoothctl scan on
```

### Check Python BLE Library
```bash
source tomato_sorter_env/bin/activate
python -c "from bleak import BleakScanner; print('Bleak OK')"
```

### Monitor Arduino Serial Output
```bash
# If using serial connection
screen /dev/ttyACM0 115200

# Or use Arduino IDE Serial Monitor
```

### Test BLE Connection Manually
```python
from bleak import BleakScanner, BleakClient
import asyncio

async def test():
    # Scan
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        print(f"{d.name}: {d.address}")
    
    # Connect
    device = await BleakScanner.find_device_by_name("FarmBot")
    if device:
        async with BleakClient(device) as client:
            print("Connected!")
            # Send command
            await client.write_gatt_char(
                "19B10001-E8F2-537E-4F6C-D104768A1214",
                b'{"cmd":"home"}'
            )

asyncio.run(test())
```

## Verification Checklist

- [ ] Arduino Serial Monitor shows "BLE device advertising as: FarmBot"
- [ ] `test_arduino_ble.py` can find "FarmBot" device
- [ ] `test_arduino_ble.py` can connect successfully
- [ ] Arduino Serial Monitor shows "✅ BLE CONNECTION ESTABLISHED!"
- [ ] Commands sent from Python appear in Arduino Serial Monitor
- [ ] Web interface shows "ble_connected: true" in status
- [ ] Arm responds to commands from web interface

## Still Not Working?

1. **Check Arduino Serial Monitor** - Look for error messages
2. **Check Python logs** - Look for connection errors
3. **Try resetting Arduino** - Power cycle the board
4. **Try restarting Bluetooth service**:
   ```bash
   sudo systemctl restart bluetooth
   ```
5. **Check system logs**:
   ```bash
   journalctl -u bluetooth -f
   ```

## Getting Help

If still having issues, provide:
1. Arduino Serial Monitor output (full startup sequence)
2. Python test script output
3. Board selection in Arduino IDE (Tools → Board)
4. Error messages from web interface
5. Output of `bluetoothctl devices`

