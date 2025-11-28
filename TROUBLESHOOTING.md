# Troubleshooting Guide

## Issue 1: Servos Moving on Power-Up

### Problem
Servos move when Arduino is powered on, even though code doesn't attach them.

### Root Cause
This is a **hardware/power sequencing issue**, not a code issue:
1. **External power supply timing**: If servo power supply is turned on before Arduino boots, servos will move
2. **Physical servo behavior**: Servos naturally move to center position (90°) when they first receive power
3. **Pin state during boot**: Arduino pins may be in undefined state during boot sequence

### Solutions

#### Solution 1: Power Sequencing (Recommended)
**Power on in this order:**
1. Power on Arduino FIRST
2. Wait for Arduino to boot (see "Ready for commands" message)
3. THEN turn on external servo power supply

#### Solution 2: Hardware Relay Control
Use a relay or MOSFET controlled by Arduino to enable servo power:
- Connect relay to Arduino pin
- Servo power goes through relay
- Arduino enables relay only after boot is complete
- This gives full software control

#### Solution 3: Accept Brief Movement
If movement is minimal and acceptable, you can ignore it. The servos will only move once on power-up, then remain stationary until commanded.

### Code Protection
The firmware already:
- Sets all servo pins to INPUT mode immediately on boot
- Does NOT attach servos until first command
- Prevents any code-initiated movement

## Issue 2: Commands Not Reaching Arduino via Bluetooth

### Problem
Arduino shows "BLE Connected: YES" but commands from web interface don't work.

### Root Cause
The Arduino's "BLE Connected" status shows connection from **OS Bluetooth stack**, but the **Python BLE client** may not be connected.

### Solutions

#### Solution 1: Check Python BLE Connection
1. Check web interface status: `http://localhost:5000`
2. Look for "Bluetooth: Connected" (green badge)
3. If it shows "Disconnected", the Python client isn't connected

#### Solution 2: Disconnect OS Bluetooth First
If OS Bluetooth is connected, disconnect it first:
```bash
bluetoothctl
disconnect 64:E8:33:69:47:65  # Replace with your device address
exit
```

Then restart the web app:
```bash
pkill -f web_interface.py
./start.sh
```

#### Solution 3: Manual Connection via Web Interface
1. Go to Control page: `http://localhost:5000/control`
2. Click "Scan for Devices" in Bluetooth section
3. Find "FarmBot" in the list
4. Click on it to connect

#### Solution 4: Check Serial Monitor
Watch Arduino Serial Monitor (115200 baud) for:
- "BLE Command received: [COMMAND]" - Command received successfully
- No message - Command not received

### Debugging Steps

1. **Check BLE Status:**
   ```bash
   curl http://localhost:5000/api/bluetooth/status
   ```
   Should show: `"ble_connected": true`

2. **Test Command Sending:**
   ```bash
   curl -X POST http://localhost:5000/api/arm/home
   ```
   Check Serial Monitor for "BLE Command received: [HOME]"

3. **Check Web App Logs:**
   Look for messages like:
   - "Sent BLE command: HOME"
   - "BLE write error: ..."

4. **Verify Device Name:**
   - Arduino advertises as: "FarmBot"
   - Python client looks for: "FarmBot"
   - Must match exactly!

## Issue 3: Servos Not Attaching on Command

### Problem
Commands are received but servos don't move (Servos Attached: NO).

### Solution
This is expected behavior! Servos only attach when first movement command is received. The firmware will:
1. Receive command
2. Attach servos
3. Move to commanded position

If servos don't attach, check:
- Command is being received (see Serial Monitor)
- No emergency stop is active
- Command format is correct

## Quick Fixes

### Restart Everything
```bash
# 1. Kill web app
pkill -f web_interface.py

# 2. Disconnect OS Bluetooth (if connected)
bluetoothctl disconnect 64:E8:33:69:47:65

# 3. Restart web app
./start.sh

# 4. Wait 10 seconds for BLE connection
# 5. Check status at http://localhost:5000
```

### Test Connection
```bash
# Test BLE status
curl http://localhost:5000/api/bluetooth/status

# Test command sending
curl -X POST http://localhost:5000/api/arm/home

# Check Arduino Serial Monitor for "BLE Command received: [HOME]"
```

## Common Issues Summary

| Issue | Symptom | Solution |
|-------|---------|----------|
| Servos move on power-up | Arm moves when Arduino powers on | Power Arduino first, then servo supply |
| Commands not received | Arduino shows BLE connected but no commands | Disconnect OS Bluetooth, restart web app |
| Web shows disconnected | Status shows "Disconnected" | Check BLE connection, restart web app |
| Servos don't attach | Commands received but no movement | This is normal - servos attach on first command |

---

**Need more help?** Check the Serial Monitor on Arduino and web app logs for detailed error messages.

