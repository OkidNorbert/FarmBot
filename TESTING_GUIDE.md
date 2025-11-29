# 🧪 Testing Guide - Arduino UNO R4 WiFi

## ✅ Arduino Upload Status

**Upload was successful!** Your firmware is now running on the Arduino.

## 🔍 BLE Connection Test Results

The test script found your Arduino:
- **Device Name**: FarmBot
- **BLE Address**: 64:E8:33:69:47:65
- **Status**: ✅ Advertising and discoverable

## 🌐 Starting the Web Interface

### Method 1: Using Virtual Environment (Recommended)

```bash
cd /home/okidi6/Documents/GitHub/emebeded
source farmbot_env/bin/activate
python web_interface.py
```

### Method 2: Direct Start

```bash
cd /home/okidi6/Documents/GitHub/emebeded
python3 web_interface.py
```

## 📱 Access the Web Interface

Once the server starts, open your web browser:

**Local Access:**
- http://localhost:5000
- http://127.0.0.1:5000

**Network Access (from other devices):**
- http://YOUR_IP_ADDRESS:5000

## 🎮 Testing Arduino Control

### 1. Check Arduino Serial Monitor
- Open Arduino IDE
- Tools → Serial Monitor
- Set baud rate to 115200
- You should see: "BLE FarmBot Ready"

### 2. Test via Web Interface
- Open http://localhost:5000
- Navigate to the control/robotic arm section
- Try sending commands:
  - **HOME** - Return to home position
  - **STATUS** - Get current status
  - **ANGLE** - Set servo angles manually

### 3. Test BLE Connection
```bash
cd /home/okidi6/Documents/GitHub/emebeded
source farmbot_env/bin/activate
python test_arduino_ble.py
```

## 🔧 Available Commands

The Arduino accepts these commands via BLE:

- `{"cmd":"home"}` - Return to home position
- `{"cmd":"stop"}` - Emergency stop
- `{"cmd":"move_joints","base":90,"shoulder":90,...}` - Manual control
- `{"cmd":"pick","id":"test1","x":320,"y":240,"class":"ripe"}` - Pick tomato

## 📊 Monitoring

### Serial Monitor Output
The Arduino will show:
- BLE connection status
- Received commands
- Servo movements
- Status updates

### Web Interface
- Connection status indicator
- Real-time telemetry
- Command history
- System status

## 🐛 Troubleshooting

### Arduino Not Found in BLE Scan
1. Check Serial Monitor - should show "BLE FarmBot Ready"
2. Reset Arduino (press reset button)
3. Make sure Bluetooth is enabled on your computer
4. Try running test script again

### Web Interface Won't Start
1. Check if port 5000 is in use: `lsof -i :5000`
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python version: `python3 --version` (needs 3.7+)

### Commands Not Working
1. Check Serial Monitor for error messages
2. Verify BLE connection is active
3. Check command format (must be valid JSON)
4. Try sending STATUS command first

## ✅ Success Indicators

You'll know everything is working when:
- ✅ Arduino Serial Monitor shows "BLE FarmBot Ready"
- ✅ BLE test script finds "FarmBot" device
- ✅ Web interface loads at http://localhost:5000
- ✅ Web interface shows "Arduino Connected"
- ✅ Commands sent from web interface appear in Serial Monitor

## 🎯 Next Steps

1. **Test Basic Movement**: Send HOME command
2. **Test Manual Control**: Use angle commands to move individual servos
3. **Test Pick Sequence**: Send a pick command with coordinates
4. **Monitor Status**: Check telemetry and status updates

Happy testing! 🚀

