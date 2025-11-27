# Arduino R4 Component Test Suite

This test sketch verifies all components of the Tomato Sorter Robotic Arm system.

## Components Tested

- **6 Servos**: Base (D3), Shoulder (D5), Elbow (D6), Wrist Vert (D9), Wrist Rot (D10), Gripper (D11)
- **VL53L0X Distance Sensor**: I2C on SDA (A4) and SCL (A5), powered by 3.3V

## Installation

1. **Install Required Library**:
   - Open Arduino IDE
   - Go to **Tools → Manage Libraries**
   - Search for "Adafruit VL53L0X"
   - Install "Adafruit VL53L0X" by Adafruit

2. **Select Board**:
   - Go to **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi** (or your specific R4 model)

3. **Select Port**:
   - Go to **Tools → Port** and select your Arduino R4

4. **Upload Sketch**:
   - Click **Upload** button
   - Wait for "Done uploading" message

## Usage

### Serial Monitor Setup

1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. Set line ending to **Newline** (or Both NL & CR)

### Available Commands

| Command | Description |
|---------|-------------|
| `TEST ALL` | Run complete test sequence (sensor + servos + coordinated movement) |
| `TEST SERVOS` | Test all 6 servos individually (0° → 90° → 180° → 90°) |
| `TEST SENSOR` | Take 10 distance measurements from VL53L0X |
| `SERVO <n> <angle>` | Move servo n (1-6) to angle (0-180) |
| `HOME` | Move all servos to home position (90° for joints, 0° for gripper) |
| `STATUS` | Show status of all components |
| `HELP` | Display command help |

### Servo Numbers

- **1** = Base (D3)
- **2** = Shoulder (D5)
- **3** = Elbow (D6)
- **4** = Wrist Vert (D9)
- **5** = Wrist Rot (D10)
- **6** = Gripper (D11)

## Test Procedures

### Quick Test (Recommended First)

1. Upload the sketch
2. Open Serial Monitor (115200 baud)
3. Type `STATUS` and press Enter
4. Verify all servos show "ATTACHED" and sensor shows "OK"

### Full Component Test

1. Type `TEST ALL` and press Enter
2. Watch the test sequence:
   - Sensor test (10 readings)
   - Individual servo tests
   - Coordinated movement test
3. Verify all components respond correctly

### Individual Component Tests

**Test Servos Only:**
```
TEST SERVOS
```

**Test Sensor Only:**
```
TEST SENSOR
```

**Test Single Servo:**
```
SERVO 1 45    (Move Base to 45°)
SERVO 6 180   (Close Gripper)
SERVO 6 0     (Open Gripper)
```

## Expected Results

### Successful Test Output

```
========================================
  Arduino R4 Component Test Suite
========================================

Initializing VL53L0X sensor... OK

Attaching servos...
  Base (D3): OK
  Shoulder (D5): OK
  Elbow (D6): OK
  Wrist Vert (D9): OK
  Wrist Rot (D10): OK
  Gripper (D11): OK

Moving to home position...
========================================
  System Ready!
========================================
```

### Troubleshooting

**Sensor Not Found:**
```
Initializing VL53L0X sensor... FAILED
  ERROR: VL53L0X sensor not found!
  Check wiring: SDA->A4, SCL->A5, VCC->3.3V, GND->GND
```

**Solutions:**
- Verify I2C connections (SDA→A4, SCL→A5)
- Check power (3.3V, not 5V unless module supports it)
- Verify ground connection
- Check if sensor module is working (test with `arduino/test_tof/test_tof.ino` first)

**Servo Not Moving:**
- Check servo power connections (5V and GND from external PSU)
- Verify PWM signal wire is connected to correct pin
- Check common ground between Arduino and servo PSU
- Ensure external 5V power supply is on

**Servo Jittering:**
- Add decoupling capacitor (1000µF) across servo power supply
- Check for loose connections
- Verify power supply can handle current draw

## Safety Notes

⚠️ **Before Testing:**
- Ensure all wiring matches the circuit diagram
- Verify external 5V power supply is properly connected
- Check that servos have enough clearance to move
- Keep hands clear of moving parts during tests

⚠️ **During Testing:**
- Watch for any unusual behavior
- Stop immediately if servos stall or make unusual noises
- Monitor power supply voltage if possible

## Next Steps

After successful testing:
1. Upload the main firmware: `arduino/tomato_arm/tomato_arm.ino`
2. Test communication with Raspberry Pi
3. Begin calibration and tuning

## Support

If components fail tests:
1. Check `WIRING_CHECKLIST.md` for connection verification
2. Review `CIRCUIT_DIAGRAM.md` for correct wiring
3. Test components individually using `SERVO` commands
4. Verify power supply ratings and connections

