# Wiring Checklist - AI Tomato Sorter

Use this checklist to verify all connections match the circuit diagram before powering on.

## ⚠️ Safety First
- [ ] **DO NOT** power servos from Arduino 5V pin
- [ ] **DO NOT** connect power until all wiring is verified
- [ ] Ensure all connections are secure and not loose
- [ ] Check for short circuits before applying power

---

## Power Connections

### External 5V 5A Power Supply (for Servos)
- [ ] Power supply rated for **5V 5A minimum**
- [ ] Power supply **GND (Negative)** → All 6 servo **Black/Brown wires**
- [ ] Power supply **5V (Positive)** → All 6 servo **Red wires**
- [ ] Power supply **GND** → Arduino **GND pin** (common ground)
- [ ] Power supply **GND** → Raspberry Pi **GND** (common ground)

### Raspberry Pi Power
- [ ] Official Raspberry Pi power supply connected to **USB-C port**
- [ ] Power supply rated for **5V 3A (Pi 4)** or **5V 5A (Pi 5)**

### Arduino Power
- [ ] Arduino powered via **USB cable** from Raspberry Pi
- [ ] USB cable provides both **data and power**

---

## Servo Connections (6 servos)

All servos receive:
- **Power (5V)**: From external power supply (Red wire)
- **Ground (GND)**: From external power supply (Black/Brown wire)
- **Signal (PWM)**: From Arduino (Yellow/White wire)

### Base Servo
- [ ] Signal wire → Arduino **Digital Pin 3 (D3)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

### Shoulder Servo
- [ ] Signal wire → Arduino **Digital Pin 5 (D5)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

### Elbow Servo
- [ ] Signal wire → Arduino **Digital Pin 6 (D6)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

### Wrist Vertical Servo
- [ ] Signal wire → Arduino **Digital Pin 9 (D9)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

### Wrist Rotational Servo
- [ ] Signal wire → Arduino **Digital Pin 10 (D10)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

### Gripper Servo
- [ ] Signal wire → Arduino **Digital Pin 11 (D11)**
- [ ] Red wire → External PSU **5V**
- [ ] Black/Brown wire → External PSU **GND**

---

## VL53L0X Distance Sensor Connections

- [ ] **VCC** → Arduino **3.3V pin**
- [ ] **GND** → Arduino **GND pin**
- [ ] **SDA** → Arduino **A4 (or SDA) pin**
- [ ] **SCL** → Arduino **A5 (or SCL) pin**

> **Note**: Most VL53L0X modules work on 3.3V. If your module has an onboard regulator, 5V might work, but 3.3V is safer.

---

## Data/Communication Connections

### Raspberry Pi ↔ Arduino
- [ ] USB cable connected from Raspberry Pi **USB port** to Arduino **USB port**
- [ ] Cable provides both **data (serial)** and **power**

### Raspberry Pi ↔ USB Camera
- [ ] USB camera connected to Raspberry Pi **USB port**
- [ ] Camera is recognized by system (check with `lsusb`)

---

## Ground Connections (Critical!)

All components must share a common ground:
- [ ] External PSU GND → Arduino GND
- [ ] External PSU GND → Raspberry Pi GND
- [ ] Arduino GND → VL53L0X GND

> **Important**: Without common ground, PWM signals won't work correctly and sensors may give incorrect readings.

---

## Verification Steps

### Before Powering On:
1. [ ] All connections double-checked against circuit diagram
2. [ ] No loose wires or exposed connections
3. [ ] No short circuits (check with multimeter if possible)
4. [ ] Servo wires have enough slack for full arm movement

### After Powering On:
1. [ ] Arduino LED indicates power (usually ON)
2. [ ] Raspberry Pi boots successfully
3. [ ] Camera detected (check with `lsusb` or `v4l2-ctl --list-devices`)
4. [ ] Arduino serial communication works (test with `screen /dev/ttyUSB0 115200`)
5. [ ] Servos respond to commands (test with HOME command)
6. [ ] VL53L0X sensor responds (test with DISTANCE command)

---

## Pin Reference Table

| Component | Arduino Pin | Type | Description |
|-----------|-------------|------|-------------|
| Base Servo | D3 | PWM | Base rotation |
| Shoulder Servo | D5 | PWM | Shoulder joint |
| Elbow Servo | D6 | PWM | Elbow joint |
| Wrist Vert Servo | D9 | PWM | Wrist vertical |
| Wrist Rot Servo | D10 | PWM | Wrist rotation |
| Gripper Servo | D11 | PWM | Gripper open/close |
| VL53L0X SDA | A4 | I2C | I2C Data line |
| VL53L0X SCL | A5 | I2C | I2C Clock line |
| VL53L0X VCC | 3.3V | Power | Sensor power |
| VL53L0X GND | GND | Ground | Sensor ground |

---

## Troubleshooting

### Servos Not Moving
- [ ] Check servo power connections (5V and GND from external PSU)
- [ ] Verify common ground between Arduino and servo PSU
- [ ] Check PWM signal wires are connected to correct pins
- [ ] Verify Arduino is receiving commands (check serial monitor)

### VL53L0X Not Working
- [ ] Check I2C connections (SDA/SCL)
- [ ] Verify 3.3V power (not 5V unless module supports it)
- [ ] Check ground connection
- [ ] Test with `arduino/test_tof/test_tof.ino` sketch first

### Camera Not Detected
- [ ] Check USB connection
- [ ] Try different USB port
- [ ] Verify camera works on another computer
- [ ] Check USB permissions: `sudo usermod -a -G video $USER`

### Arduino Not Communicating
- [ ] Check USB cable (data-capable, not charge-only)
- [ ] Verify baud rate (115200)
- [ ] Check serial port: `ls /dev/ttyUSB* /dev/ttyACM*`
- [ ] Verify user has permissions: `sudo usermod -a -G dialout $USER`

---

## Quick Test Commands

### Test Arduino Connection:
```bash
screen /dev/ttyUSB0 115200
# Then type: STATUS
# Should see: STATUS: READY | VL53L0X: OK
```

### Test Distance Sensor:
```bash
screen /dev/ttyUSB0 115200
# Then type: DISTANCE
# Should see: DISTANCE: <number> (in mm)
```

### Test Servos:
```bash
screen /dev/ttyUSB0 115200
# Then type: HOME
# Should see: OK: HOME
# Arm should move to home position
```

---

## Notes

- Keep servo power wires (5V/GND) separate from signal wires to reduce interference
- Use proper wire gauge (≥18AWG) for servo power connections
- Ensure servo cables are long enough for full arm movement
- Label all connections for easy troubleshooting
- Document any deviations from this checklist

---

**Last Updated**: Based on circuit diagram with Arduino R4 WiFi/Minima, 6 servos, and VL53L0X sensor.

