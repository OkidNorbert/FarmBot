# HTS-16L Servo Setup Guide

## Overview

The firmware now supports **HTS-16L serial servo** for the shoulder joint. This provides high torque (16 kg·cm) and precise position control.

## Hardware Wiring

### HTS-16L Connection to Arduino UNO R4 WiFi

1. **Power** (5-8.4V):
   - **Red wire** → External 5-8.4V power supply (positive)
   - **Black wire** → Common ground (connect to Arduino GND and power supply GND)
   - ⚠️ **DO NOT** power from Arduino 5V pin (needs more current)

2. **Serial Communication**:
   - **Yellow/White wire (Data)** → Arduino Serial1 RX pin
   - **Note**: Check your HTS-16L wiring - some models use different colors
   - Default: Connect to **Serial1** (pins vary by board)

### Arduino UNO R4 WiFi Serial Ports

- **Serial** (USB): Used for debugging/monitoring
- **Serial1**: Hardware UART (check pinout for RX/TX pins)
- **Serial2**: Additional hardware UART (if available)

**Important**: You may need to adjust which serial port to use in `servo_manager.cpp`:
```cpp
hts16l_serial = &Serial1; // Change to Serial2 or Serial if needed
```

## Configuration

In `config.h`:

```cpp
#define SHOULDER_USE_HTS16L  true  // Enable HTS-16L for shoulder
#define HTS16L_SERVO_ID     1      // Servo ID (default is 1)
#define HTS16L_BAUD_RATE    115200 // Communication speed
```

## Servo ID Setup

The HTS-16L servo has a default ID of 1. If you need to change it:
1. Use Hiwonder configuration software
2. Or modify `HTS16L_SERVO_ID` in `config.h`

## Testing

1. **Upload firmware** to Arduino
2. **Open Serial Monitor** (115200 baud)
3. You should see: `"HTS-16L shoulder servo initialized"`
4. **Test movement** via web interface or commands:
   - Send shoulder to 60°: Should move smoothly
   - Send shoulder to 120°: Should move smoothly
   - Send shoulder to 90° (home): Should return

## Troubleshooting

### Servo Doesn't Move
- ✅ Check power supply (5-8.4V, sufficient current)
- ✅ Verify serial connection (RX pin)
- ✅ Check servo ID matches configuration
- ✅ Verify baud rate is 115200
- ✅ Check Serial Monitor for initialization message

### Servo Moves Erratically
- ✅ Check power supply stability
- ✅ Verify ground connections (common ground required)
- ✅ Check for loose connections

### Serial Port Issues
- ✅ Try different serial port (Serial1, Serial2)
- ✅ Check pinout for your specific Arduino model
- ✅ Verify no conflicts with other serial devices

## Advantages of HTS-16L

- ✅ **High Torque**: 16 kg·cm (much stronger than MG996R)
- ✅ **Precise Control**: 0-240° range (vs 0-180° for standard servos)
- ✅ **Smooth Movement**: Built-in speed control
- ✅ **Reliable**: Serial communication (digital, not analog PWM)

## Switching Back to PWM Servo

If you want to use a standard PWM servo instead:

```cpp
#define SHOULDER_USE_HTS16L  false
```

Then reconnect shoulder to PWM pin (D6) and re-upload firmware.

