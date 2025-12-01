# HTS-16L Servo Compatibility

## Overview

The **HTS-16L** is a **serial bus servo** (UART-based), not a standard PWM servo. It requires different control methods than the current firmware supports.

## Key Differences

| Feature | Standard PWM Servos (SG90/MG996R) | HTS-16L Serial Servo |
|---------|----------------------------------|----------------------|
| **Control Method** | PWM (Pulse Width Modulation) | UART Serial (115200 baud) |
| **Library** | Arduino `Servo` library | Hiwonder serial servo library |
| **Wiring** | PWM pin (D2-D7) | UART pins (TX/RX) |
| **Rotation Range** | 0-180° | 0-240° |
| **Torque** | 1.8-10 kg·cm | 16 kg·cm @ 7.4V |
| **Voltage** | 4.8-6V | 5-8.4V |
| **Communication** | Analog signal | Digital serial commands |

## Current System

The firmware currently uses:
- **Arduino Servo library** for PWM control
- **PWM pins** (D2-D7) for servo signals
- **Standard pulse widths** (500-2400 microseconds)

## To Use HTS-16L

### Required Changes:

1. **Hardware Wiring**:
   - Connect HTS-16L to Arduino UART pins (TX/RX)
   - Requires UART connection, not PWM pin
   - May need level shifter if voltage mismatch

2. **Software Library**:
   - Install Hiwonder serial servo library
   - Cannot use Arduino `Servo` library

3. **Firmware Modifications**:
   - Rewrite servo control code for serial communication
   - Add serial servo driver class
   - Modify `ServoManager` to support both PWM and serial servos

4. **Power Supply**:
   - HTS-16L needs 5-8.4V (higher than standard servos)
   - Ensure power supply can handle 16kg·cm torque requirements

## Recommendation

### Option 1: Use Standard PWM Servos (Recommended)
- ✅ **No code changes needed**
- ✅ **Plug and play**
- ✅ **Well tested**
- ✅ **Lower cost**
- ✅ **Simpler wiring**

**For claw**: Use SG90 or MG90S (sufficient for gripping tomatoes)

### Option 2: Add HTS-16L Support
- ⚠️ Requires firmware modifications
- ⚠️ More complex wiring
- ⚠️ Higher cost
- ✅ Higher torque (16kg·cm)
- ✅ Better for heavy-duty applications

**Best for**: If you need very high torque or already have HTS-16L servos

## If You Want to Proceed with HTS-16L

I can modify the firmware to:
1. Add serial servo driver support
2. Support mixed setup (PWM servos + HTS-16L)
3. Configure HTS-16L for claw or other joints

**Note**: This requires significant firmware changes and testing.

## Alternative: High-Torque PWM Servos

If you need more torque than SG90, consider:
- **MG996R** (10 kg·cm) - Already supported
- **DS3225** (25 kg·cm) - PWM compatible
- **DS3235** (35 kg·cm) - PWM compatible

These work with the current firmware without modifications!

