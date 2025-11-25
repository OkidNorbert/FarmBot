# Servo Troubleshooting Guide

## Problem: Only One Servo Moving

If only one servo is moving, this is almost always a **power or wiring issue**.

### Quick Diagnostic

1. **Run the diagnostic test**:
   ```bash
   # Upload diagnose_servos.ino
   arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:renesas_uno:unor4wifi arduino/test_all_components/diagnose_servos.ino
   ```

2. **Watch which servos move** and report back

### Most Common Causes

#### 1. External Power Supply Not Connected ⚠️ **MOST COMMON**

**Symptoms:**
- Only one servo moves (usually the first one tested)
- Servos jitter or don't move at all
- Arduino resets when servos try to move

**Solution:**
- ✅ Connect **external 5V 5A power supply** to servos
- ✅ Connect **all servo red wires** to PSU **5V (positive)**
- ✅ Connect **all servo black/brown wires** to PSU **GND (negative)**
- ✅ **DO NOT** power servos from Arduino 5V pin

**Check:**
- Is your external power supply turned ON?
- Is it rated for at least 5V 5A?
- Are all servo power wires connected?

#### 2. Missing Common Ground ⚠️ **CRITICAL**

**Symptoms:**
- Servos don't respond to commands
- Servos jitter randomly
- Only some servos work

**Solution:**
- ✅ Connect **external PSU GND** to **Arduino GND pin**
- ✅ This is REQUIRED for PWM signals to work
- ✅ Without common ground, servos can't read the signal

**Check:**
- Is there a wire from PSU GND to Arduino GND?
- Is the connection secure?

#### 3. Power Distribution Issue

**Symptoms:**
- Some servos work, others don't
- Servos work individually but not together

**Solution:**
- ✅ Use a **power distribution board** or **terminal block**
- ✅ Connect PSU 5V to distribution rail
- ✅ Connect all servo red wires to distribution rail
- ✅ Connect PSU GND to distribution rail
- ✅ Connect all servo black wires to distribution rail

**Check:**
- Are all servos getting power from the same source?
- Are connections secure (not loose)?

#### 4. Signal Wire Issues

**Symptoms:**
- Servo doesn't move but power LED is on
- Servo jitters but doesn't move properly

**Solution:**
- ✅ Check **signal wire** (yellow/white) is connected to correct Arduino pin
- ✅ Verify pin numbers match:
  - Base: D3
  - Shoulder: D5
  - Elbow: D6
  - Wrist Vert: D9
  - Wrist Rot: D10
  - Gripper: D11

**Check:**
- Are signal wires connected to correct pins?
- Are connections secure?

#### 5. Insufficient Power Supply

**Symptoms:**
- Servos work individually but not together
- Power supply gets hot
- Voltage drops when servos move

**Solution:**
- ✅ Use **5V 5A minimum** (10A recommended for safety)
- ✅ Check power supply can handle all 6 servos
- ✅ Add **decoupling capacitor** (1000-2200µF) across power rails

**Check:**
- Is your power supply rated for enough current?
- Does it have enough capacity for 6 servos?

### Step-by-Step Diagnostic

1. **Test Power Supply**:
   - Measure voltage at servo power rail (should be ~5V)
   - Check with multimeter if possible

2. **Test Each Servo Individually**:
   - Upload `diagnose_servos.ino`
   - Watch which servos move
   - Note which ones don't work

3. **Check Wiring**:
   - Verify each servo has:
     - Red wire → PSU 5V
     - Black wire → PSU GND
     - Signal wire → Correct Arduino pin

4. **Check Common Ground**:
   - Verify PSU GND → Arduino GND connection
   - This is critical for signal communication

5. **Test Signal Wires**:
   - Try swapping signal wires between working and non-working servos
   - If servo works with different signal wire, original wire may be bad

### Wiring Verification Checklist

For each servo, verify:

- [ ] **Red wire** → External PSU **5V** (NOT Arduino 5V)
- [ ] **Black/Brown wire** → External PSU **GND**
- [ ] **Yellow/White wire** → Correct Arduino pin (D3, D5, D6, D9, D10, or D11)
- [ ] **Common ground** → PSU GND connected to Arduino GND
- [ ] **Power supply** → Turned ON and rated for 5V 5A+

### Quick Test Commands

After uploading the main test sketch, try these commands in Serial Monitor:

```
SERVO 1 90    # Test Base (D3)
SERVO 2 90    # Test Shoulder (D5)
SERVO 3 90    # Test Elbow (D6)
SERVO 4 90    # Test Wrist Vert (D9)
SERVO 5 90    # Test Wrist Rot (D10)
SERVO 6 90    # Test Gripper (D11)
```

Watch which servos respond!

### Still Not Working?

1. **Check Serial Monitor**:
   - Does it show "ATTACHED" for all servos?
   - Are there any error messages?

2. **Test with Multimeter**:
   - Measure voltage at servo power rail
   - Check continuity of signal wires
   - Verify ground connections

3. **Test Servos Individually**:
   - Disconnect all servos
   - Connect one servo at a time
   - Test each one individually

4. **Check Power Supply**:
   - Is it actually outputting 5V?
   - Can it supply enough current?
   - Try a different power supply if available

### Expected Behavior

When everything is wired correctly:
- ✅ All 6 servos should move smoothly
- ✅ No jittering or random movements
- ✅ Arduino should not reset
- ✅ Power supply should stay cool
- ✅ Serial monitor should show "ATTACHED" for all servos

### Safety Reminders

⚠️ **Before Testing:**
- Double-check all wiring
- Verify power supply voltage
- Ensure no short circuits
- Keep hands clear of moving parts

⚠️ **If Servos Don't Move:**
- Check power supply is ON
- Verify external PSU is connected
- Check common ground connection
- Verify signal wires are connected

