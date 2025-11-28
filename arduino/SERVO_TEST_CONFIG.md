# Servo Test Configuration and Neutral Positions

This document tracks the tested servos, their neutral positions, and stop methods.

## Base Servo - Pin 7

**Status:** ✅ Tested and Working

**Hardware:** Tower Pro MG996R

**Test File:** `test_base_pin2.ino`

**Neutral/Stop Position:** 90 degrees

**Stop Method:**
1. Send 90 degrees (neutral position)
2. Detach servo: `baseServo.detach()`
3. Set pin to INPUT: `pinMode(BASE_PIN, INPUT)`

**Rotation Range:** 0° → 180° → 0° (360° total path)

**Code Pattern:**
```cpp
// Stop sequence
baseServo.write(90);  // Neutral/stop position
delay(1000);
baseServo.detach();
pinMode(BASE_PIN, INPUT);
```

---

## Forearm/Shoulder Servo - Pin 6

**Status:** ✅ Tested and Working

**Hardware:** Tower Pro MG996R

**Test File:** `test_forearm_pin6.ino`

**Reference Position:** 90 degrees (set on power-on, NO assumptions about arm position)

**Stop Method:**
1. Send 90 degrees (reference position)
2. Detach servo: `forearmServo.detach()`
3. Set pin to INPUT: `pinMode(FOREARM_PIN, INPUT)`

**Rotation Range:** 0° - 180° (MAX 180° - NEVER EXCEED)

**Movement Logic:**
- All movements are **relative to reference position (90°)**
- Moving LEFT (negative): `90° - offset` (minimum 0°)
- Moving RIGHT (positive): `90° + offset` (maximum 180°)
- **No assumptions** about arm position before power-on

**Power-On Sequence:**
1. Set pin to OUTPUT
2. Attach servo
3. **IMMEDIATELY** write 90° (no delay before this!)
4. This sets the reference position

**Safety Limits:**
- MIN: 0 degrees
- MAX: 180 degrees (hard limit)
- All angles constrained with `constrain(angle, MIN_ANGLE, MAX_ANGLE)`

**Code Pattern:**
```cpp
const int REFERENCE_ANGLE = 90;  // Set on power-on

// Power-on sequence
pinMode(FOREARM_PIN, OUTPUT);
forearmServo.attach(FOREARM_PIN);
forearmServo.write(90);  // IMMEDIATELY - no delay before this!

// Movement relative to reference position
int angle = REFERENCE_ANGLE + offset;  // Right (+)
int angle = REFERENCE_ANGLE - offset;  // Left (-)
int safeAngle = constrain(angle, 0, 180);

// Stop sequence
forearmServo.write(90);  // Reference/stop position
delay(1000);
forearmServo.detach();
pinMode(FOREARM_PIN, INPUT);
```

---

## Claw/Gripper Servo - Pin 2

**Status:** ✅ Tested and Working

**Hardware:** SG90 Micro Servo (9g)

**Test File:** `test_claw_pin2.ino`

**Reference Position:** 90 degrees (closed position, set on power-on)

**Stop Method:**
1. Send 90 degrees (closed/safe position)
2. Detach servo: `clawServo.detach()`
3. Set pin to INPUT: `pinMode(CLAW_PIN, INPUT)`

**Rotation Range:** 0° - 90° (MAX 90° - NEVER EXCEED)

**Claw Behavior:**
- **CLOSED Position:** 90 degrees
- **OPEN Position:** 0 degrees
- **Movement:** Only moves between 0° (open) and 90° (closed)
- **No assumptions** about claw position before power-on

**Power-On Sequence:**
1. Set pin to OUTPUT
2. Attach servo
3. **IMMEDIATELY** write 90° (closed position, no delay before this!)
4. This sets the safe/closed position

**Safety Limits:**
- MIN: 0 degrees (open position)
- MAX: 90 degrees (closed position, hard limit)
- All angles constrained with `constrain(angle, MIN_ANGLE, MAX_ANGLE)`

**Code Pattern:**
```cpp
const int CLAW_CLOSED = 90;  // Closed position
const int CLAW_OPEN = 0;     // Open position
const int MAX_ANGLE = 90;    // Never exceed

// Power-on sequence
pinMode(CLAW_PIN, OUTPUT);
clawServo.attach(CLAW_PIN);
clawServo.write(90);  // IMMEDIATELY - no delay before this!

// Opening: 90° → 0°
for (int angle = CLAW_CLOSED; angle >= CLAW_OPEN; angle--) {
  int safeAngle = constrain(angle, 0, MAX_ANGLE);
  clawServo.write(safeAngle);
  delay(15);
}

// Closing: 0° → 90°
for (int angle = CLAW_OPEN; angle <= CLAW_CLOSED; angle++) {
  int safeAngle = constrain(angle, 0, MAX_ANGLE);
  clawServo.write(safeAngle);
  delay(15);
}

// Stop sequence
clawServo.write(90);  // Closed/stop position
delay(1000);
clawServo.detach();
pinMode(CLAW_PIN, INPUT);
```

---

## Key Learnings

1. **Neutral Position:** 90 degrees works as the stop position for MG996R servos
2. **Claw Position:** 90° = closed, 0° = open (opposite of typical servo logic)
3. **Stop Sequence:** Detaching and setting pin to INPUT prevents continuous rotation
4. **Safety:** Always use `constrain()` to prevent exceeding limits
5. **Testing:** Each servo should be tested individually before integration

---

## Next Steps

- [x] Test forearm servo (pin 6) and verify it stops properly
- [x] Test claw servo (pin 2) and verify it stops properly
- [ ] Document any additional servos as they are tested
- [ ] Update main firmware with verified neutral positions

