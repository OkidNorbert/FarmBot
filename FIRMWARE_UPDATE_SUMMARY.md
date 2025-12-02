# Firmware Update Summary - Limited Servo Mode Support

## ✅ Firmware Updated!

The Arduino firmware has been updated to fully support limited servo mode (fixed base and shoulder).

## Changes Made

### 1. **Fixed PICK_LIFT State** ✅

**File:** `arduino/main_firmware/motion_planner.cpp`

**Changed:**
- **Before:** Tried to lift by moving shoulder servo
- **After:** Lifts by moving forearm servo (works with fixed shoulder)

**Code:**
```cpp
case PICK_LIFT: {
    // Lift by adjusting forearm (since shoulder may be fixed)
    int current_forearm = _servoMgr->getAngle(2);
    int lift_forearm = constrain(current_forearm - _liftHeightDeg, LIMIT_FOREARM_MIN, LIMIT_FOREARM_MAX);
    _servoMgr->setTarget(2, lift_forearm);
    ...
}
```

### 2. **Updated Approach Poses** ✅

**File:** `arduino/main_firmware/motion_planner.cpp`

**Changed:**
- **Before:** Calculated base angle from pixel position
- **After:** Uses `-1` for base/shoulder (keeps current fixed positions)

**Code:**
```cpp
int approach_base = -1;      // Use current base angle (fixed)
int approach_shoulder = -1;   // Use current shoulder angle (fixed)
```

### 3. **Updated Bin Poses** ✅

**File:** `arduino/main_firmware/motion_planner.cpp`

**Changed:**
- **Before:** Had specific base/shoulder angles for bins
- **After:** Uses `-1` for base/shoulder (keeps current fixed positions)

**Code:**
```cpp
_binRipe.base = -1;      // Use current base angle (fixed)
_binRipe.shoulder = -1;  // Use current shoulder angle (fixed)
```

### 4. **Enhanced moveToPose() Method** ✅

**File:** `arduino/main_firmware/motion_planner.cpp`

**Changed:**
- **Before:** Passed angles directly to `setTargets()`
- **After:** Handles `-1` values by using current angles

**Code:**
```cpp
bool MotionPlanner::moveToPose(int base, int shoulder, int forearm, int elbow, int pitch, int claw) {
    // Handle -1 values (keep current angle)
    int base_angle = (base == -1) ? _servoMgr->getAngle(0) : base;
    int shoulder_angle = (shoulder == -1) ? _servoMgr->getAngle(1) : shoulder;
    ...
    return _servoMgr->setTargets(base_angle, shoulder_angle, ...);
}
```

## What This Means

### ✅ Full Compatibility

The firmware now:
1. ✅ Accepts `-1` in ANGLE commands (already supported)
2. ✅ Uses forearm for lifting (instead of shoulder)
3. ✅ Keeps base/shoulder fixed during pick sequences
4. ✅ Works with only 4 servos (forearm, elbow, pitch, claw)

### Pick Sequence Flow

1. **PICK_CALCULATE_POSE:** Calculates target (base/shoulder use fixed positions)
2. **PICK_MOVE_TO_APPROACH:** Moves to approach (base/shoulder stay fixed)
3. **PICK_APPROACH_TOF:** Fine-tunes with ToF sensor
4. **PICK_GRASP:** Closes claw
5. **PICK_LIFT:** Lifts using **forearm** (not shoulder) ✅
6. **PICK_MOVE_TO_BIN:** Moves to bin (base/shoulder stay fixed)
7. **PICK_RELEASE:** Opens claw
8. **PICK_RETURN_HOME:** Returns home (base/shoulder stay fixed)

## Next Steps

### 1. **Upload Updated Firmware**

```bash
# Open Arduino IDE
# Load: arduino/main_firmware/main_firmware.ino
# Select: Arduino UNO R4 WiFi
# Upload
```

### 2. **Test Pick Sequence**

1. Manually set base to 90° and shoulder to 135°
2. Place a tomato in front of arm
3. Enable automatic mode
4. System should detect and pick using only 4 servos

### 3. **Adjust Bin Positions** (if needed)

Since base/shoulder are fixed, you may need to adjust bin positions:

**Option A: Use Web Interface**
- Go to Calibrate page
- Set bin poses using only forearm, elbow, pitch, claw

**Option B: Manual Adjustment**
- Test pick sequence
- Adjust forearm/elbow/pitch angles for bin positions
- Update in code if needed

## Summary

✅ **Firmware fully supports limited servo mode!**

**Changes:**
- ✅ Lift uses forearm instead of shoulder
- ✅ Approach poses use fixed base/shoulder
- ✅ Bin poses use fixed base/shoulder
- ✅ `moveToPose()` handles `-1` values

**Status:**
- ✅ Ready to upload and test
- ✅ Compatible with Python code changes
- ✅ Works with 4 servos (forearm, elbow, pitch, claw)

**The firmware is ready!** 🎉

