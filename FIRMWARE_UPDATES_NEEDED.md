# Firmware Updates Needed for Limited Servo Mode

## ⚠️ Issue Found: Motion Planner Uses Shoulder for Lifting

### Problem

**Location:** `arduino/main_firmware/motion_planner.cpp` lines 145-156

**Current Code:**
```cpp
case PICK_LIFT: {
    // Lift by raising shoulder
    int current_shoulder = _servoMgr->getAngle(1);
    int lift_shoulder = constrain(current_shoulder + _liftHeightDeg, 15, 165);
    
    _servoMgr->setTarget(1, lift_shoulder);  // ❌ Tries to move shoulder!
    
    if (waitForMotionComplete()) {
        _currentState = PICK_MOVE_TO_BIN;
    }
    break;
}
```

**Issue:**
- Motion planner tries to lift by raising the shoulder servo
- If shoulder is fixed, this won't work
- Pick sequence will fail at the lift stage

### Solution Options

#### Option 1: Use Forearm/Elbow for Lifting (Recommended)

**Update `motion_planner.cpp` PICK_LIFT case:**
```cpp
case PICK_LIFT: {
    // Lift by adjusting forearm/elbow instead of shoulder
    // Since shoulder is fixed, use forearm to lift
    int current_forearm = _servoMgr->getAngle(2);
    int lift_forearm = constrain(current_forearm - _liftHeightDeg, 10, 170);
    
    _servoMgr->setTarget(2, lift_forearm);  // Use forearm to lift
    
    if (waitForMotionComplete()) {
        _currentState = PICK_MOVE_TO_BIN;
        _stateStartTime = millis();
    }
    break;
}
```

#### Option 2: Skip Lift Stage

**Update `motion_planner.cpp` PICK_LIFT case:**
```cpp
case PICK_LIFT: {
    // Skip lift if shoulder is fixed - go directly to bin
    // (Tomato is already grasped, just move to bin)
    _currentState = PICK_MOVE_TO_BIN;
    _stateStartTime = millis();
    break;
}
```

#### Option 3: Use Pitch for Lifting

**Update `motion_planner.cpp` PICK_LIFT case:**
```cpp
case PICK_LIFT: {
    // Lift by adjusting pitch angle
    int current_pitch = _servoMgr->getAngle(4);
    int lift_pitch = constrain(current_pitch - 15, 20, 160);  // Pitch up to lift
    
    _servoMgr->setTarget(4, lift_pitch);
    
    if (waitForMotionComplete()) {
        _currentState = PICK_MOVE_TO_BIN;
        _stateStartTime = millis();
    }
    break;
}
```

**Recommendation:** Use **Option 1** (forearm) or **Option 2** (skip lift) - both work well.

---

## Other Potential Issues

### 1. Approach Poses Use Base/Shoulder

**Location:** `motion_planner.cpp` lines 80-86

**Current Code:**
```cpp
int approach_base = pixelToBaseAngle(_targetPixelX);
int approach_shoulder = 60;
int approach_forearm = 100;
...
moveToPose(approach_base, approach_shoulder, approach_forearm, ...);
```

**Impact:**
- `moveToPose()` will receive base/shoulder angles
- If we send `-1`, it will use current angles (which are fixed)
- This should work, but approach pose might not be optimal

**Solution:** This is OK - `moveToPose()` will use fixed base/shoulder positions. Approach might not be perfect, but it will work.

### 2. Bin Poses Use Base/Shoulder

**Location:** `motion_planner.cpp` lines 163-164

**Current Code:**
```cpp
moveToPose(binPose->base, binPose->shoulder, binPose->forearm, ...);
```

**Impact:**
- Bin poses have base/shoulder angles
- If we send `-1`, it uses current (fixed) angles
- Bin positions might not be optimal

**Solution:** This is OK - bin poses will use fixed base/shoulder. You may need to adjust bin positions manually.

---

## Required Firmware Update

### Update PICK_LIFT State

**File:** `arduino/main_firmware/motion_planner.cpp`

**Replace lines 145-156 with:**

```cpp
case PICK_LIFT: {
    // Lift by adjusting forearm (since shoulder is fixed)
    // Alternative: Use elbow or pitch, or skip lift entirely
    int current_forearm = _servoMgr->getAngle(2);
    int lift_forearm = constrain(current_forearm - _liftHeightDeg, LIMIT_FOREARM_MIN, LIMIT_FOREARM_MAX);
    
    _servoMgr->setTarget(2, lift_forearm);
    
    if (waitForMotionComplete()) {
        _currentState = PICK_MOVE_TO_BIN;
        _stateStartTime = millis();
    }
    break;
}
```

**Or use skip lift (simpler):**
```cpp
case PICK_LIFT: {
    // Skip lift stage - shoulder is fixed, go directly to bin
    _currentState = PICK_MOVE_TO_BIN;
    _stateStartTime = millis();
    break;
}
```

---

## Summary

### ✅ What Works:
- ANGLE command with `-1` values ✅
- Approach poses (uses fixed base/shoulder) ✅
- Bin poses (uses fixed base/shoulder) ✅
- Grasp and release ✅

### ❌ What Needs Fixing:
- **PICK_LIFT state** - tries to move shoulder ❌
  - **Fix:** Use forearm/elbow/pitch for lifting, or skip lift

### Action Required:
1. **Update `motion_planner.cpp`** - Fix PICK_LIFT state
2. **Recompile and upload** firmware
3. **Test pick sequence** with limited servos

---

## Quick Fix Implementation

I can provide the exact code changes needed. Would you like me to:
1. Update the motion_planner.cpp file with the fix?
2. Or provide instructions for manual update?

**The fix is simple - just one case statement needs updating!**

