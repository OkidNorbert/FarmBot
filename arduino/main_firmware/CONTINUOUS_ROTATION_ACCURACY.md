# Continuous Rotation Servo Accuracy Analysis

## Overview

This document analyzes the accuracy characteristics of the continuous rotation servo implementation for the base (pin D7). Understanding these limitations is crucial for reliable operation.

## Current Accuracy Characteristics

### 1. **Position Holding Accuracy**

**Theoretical Accuracy:**
- **Stop Tolerance**: ±2 degrees (hardcoded in `updateContinuousRotation()`)
- **Physical Reality**: Continuous rotation servos **do not hold position** - they only stop rotating
- **No Position Lock**: Once stopped, the base can be moved manually or by external forces

**Actual Behavior:**
- When target is reached (within 2°), servo command = 90° (stop signal)
- Servo stops rotating but has **no mechanical lock**
- Position can drift if:
  - External forces are applied
  - Vibration occurs
  - Servo has mechanical play/backlash
  - Power fluctuations affect servo behavior

**Expected Accuracy:**
- **Short-term (seconds)**: ±2-5° (if no external forces)
- **Long-term (minutes)**: ±5-15° (due to drift, vibration, mechanical play)
- **After power cycle**: Position is lost - must re-home

### 2. **Angle Detection/Targeting Accuracy**

**Current Implementation:**
- Uses **time-based virtual position tracking**
- Calculates position from: `elapsed_time × rotation_speed`
- No actual position feedback

**Accuracy Factors:**

#### A. **Calibration Accuracy** (BASE_ROTATION_SPEED)
- **Impact**: High - if calibration is off by 10%, position error = 10%
- **Typical Error**: ±5-15% if not properly calibrated
- **Example**: If actual speed is 28 deg/s but calibrated as 30 deg/s:
  - After 90° rotation: Actual = 84°, Virtual = 90° → **6° error**

#### B. **Speed Variation**
- **Load-dependent**: Speed changes with arm position/load
- **Voltage-dependent**: Lower voltage = slower rotation
- **Temperature-dependent**: Motor resistance changes
- **Typical Variation**: ±5-10% under normal conditions

#### C. **Stop Accuracy (Coasting)**
- **Problem**: Servo doesn't stop instantly when command = 90°
- **Coasting Distance**: Typically 2-10° depending on speed
- **Current Code**: Doesn't account for coasting - stops at virtual position, not physical
- **Error**: ±2-10° depending on rotation speed

#### D. **Timer Resolution**
- **millis() Resolution**: ~1ms (on most Arduino boards)
- **Impact**: Low for slow rotations, higher for fast
- **Example**: At 30 deg/s, 1ms = 0.03° error (negligible)

#### E. **Cumulative Error**
- **Accumulation**: Errors add up over multiple movements
- **Example**: 5% error per 90° rotation → after 4 rotations (360°), error = 18°
- **No Reset Mechanism**: Position can drift indefinitely

### 3. **Real-World Accuracy Estimates**

**Best Case Scenario** (well-calibrated, no load, stable voltage):
- **Single Movement**: ±3-5° accuracy
- **Position Hold**: ±2-5° (short-term)
- **Repeatability**: ±5-10° (same target, multiple attempts)

**Typical Scenario** (moderate calibration, normal load):
- **Single Movement**: ±5-10° accuracy
- **Position Hold**: ±5-15° (short-term)
- **Repeatability**: ±10-20° (same target, multiple attempts)

**Worst Case Scenario** (poor calibration, heavy load, voltage fluctuations):
- **Single Movement**: ±10-20° accuracy
- **Position Hold**: ±15-30° (short-term)
- **Repeatability**: ±20-40° (same target, multiple attempts)

## Code Analysis

### Current Tolerance Setting
```cpp
if (abs(diff) < 2) { // 2 degree tolerance
    // Stop rotation
}
```

**Issue**: This is the **virtual position** tolerance, not physical position.

**Reality**: 
- Virtual position may be within 2° of target
- Physical position may be 5-15° off due to:
  - Coasting after stop command
  - Calibration errors
  - Speed variations

### Virtual Position Tracking
```cpp
float degrees_rotated = (actual_rotation_speed * elapsed) / 1000.0;
_base_virtual_angle = _base_virtual_angle + degrees_rotated;
```

**Limitations**:
1. Assumes constant rotation speed (not always true)
2. No feedback to verify actual position
3. Errors accumulate over time
4. No compensation for coasting

## Recommendations for Improved Accuracy

### 1. **Add Coasting Compensation**
```cpp
// Estimate coasting distance based on rotation speed
float coasting_distance = actual_rotation_speed * 0.1; // 100ms coast time
// Stop rotation earlier to account for coasting
if (abs(diff) < (2 + coasting_distance)) {
    // Stop rotation
}
```

### 2. **Add Home Position Sensor**
- **Mechanical Switch**: Physical limit switch at home position (90°)
- **Optical Sensor**: IR sensor to detect home mark
- **Magnetic Sensor**: Hall effect sensor for home detection
- **Benefit**: Periodic recalibration to reset drift

### 3. **Improve Calibration Method**
- **Multi-point Calibration**: Measure speed at different angles
- **Load-dependent Calibration**: Calibrate with arm in different positions
- **Voltage Compensation**: Adjust speed based on battery voltage

### 4. **Add Position Feedback** (Best Solution)
- **Rotary Encoder**: Absolute or incremental encoder on base shaft
- **Potentiometer**: Analog position sensor
- **Magnetic Encoder**: Non-contact position sensing
- **Benefit**: Actual position feedback eliminates drift

### 5. **Reduce Speed for Better Accuracy**
- **Slower = More Accurate**: Lower speed = less coasting, more predictable
- **Recommendation**: Use 20-30 deg/s for pick operations (current AUTO_MODE_SPEED = 45)
- **Trade-off**: Slower movement but better accuracy

## Practical Usage Guidelines

### For Pick Operations:
1. **Calibrate BASE_ROTATION_SPEED** carefully with arm in typical position
2. **Use slower speeds** (20-30 deg/s) for better accuracy
3. **Add safety margin** in motion planning (target ±10° tolerance)
4. **Consider home recalibration** after every N picks (e.g., every 10 picks)

### For Manual Control:
1. **Acceptable Accuracy**: ±10-15° is reasonable for manual positioning
2. **Visual Feedback**: Use camera feed to verify position
3. **Fine Adjustment**: Use small angle corrections to fine-tune

### For Automated Operations:
1. **Not Recommended** for high-precision tasks without feedback
2. **Use with ToF Sensor**: ToF sensor can help compensate for base position errors
3. **Add Home Switch**: Essential for long-term operation
4. **Consider Standard Servo**: For better accuracy, use standard 180° servo

## Conclusion

**Current Accuracy:**
- **Short-term targeting**: ±5-10° (typical)
- **Position holding**: ±5-15° (short-term, no external forces)
- **Long-term drift**: Can accumulate to ±20-40° or more

**Suitable For:**
- ✅ Rough positioning (±10-15° acceptable)
- ✅ Manual control with visual feedback
- ✅ Operations where ToF sensor compensates for errors
- ✅ Applications where periodic re-homing is acceptable

**Not Suitable For:**
- ❌ High-precision positioning (<5° required)
- ❌ Long-term autonomous operation without feedback
- ❌ Applications requiring absolute position accuracy
- ❌ Operations where position must be maintained for extended periods

**Recommendation**: For production use, consider adding a home position sensor or upgrading to a standard 180° servo with position feedback for better accuracy.

