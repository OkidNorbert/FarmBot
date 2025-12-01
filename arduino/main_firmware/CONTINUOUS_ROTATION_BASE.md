# Continuous Rotation Servo for Base

## Overview

The firmware now supports using a **360° continuous rotation servo** for the base instead of a standard 180° position servo.

## How It Works

### Key Differences

1. **No Position Feedback**: Continuous rotation servos don't provide position feedback, so the firmware tracks a "virtual position" based on:
   - Rotation direction (CW/CCW)
   - Rotation speed (time-based calculation)
   - Rotation duration

2. **Speed Control**: Instead of position control, continuous rotation servos use:
   - **90°** = Stop
   - **0-89°** = Rotate Counter-Clockwise (speed increases toward 0°)
   - **91-180°** = Rotate Clockwise (speed increases toward 180°)

3. **Virtual Position Tracking**: The firmware maintains a virtual angle (0-180°) that represents where the base "should be" based on rotation calculations.

## Configuration

In `config.h`:

```cpp
#define BASE_CONTINUOUS_ROTATION  true  // Enable continuous rotation mode
#define BASE_ROTATION_SPEED 30          // Degrees per second (calibrate this!)
```

### Calibration

**IMPORTANT**: You need to calibrate `BASE_ROTATION_SPEED` to match your servo's actual rotation speed:

1. Set a target angle (e.g., 90°)
2. Measure how long it takes to rotate 90° from the current position
3. Calculate: `BASE_ROTATION_SPEED = 90 / (time_in_seconds)`
4. Adjust the value in `config.h` and re-upload

**Example**: If it takes 3 seconds to rotate 90°, set `BASE_ROTATION_SPEED = 30`

## Limitations

⚠️ **Important Notes**:

1. **No Absolute Position**: The virtual position can drift over time. The system doesn't know the actual physical position.

2. **Calibration Required**: You must calibrate `BASE_ROTATION_SPEED` for accurate positioning.

3. **Power Loss**: If power is lost, the virtual position resets. You may need to manually home the base after power-up.

4. **Accuracy**: Position accuracy depends on:
   - Consistent rotation speed
   - Accurate calibration
   - No external forces affecting rotation

## Usage

The firmware automatically handles continuous rotation when `BASE_CONTINUOUS_ROTATION` is set to `true`. All existing commands work the same way:

- `move_joints` - Works with virtual position
- `pick` - Uses virtual position for targeting
- `home` - Returns to virtual 90° position

## Switching Back to Standard Servo

To use a standard 180° servo instead:

```cpp
#define BASE_CONTINUOUS_ROTATION  false
```

Then re-upload the firmware.

## Troubleshooting

### Base Doesn't Stop at Target
- Check `BASE_ROTATION_SPEED` calibration
- Verify servo is properly powered
- Check for mechanical binding

### Base Rotates Wrong Direction
- Swap the servo wires OR
- Adjust the rotation direction in `updateContinuousRotation()` (swap 75° and 105°)

### Position Drifts Over Time
- This is expected with continuous rotation servos
- Consider adding a physical home switch for periodic recalibration
- Or use a standard 180° servo for more accurate positioning

