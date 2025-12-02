# Limited Servo Mode - Working with Available Servos

## Overview

The system has been configured to work with only **4 available servos** (forearm, elbow, pitch, claw) while base and shoulder servos are manually fixed.

## Current Configuration

### ✅ Available Servos:
- **Forearm** (Index 2) - Working
- **Elbow** (Index 3) - Working  
- **Pitch** (Index 4) - Working
- **Claw** (Index 5) - Working

### 🔧 Manually Fixed Servos:
- **Base** (Index 0) - Fixed at **90°** (manually set)
- **Shoulder** (Index 1) - Fixed at **135°** (manually adjusted for forearm clearance)

## What Was Changed

### 1. **Hardware Controller** (`hardware_controller.py`)

**Added Servo Availability Configuration:**
```python
self.servo_available = {
    'base': False,      # Base servo not available - manually fixed
    'shoulder': False,  # Shoulder servo not available - manually adjusted
    'forearm': True,   # Forearm servo available
    'elbow': True,     # Elbow servo available
    'pitch': True,     # Pitch servo available
    'claw': True       # Claw servo available
}

self.fixed_servo_angles = {
    'base': 90,        # Manually fixed base position
    'shoulder': 135,   # Manually adjusted shoulder (gives forearm clearance)
}
```

**Added Command Filtering:**
- `filter_servo_command()` - Filters ANGLE commands to skip unavailable servos
- Sets unavailable servos to `-1` (no change) in ANGLE commands
- Automatically applied to all servo commands

**Updated Arm Orientation Detection:**
- Uses fixed shoulder angle (135°) for front/back detection
- Works correctly with fixed shoulder position

### 2. **Web Interface** (`web_interface.py`)

**Added Servo Availability Check:**
- Checks if servo is available before sending commands
- Shows message if user tries to control unavailable servo
- Prevents sending commands to base/shoulder

### 3. **Control Interface** (`templates/arm_control.html`)

**Disabled Base and Shoulder Controls:**
- Base slider: Disabled, shows "Fixed" label
- Shoulder slider: Disabled, shows "Fixed" label
- Visual indication (opacity 0.5) that they're not available
- Tooltips explain they're manually fixed

## How It Works

### Command Filtering

When a servo command is sent:
1. System checks if servos are available
2. Unavailable servos (base/shoulder) are set to `-1` in ANGLE command
3. Arduino receives: `ANGLE -1 -1 90 90 90 0` (only moves available servos)
4. Fixed servos remain at their manual positions

### Automatic Picking

Automatic mode works with limited servos:
1. **Detection:** Still detects tomatoes using YOLO/ResNet
2. **Coordinate Conversion:** Converts pixel → arm coordinates
3. **Pick Command:** Sends PICK command with coordinates
4. **Arduino:** Uses only available servos (forearm, elbow, pitch, claw) to reach position
5. **Fixed Servos:** Base and shoulder remain at fixed positions

### Manual Control

Manual control works normally:
- Forearm, Elbow, Pitch, Claw sliders work as usual
- Base and Shoulder sliders are disabled (grayed out)
- System automatically filters out base/shoulder commands

## Adjusting Fixed Positions

If you need to change the fixed shoulder position:

**In `hardware_controller.py`:**
```python
self.fixed_servo_angles = {
    'base': 90,        # Change if needed
    'shoulder': 135,   # Adjust this value (90-180° recommended)
}
```

**Recommended Shoulder Angles:**
- **90-100°**: Arm more vertical, less reach
- **120-135°**: Good balance, forearm can reach floor
- **150-165°**: Arm more horizontal, maximum reach

**Current Setting (135°):**
- Provides good clearance for forearm to reach floor
- Allows reasonable workspace coverage
- Good balance between reach and stability

## Workspace Considerations

With fixed base and shoulder:
- **Base fixed at 90°**: Arm faces forward (straight ahead)
- **Shoulder fixed at 135°**: Arm angled forward and down
- **Reach:** Forearm, elbow, pitch, and claw can still reach tomatoes
- **Limitation:** Cannot rotate base or adjust shoulder angle
- **Solution:** Position tomatoes within reach of fixed arm configuration

## Testing

### Test Manual Control:
1. Go to Control page
2. Try moving Forearm, Elbow, Pitch, Claw - should work
3. Try moving Base/Shoulder - should be disabled

### Test Automatic Mode:
1. Toggle "Automatic" ON
2. Place tomatoes in front of arm (within reach)
3. System should detect and pick using available servos
4. Check logs for successful picks

### Verify Fixed Positions:
1. Check that base is at 90° (manually)
2. Check that shoulder is at 135° (manually)
3. Verify forearm can reach floor with this configuration

## When More Servos Arrive

To re-enable base and shoulder servos:

**In `hardware_controller.py`:**
```python
self.servo_available = {
    'base': True,      # Re-enable base
    'shoulder': True,  # Re-enable shoulder
    'forearm': True,
    'elbow': True,
    'pitch': True,
    'claw': True
}
```

**In `templates/arm_control.html`:**
- Remove `disabled` attribute from base/shoulder sliders
- Remove `opacity: 0.5` and `pointer-events: none` styles
- Remove "(Fixed)" labels

## Summary

✅ **System works with 4 servos:**
- Forearm, Elbow, Pitch, Claw are fully functional
- Base and Shoulder are manually fixed
- Automatic picking works with limited servos
- Manual control works for available servos

✅ **Features:**
- Command filtering prevents sending commands to unavailable servos
- Control interface shows which servos are fixed
- Automatic mode adapts to available servos
- Coordinate conversion still works

**The system is ready to use with your current hardware setup!** 🎉

