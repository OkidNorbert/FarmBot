# Control Interface Updates

## Summary of Changes

The control interface has been updated to match the intended functionality:

### ✅ **"Start" Button → Records Arm Movements**

**Previous Behavior:** Started automatic mode  
**New Behavior:** Records arm movement sequences

**How it works:**
1. Click "Start Recording" to begin recording
2. Move the arm using the sliders - all movements are tracked
3. Click "Stop Recording" to save the sequence
4. Recording is saved to `saved_recordings/` directory as JSON

**Features:**
- Records all servo movements with timestamps
- Saves as JSON file for playback later
- Button changes to red "Stop Recording" while active

### ✅ **"Automatic" Toggle → Tomato Detection & Picking**

**Previous Behavior:** Just switched manual/auto mode  
**New Behavior:** Starts/stops automatic tomato detection and picking

**How it works:**
1. Toggle "Automatic" ON to start automatic mode
2. System automatically:
   - Uses **YOLO** (or ResNet fallback) to detect tomatoes
   - Uses **ToF sensor** (VL53L0X) to calculate distance
   - Filters for **"ready" (ripe) tomatoes only**
   - Automatically picks ready tomatoes
   - Moves them to the appropriate bin
3. Toggle OFF to stop automatic mode

**Automatic Mode Features:**
- ✅ Uses YOLO for detection (if available)
- ✅ Uses ToF sensor for distance measurement
- ✅ Only picks "ready" tomatoes (filters out unripe/spoilt)
- ✅ Calculates 3D coordinates (x, y from camera, z from ToF)
- ✅ Sends PICK command to Arduino with coordinates
- ✅ Runs continuously until toggled off

## Technical Details

### Recording System

**Frontend (`static/js/modern_control.js`):**
- `startRecordingMovements()` - Toggles recording on/off
- `trackServoChange()` - Tracks each servo movement during recording
- Saves movements with timestamps relative to recording start

**Backend (`web_interface.py`):**
- `save_recording` command handler
- Saves to `saved_recordings/` directory
- JSON format with timestamps and servo angles

### Automatic Mode System

**Frontend (`static/js/modern_control.js`):**
- `toggleAutomaticMode()` - Calls `/api/auto/start` or `/api/auto/stop`
- Updates UI based on toggle state

**Backend (`web_interface.py`):**
- `/api/auto/start` - Enables automatic mode
- `/api/auto/stop` - Disables automatic mode
- Sets `hw_controller.auto_mode = True/False`

**Hardware Controller (`hardware_controller.py`):**
- `process_auto_cycle()` - Main automatic detection loop
- Uses `detect_tomatoes()` which prioritizes YOLO
- Uses `get_distance_sensor()` for ToF distance
- Filters for "ready" tomatoes only
- Sends PICK commands to Arduino

## Button Functions Summary

| Button | Function | Description |
|--------|----------|-------------|
| **Start** | Record Movements | Start/stop recording arm movement sequences |
| **Save** | Save Pose | Save current arm position as a single pose |
| **Reset** | Home Position | Reset all servos to home position |
| **Automatic Toggle** | Auto Detection | Start/stop automatic tomato detection and picking |

## Usage Guide

### Recording Arm Movements

1. **Connect** to Arduino
2. **Move arm** to starting position
3. Click **"Start Recording"** (button turns red)
4. **Move the arm** using sliders - all movements are recorded
5. Click **"Stop Recording"** (button turns blue)
6. Recording saved to `saved_recordings/Recording_YYYYMMDD_HHMMSS.json`

### Automatic Tomato Picking

1. **Connect** to Arduino
2. Ensure **camera** is working
3. Ensure **ToF sensor** is connected
4. Toggle **"Automatic"** ON (top of page)
5. System will:
   - Detect tomatoes using YOLO/ResNet
   - Calculate distance using ToF sensor
   - Pick only "ready" tomatoes
   - Move them to appropriate bin
6. Toggle **"Automatic"** OFF to stop

## File Locations

- **Recordings:** `saved_recordings/Recording_*.json`
- **Poses:** `saved_poses/pose_*.json`

## Automatic Mode Requirements

For automatic mode to work properly:

1. ✅ **YOLO Model** (or ResNet fallback) - for tomato detection
2. ✅ **ToF Sensor** (VL53L0X) - for distance measurement
3. ✅ **Camera** - for capturing frames
4. ✅ **Arduino Connected** - for arm control

The system automatically uses:
- **YOLO** if model is available (best accuracy)
- **ResNet + Color Detection** as fallback
- **ToF sensor** for z-depth calculation
- **Filters** for "ready" tomatoes only

## Notes

- Recording captures all servo movements with timestamps
- Automatic mode only picks "ready" (ripe) tomatoes
- ToF sensor provides accurate distance for 3D positioning
- YOLO provides better detection than color-based methods
- System gracefully falls back if YOLO not available

