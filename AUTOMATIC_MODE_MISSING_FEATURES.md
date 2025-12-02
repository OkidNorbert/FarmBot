# Missing Features for Automatic Mode

## Critical Missing Features

### 🔴 **1. Pixel-to-World Coordinate Mapping (CRITICAL)**

**Current Issue:**
- `hardware_controller.py` line 411-413 sends **pixel coordinates directly** to Arduino
- Arduino expects **real-world coordinates in millimeters**
- Comment says "Simplified mapping" and "In real usage, map pixels to mm"

**What's Missing:**
```python
# Current code (WRONG):
center_x, center_y = target['center']  # These are pixels (e.g., 320, 240)
self.send_command(f"PICK {center_x} {center_y} {z_depth} {class_id}")  # Sends pixels!

# Should be:
arm_x, arm_y = self.pixel_to_arm_coordinates(center_x, center_y)  # Convert to mm
self.send_command(f"PICK {arm_x} {arm_y} {z_depth} {class_id}")  # Sends mm!
```

**Solution Needed:**
- Integrate homography/calibration system into `hardware_controller.py`
- Use `coordinate_mapper.py` or calibration data from web interface
- Convert pixel coordinates to arm coordinates (mm) before sending PICK command

**Files to Update:**
- `hardware_controller.py` - Add coordinate transformation in `process_auto_cycle()`
- Load calibration data (homography matrix) on initialization

---

### 🔴 **2. Camera-to-Arm Calibration Integration**

**Current Issue:**
- Calibration system exists (`coordinate_mapper.py`, `calibrate_homography.py`)
- Calibration endpoints exist in `web_interface.py`
- **But `hardware_controller.py` doesn't use them!**

**What's Missing:**
- Load calibration data (homography matrix) in `HardwareController.__init__()`
- Store transformation matrix as instance variable
- Use it in `process_auto_cycle()` to convert pixels to mm

**Solution Needed:**
```python
# In HardwareController.__init__():
self.homography_matrix = self.load_calibration()

# In process_auto_cycle():
arm_x, arm_y = self.pixel_to_arm(center_x, center_y)
```

**Files to Update:**
- `hardware_controller.py` - Add calibration loading and coordinate conversion

---

### 🟡 **3. ToF Sensor Positioning Logic**

**Current Issue:**
- ToF sensor reading is taken (`get_distance_sensor()`)
- But it might be reading distance to **workspace surface**, not **tomato**
- No logic to account for tomato height above surface

**What's Missing:**
- Logic to determine if ToF is reading tomato or surface
- Adjustment for tomato height (tomatoes sit on surface, so z = surface_distance - tomato_radius)
- Or use ToF reading when arm is positioned above tomato

**Current Code:**
```python
z_depth = self.get_distance_sensor()  # Might be surface distance, not tomato
```

**Solution Needed:**
- Either: Use ToF when arm is positioned above tomato (requires pre-positioning)
- Or: Estimate tomato height from size and subtract from surface distance
- Or: Use fixed offset (tomatoes are ~25-50mm tall, so z = surface_distance - 25mm)

---

### 🟡 **4. Pick Command Validation**

**Current Issue:**
- No validation that coordinates are within arm workspace
- No check if arm can reach the target position
- Could send invalid coordinates that cause arm errors

**What's Missing:**
- Workspace bounds checking
- Inverse kinematics validation (can arm reach this position?)
- Coordinate range validation before sending PICK command

**Solution Needed:**
```python
# Validate coordinates before sending
if not self.is_position_reachable(arm_x, arm_y, z_depth):
    self.logger.warning(f"Position ({arm_x}, {arm_y}, {z_depth}) not reachable")
    continue  # Skip this tomato
```

---

### 🟡 **5. Pick Sequence Status Tracking**

**Current Issue:**
- Code sends PICK command and waits 5 seconds
- No feedback from Arduino about pick success/failure
- No way to know if pick completed or failed

**What's Missing:**
- Listen for pick result from Arduino (`pick_result` event)
- Track pick status (success, failed, aborted)
- Retry logic for failed picks
- Skip tomatoes that failed multiple times

**Current Code:**
```python
self.send_command(f"PICK {arm_x} {arm_y} {z_depth} {class_id}")
time.sleep(5)  # Just waits, doesn't check result
```

**Solution Needed:**
- Subscribe to Arduino pick result events
- Wait for pick completion with timeout
- Handle success/failure appropriately

---

### 🟢 **6. Multiple Tomato Pick Coordination**

**Current Issue:**
- Code picks tomatoes one by one in sequence
- No logic to avoid picking same tomato twice
- No logic to handle tomatoes that move during picking

**What's Missing:**
- Track which tomatoes have been picked
- Re-detect after each pick to update positions
- Skip tomatoes that are too close to already-picked positions

---

### 🟢 **7. Error Handling & Recovery**

**Current Issue:**
- Limited error handling for failed picks
- No recovery if ToF sensor fails
- No fallback if coordinate conversion fails

**What's Missing:**
- Retry logic for failed picks
- Fallback coordinate estimation if calibration missing
- Graceful degradation if sensors fail

---

## Implementation Priority

### **Priority 1 (CRITICAL - Must Have):**
1. ✅ **Pixel-to-World Coordinate Mapping** - Without this, automatic mode won't work at all
2. ✅ **Calibration Integration** - Required for coordinate mapping

### **Priority 2 (IMPORTANT - Should Have):**
3. ✅ **ToF Sensor Positioning Logic** - Improves accuracy
4. ✅ **Pick Command Validation** - Prevents errors

### **Priority 3 (NICE TO HAVE):**
5. ✅ **Pick Sequence Status Tracking** - Better feedback
6. ✅ **Multiple Tomato Coordination** - Prevents duplicate picks
7. ✅ **Error Handling & Recovery** - More robust

---

## Quick Fixes Needed

### Fix 1: Add Coordinate Conversion

**File:** `hardware_controller.py`

**In `__init__()`:**
```python
# Load calibration/homography
self.homography_matrix = self.load_calibration_matrix()
```

**In `process_auto_cycle()`:**
```python
# Convert pixel to arm coordinates
if self.homography_matrix is not None:
    arm_x, arm_y = self.pixel_to_arm_coordinates(center_x, center_y)
else:
    # Fallback: simple scaling (needs calibration!)
    self.logger.warning("No calibration - using fallback scaling")
    arm_x = center_x / 10.0  # Rough estimate
    arm_y = center_y / 10.0
```

### Fix 2: Add Calibration Loading

**Add method to `HardwareController`:**
```python
def load_calibration_matrix(self):
    """Load homography matrix from calibration file"""
    try:
        # Try to load from web interface calibration
        calib_file = self.project_root / 'calibration_data.json'
        if calib_file.exists():
            import json
            with open(calib_file, 'r') as f:
                data = json.load(f)
                if 'homography' in data:
                    import numpy as np
                    return np.array(data['homography'])
        
        # Try alternative locations
        # ... check other calibration file locations
        
        return None
    except Exception as e:
        self.logger.warning(f"Failed to load calibration: {e}")
        return None

def pixel_to_arm_coordinates(self, pixel_x, pixel_y):
    """Convert pixel coordinates to arm coordinates (mm)"""
    if self.homography_matrix is None:
        # Fallback: simple scaling
        return pixel_x / 10.0, pixel_y / 10.0
    
    import cv2
    import numpy as np
    
    # Convert using homography
    pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    arm_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
    return float(arm_point[0][0][0]), float(arm_point[0][0][1])
```

### Fix 3: Improve ToF Usage

**In `process_auto_cycle()`:**
```python
# Get ToF distance (this is distance to surface)
surface_distance = self.get_distance_sensor()

# Estimate tomato height (tomatoes are ~25-50mm tall)
# Use bounding box size to estimate tomato size
bbox = target.get('bbox', [0, 0, 50, 50])  # [x, y, w, h]
tomato_height_estimate = max(bbox[2], bbox[3]) * 0.4  # Rough estimate

# Z depth = surface distance - tomato height
z_depth = surface_distance - tomato_height_estimate if surface_distance else 50
z_depth = max(20, min(100, z_depth))  # Clamp to reasonable range
```

---

## Summary

**Most Critical Missing Feature:**
🔴 **Pixel-to-World Coordinate Mapping** - Without this, the arm will try to move to pixel coordinates (e.g., 320mm, 240mm) instead of real-world coordinates, causing incorrect positioning.

**Action Required:**
1. Integrate calibration system into `hardware_controller.py`
2. Add `pixel_to_arm_coordinates()` method
3. Use it in `process_auto_cycle()` before sending PICK command

**Without This Fix:**
- Automatic mode will send wrong coordinates
- Arm will move to incorrect positions
- Picks will fail or miss tomatoes

**With This Fix:**
- Automatic mode will work correctly
- Arm will move to actual tomato positions
- Picks will be accurate

