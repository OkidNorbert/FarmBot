# Camera-to-Arm Calibration Guide

This guide explains how to calibrate the camera coordinate system to match the robotic arm's coordinate system.

## Overview

The camera sees the workspace in **pixels** (e.g., 640x480), but the arm moves in **millimeters**. We need to create a mapping between these two coordinate systems.

## Equipment Needed

- Calibrated robotic arm (homed position known)
- Camera mounted above workspace
- Ruler or measuring tape
- Small marker or object for reference points

## Calibration Process

### Step 1: Define Reference Points

Place 4 markers at known positions in the workspace:

| Marker | Arm Position (mm) | Camera Pixel (to be measured) |
|--------|-------------------|-------------------------------|
| A (Top-Left) | (50, 50) | (px_a, py_a) |
| B (Top-Right) | (250, 50) | (px_b, py_b) |
| C (Bottom-Left) | (50, 250) | (px_c, py_c) |
| D (Bottom-Right) | (250, 250) | (px_d, py_d) |

### Step 2: Capture Reference Pixels

1. **Open the web interface** and go to the **Calibrate** page.
2. **Manually move the arm** to each reference position (A, B, C, D).
3. **Capture an image** at each position.
4. **Record the pixel coordinates** of the marker in each image.

### Step 3: Calculate Transformation

Use a simple linear transformation (homography):

```python
# Example calibration code (add to hardware_controller.py)

import numpy as np

# Known arm positions (mm)
arm_points = np.array([
    [50, 50],    # A
    [250, 50],   # B
    [50, 250],   # C
    [250, 250]   # D
])

# Measured camera pixels
camera_points = np.array([
    [120, 80],   # A (example, replace with actual)
    [520, 85],   # B
    [115, 395],  # C
    [515, 400]   # D
])

# Calculate transformation matrix
# This maps camera pixels -> arm mm
def calculate_transform(camera_pts, arm_pts):
    # Using least squares to find affine transform
    # [x_mm]   [a  b  c]   [px]
    # [y_mm] = [d  e  f] * [py]
    # [1   ]   [0  0  1]   [1 ]
    
    A = []
    b = []
    for (px, py), (x_mm, y_mm) in zip(camera_pts, arm_pts):
        A.append([px, py, 1, 0, 0, 0])
        A.append([0, 0, 0, px, py, 1])
        b.append(x_mm)
        b.append(y_mm)
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve for transformation parameters
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    
    transform_matrix = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ])
    
    return transform_matrix

transform = calculate_transform(camera_points, arm_points)
print("Transformation Matrix:")
print(transform)
```

### Step 4: Apply Transformation

Add this function to `hardware_controller.py`:

```python
def pixel_to_mm(self, px, py):
    """Convert camera pixel coordinates to arm millimeter coordinates"""
    # Apply transformation matrix
    pixel_vec = np.array([px, py, 1])
    mm_vec = self.transform_matrix @ pixel_vec
    
    x_mm = mm_vec[0]
    y_mm = mm_vec[1]
    
    return x_mm, y_mm
```

### Step 5: Verify Calibration

1. Place a test object at a known position.
2. Detect it with the camera.
3. Convert pixel coordinates to mm.
4. Command the arm to move to that position.
5. Verify the gripper is directly above the object.

**Acceptable Error:** ±5mm

## Saving Calibration

Store the transformation matrix in a config file:

```yaml
# config.yaml
calibration:
  transform_matrix:
    - [0.5, 0.0, -10.0]
    - [0.0, 0.5, -5.0]
    - [0.0, 0.0, 1.0]
```

Load it on startup in `hardware_controller.py`:

```python
import yaml

def load_calibration(self):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    self.transform_matrix = np.array(config['calibration']['transform_matrix'])
```

## Troubleshooting

- **Arm misses objects:** Re-run calibration with more reference points.
- **Distortion at edges:** Camera lens distortion. Use OpenCV's `cv2.undistort()`.
- **Inconsistent results:** Ensure camera is rigidly mounted and doesn't move.
