# Coordinate System Explained - Simple Guide

## Understanding Real-World Coordinates

### What Are Real-World X, Y Coordinates?

**Real-world coordinates** are the **actual physical positions** in your workspace, measured in **millimeters (mm)**.

Think of it like a map:
- **Camera sees:** Pixels (e.g., 320, 240) - like coordinates on a photo
- **Arm needs:** Millimeters (e.g., 100mm, 150mm) - like real-world GPS coordinates

### Coordinate System Setup

#### **X-Axis (Left-Right)**
- **0 mm** = Center of workspace (or left edge, depending on your setup)
- **Positive X** = Right direction
- **Negative X** = Left direction
- **Range:** Typically -150mm to +150mm (300mm total width)

#### **Y-Axis (Forward-Backward)**
- **0 mm** = Base of arm (or front edge)
- **Positive Y** = Forward direction (away from arm)
- **Range:** Typically 50mm to 250mm (200mm reach)

#### **Z-Axis (Up-Down)**
- **0 mm** = Workspace surface (floor/table)
- **Positive Z** = Up (height above surface)
- **Range:** Typically 20mm to 150mm

### Visual Example

```
                    Y (Forward)
                    ↑
                    |
        ┌───────────┼───────────┐
        │           │           │
        │   -X      │      +X   │
    ←───┼───────────┼───────────┼───→
        │  (Left)   │  (Right)  │
        │           │           │
        └───────────┴───────────┘
                    │
                    │
                  Arm Base
                 (Origin)
```

## How to Determine Your Coordinate System

### Method 1: Measure Your Workspace (Easiest)

1. **Place a ruler or measuring tape** on your workspace
2. **Measure from arm base:**
   - **X = 0:** Center of arm base (or left edge)
   - **Y = 0:** Front edge of workspace (closest to arm)
3. **Mark 4 reference points:**
   - **Point A (Top-Left):** X = -100mm, Y = 200mm
   - **Point B (Top-Right):** X = +100mm, Y = 200mm
   - **Point C (Bottom-Left):** X = -100mm, Y = 50mm
   - **Point D (Bottom-Right):** X = +100mm, Y = 50mm

4. **Use these points for calibration**

### Method 2: Use Arm Movement (More Accurate)

1. **Manually move arm** to a known position
2. **Place a marker** (coin, small object) at that position
3. **Record:**
   - **Arm position:** Servo angles
   - **Real-world position:** Measure with ruler
4. **Repeat for 4+ points**
5. **Use calibration page** to map pixels → real-world

### Method 3: Simple Default Setup

If you don't know exact coordinates, use a **simple default**:

```
Workspace: 300mm x 200mm rectangle
Origin (0,0): Center of workspace
X: -150mm (left) to +150mm (right)
Y: 50mm (close) to 250mm (far)
```

**This works for most setups!**

## Camera Position - Does It Matter?

### ✅ **YES, Camera Position Matters!**

The camera position affects:
1. **Field of view** - What the camera can see
2. **Coordinate mapping** - How pixels map to real-world positions
3. **Calibration accuracy** - Different positions need different calibration

### Recommended Camera Positions

#### **Option 1: Above Workspace (Best)**
```
Camera
  ↓
  ↓
[Workspace]
  ↑
  Arm
```

**Advantages:**
- ✅ Sees entire workspace
- ✅ No occlusion from arm
- ✅ Easy to calibrate
- ✅ Consistent view

**Setup:**
- Mount camera 30-50cm above workspace
- Point straight down
- Center over workspace

#### **Option 2: Side View**
```
Camera → [Workspace] ← Arm
```

**Advantages:**
- ✅ Can see depth better
- ✅ Less occlusion

**Disadvantages:**
- ⚠️ More complex calibration
- ⚠️ Arm may block view

### Camera Position for Your Setup

**Since base/shoulder are fixed:**
- **Camera should be positioned** to see the workspace area your arm can reach
- **Calibrate** with camera in its final position
- **Don't move camera** after calibration (or recalibrate if moved)

## ToF Sensor on Claw - Important Considerations

### ⚠️ **ToF on Claw Changes Everything!**

**Your Setup:**
- ToF sensor is **on top of the claw**
- ToF **moves with the arm**
- ToF distance = **distance from claw to object**

### How This Affects Depth Calculation

#### **Current Code Assumption:**
The code assumes ToF is **fixed above workspace**, measuring distance to surface.

#### **Your Actual Setup:**
ToF is **on the claw**, so:
- When arm moves, ToF moves
- Distance reading changes as arm approaches
- Need to account for claw position

### Updated Depth Calculation Needed

**Current Code:**
```python
# Assumes ToF is fixed above workspace
surface_distance = get_distance_sensor()  # Distance to surface
depth = surface_distance - tomato_radius  # Depth for picking
```

**What You Actually Need:**
```python
# ToF is on claw, moves with arm
# When arm is at approach position, ToF reads distance to tomato
# This IS the depth we need!
tof_distance = get_distance_sensor()  # Distance from claw to tomato
depth = tof_distance  # Use directly (or adjust for claw offset)
```

### Solution

Since ToF is on the claw:
1. **Move arm to approach position** (above tomato)
2. **Read ToF distance** - this is distance from claw to tomato
3. **Use this distance directly** for picking
4. **No need to subtract tomato radius** - ToF already measures to tomato surface

## Practical Setup Guide

### Step 1: Set Up Your Workspace

1. **Place workspace** in front of arm
2. **Measure workspace:**
   - Width (left-right): e.g., 300mm
   - Depth (forward): e.g., 200mm
3. **Define origin:**
   - X = 0: Center or left edge
   - Y = 0: Front edge (closest to arm)

### Step 2: Position Camera

1. **Mount camera above workspace**
2. **Point straight down**
3. **Center over workspace**
4. **Lock camera position** (don't move after calibration)

### Step 3: Calibrate Coordinate System

**Option A: Use Calibration Page (Recommended)**
1. Go to **Calibrate** page in web interface
2. **Manually move arm** to 4 known positions
3. **Click on image** where arm is pointing
4. **Enter real-world coordinates** (X, Y in mm)
5. **Save calibration**

**Option B: Simple Default**
1. Use default workspace bounds:
   - X: -150 to +150mm
   - Y: 50 to 250mm
2. System will use fallback scaling
3. Less accurate but works

### Step 4: Adjust for ToF on Claw

The code needs a small update to handle ToF on claw properly. I can update it if you want.

## Quick Reference

### Coordinate System
- **X-axis:** Left (-) to Right (+)
- **Y-axis:** Back (0) to Forward (+)
- **Z-axis:** Down (0) to Up (+)
- **Units:** Millimeters (mm)

### Default Workspace
```
X: -150mm to +150mm (300mm wide)
Y: 50mm to 250mm (200mm deep)
Z: 20mm to 150mm (height)
```

### Camera Position
- **Best:** Above workspace, pointing down
- **Height:** 30-50cm above workspace
- **Important:** Don't move after calibration

### ToF on Claw
- **Current:** ToF reads distance from claw to object
- **Use:** Distance directly (no subtraction needed)
- **Update needed:** Code assumes fixed ToF

## Next Steps

1. **Measure your workspace** (width and depth)
2. **Position camera** above workspace
3. **Calibrate** using 4 reference points
4. **Update depth calculation** for ToF on claw (I can help with this)

Would you like me to:
1. Update the depth calculation for ToF on claw?
2. Create a simple calibration tool?
3. Add default workspace configuration?

