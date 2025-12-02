# Coordinate System - Quick Start Guide

## What Are Real-World X, Y Coordinates?

**Simple Answer:** Real-world coordinates are **actual physical positions** measured in **millimeters (mm)**.

### Visual Example

```
Top-Down View of Your Workspace:

        Y (Forward - Away from Arm)
        ↑
        |
  250mm |  ┌─────────────┐
        |  │             │
  200mm |  │  Workspace  │ ← Tomatoes placed here
        |  │             │
  150mm |  │             │
        |  │             │
  100mm |  │             │
        |  │             │
   50mm |  └─────────────┘
    0mm └─────────────────→ X (Right)
         -150mm  0mm  +150mm
         (Left)      (Right)
```

### Coordinate System Explained

**X-Axis (Left-Right):**
- **0 mm** = Center of workspace (or left edge)
- **Negative X** = Left side (e.g., -100mm)
- **Positive X** = Right side (e.g., +100mm)

**Y-Axis (Forward-Backward):**
- **0 mm** = Front edge (closest to arm)
- **Positive Y** = Forward (away from arm)
- Example: Y = 150mm means 150mm forward from arm

## How to Determine Your Coordinates

### Method 1: Use Default Values (Easiest - Start Here!)

**Just use these defaults - they work for most setups:**

```
X: -150mm (left) to +150mm (right)  = 300mm wide workspace
Y: 50mm (close) to 250mm (far)      = 200mm deep workspace
```

**No measurement needed!** The system will use these defaults.

### Method 2: Measure Your Workspace

1. **Place a ruler** on your workspace
2. **Measure from arm base:**
   - How far **left** can arm reach? → This is X minimum (e.g., -150mm)
   - How far **right** can arm reach? → This is X maximum (e.g., +150mm)
   - How far **forward** can arm reach? → This is Y maximum (e.g., 250mm)
   - Minimum forward distance? → This is Y minimum (e.g., 50mm)

3. **Update in code** (optional):
   ```python
   # In hardware_controller.py
   self.workspace_bounds = {
       'x_min': -150, 'x_max': 150,  # Your measured values
       'y_min': 50, 'y_max': 250,    # Your measured values
   }
   ```

### Method 3: Calibrate with 4 Points (Most Accurate)

1. **Place 4 markers** (coins, small objects) at workspace corners
2. **Measure their positions** with ruler
3. **Use Calibration page** in web interface
4. **Click on markers** in camera image
5. **Enter coordinates** (X, Y in mm)
6. **Save calibration**

## Camera Position - Does It Matter?

### ✅ **YES! Camera Position Matters**

**Best Setup: Camera Above Workspace**

```
        Camera (30-50cm above)
           ↓
           ↓
    ┌─────────────┐
    │  Workspace  │ ← Tomatoes here
    │             │
    └─────────────┘
           ↑
        Arm Base
```

**Why This Works:**
- ✅ Sees entire workspace
- ✅ No occlusion from arm
- ✅ Easy to calibrate
- ✅ Consistent view

**Important Rules:**
1. **Mount camera** above workspace (30-50cm high)
2. **Point straight down** at workspace
3. **Lock camera** in place (tape, screws, etc.)
4. **Don't move camera** after calibration
5. **If you move camera, recalibrate!**

### Camera Position Options

**Option 1: Above Workspace (Recommended)**
- Best for accuracy
- Easy to calibrate
- No occlusion

**Option 2: Side View**
- Can see depth better
- More complex calibration
- Arm may block view

**For Your Setup:** Use **above workspace** if possible.

## ToF Sensor on Claw - How It Works

### Your Setup: ToF on Claw ✅

**Important:** Your ToF sensor is **on top of the claw**, which is actually **BETTER** for picking!

### How It Works

**Pick Sequence:**
1. **Arm moves to approach position** (above tomato)
2. **ToF sensor (on claw) reads distance** from claw to tomato
3. **Use this distance directly** - it's already what we need!

**Visual:**
```
Arm at approach position:
  
  Claw (with ToF)
     ↓
     ↓ 50mm ← ToF reading (distance from claw to tomato)
     ↓
  ┌─────┐
  │Tomato│
  └─────┘
```

**Depth = 50mm** (or slightly less to account for tomato radius)

### Why This Is Better

**ToF on Claw Advantages:**
- ✅ Reads distance **directly** from claw to tomato
- ✅ No need to calculate surface distance
- ✅ More accurate for picking
- ✅ Works at any arm position

**Code Updated:**
- System now uses ToF distance directly
- Subtracts small amount for tomato radius
- Works correctly with ToF on claw

## Quick Setup Steps

### Step 1: Set Up Workspace (5 minutes)

1. **Place workspace** in front of arm
2. **Use default coordinates:**
   - X: -150 to +150mm
   - Y: 50 to 250mm
3. **Done!** (You can calibrate later for better accuracy)

### Step 2: Position Camera (5 minutes)

1. **Mount camera** 30-50cm above workspace
2. **Point straight down**
3. **Lock in place**
4. **Don't move** after this

### Step 3: Test (2 minutes)

1. **Place a tomato** in workspace
2. **Enable automatic mode**
3. **System should detect and pick**

### Step 4: Calibrate (Optional - 10 minutes)

1. **Go to Calibrate page**
2. **Place 4 markers** at known positions
3. **Click on markers** in image
4. **Enter coordinates** (mm)
5. **Save calibration**

## Default Values (Use These!)

### Workspace Bounds
```
X: -150mm to +150mm  (300mm wide, centered)
Y: 50mm to 250mm     (200mm deep, starting 50mm from arm)
Z: 20mm to 150mm     (height above surface)
```

### Coordinate Origin
- **X = 0, Y = 0:** Center of workspace
- **Or:** Front-left corner (adjust based on your setup)

## Common Questions

### Q: Do I need to measure everything?
**A:** No! Use default values to start. Calibrate later for better accuracy.

### Q: What if my workspace is different size?
**A:** Adjust `workspace_bounds` in `hardware_controller.py` or calibrate with 4 points.

### Q: Does camera position matter?
**A:** Yes! Above workspace is best. Lock it in place after mounting.

### Q: ToF on claw - is that OK?
**A:** Yes! Actually better. Code is updated to handle it correctly.

### Q: How do I know if coordinates are right?
**A:** Test with a tomato. If arm reaches it correctly, coordinates are good!

## Summary

### Real-World Coordinates:
- **X, Y in millimeters** - physical positions
- **Use defaults** to start (-150 to +150mm X, 50 to 250mm Y)
- **Calibrate** later for better accuracy

### Camera Position:
- **Above workspace** is best
- **Point straight down**
- **Lock in place**
- **Don't move** after calibration

### ToF on Claw:
- ✅ **Actually better** for picking!
- ✅ Reads distance from claw to tomato
- ✅ Code updated to handle it
- ✅ Works correctly

**You're ready to go!** Use default coordinates to start, then calibrate if needed.

