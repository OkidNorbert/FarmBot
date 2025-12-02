# Simple Calibration Guide - Step by Step

## Understanding Coordinates - Simple Explanation

### What Are Real-World X, Y Coordinates?

Think of your workspace like a **grid on the floor**:

```
        Y (Forward - Away from Arm)
        ↑
        |
  200mm |  [Workspace]
        |     ↑
  150mm |     | Tomatoes go here
        |     |
  100mm |     |
   50mm |     |
    0mm └─────┴─────→ X (Right)
        0mm  100mm  200mm
```

**X-axis (Left-Right):**
- **0 mm** = Center of workspace (or left edge)
- **+X** = Move right
- **-X** = Move left

**Y-axis (Forward-Backward):**
- **0 mm** = Front edge (closest to arm)
- **+Y** = Move forward (away from arm)

### How to Determine Your Coordinates

#### **Method 1: Simple Measurement (Easiest)**

1. **Place a ruler** on your workspace
2. **Measure from arm base:**
   - How far **left/right** can arm reach? (This is X range)
   - How far **forward** can arm reach? (This is Y range)
3. **Example:**
   - Arm can reach 150mm left to 150mm right → X: -150 to +150mm
   - Arm can reach 50mm to 250mm forward → Y: 50 to 250mm

#### **Method 2: Use Default Values (Works for Most)**

**Just use these defaults:**
```
X: -150mm (left) to +150mm (right)  = 300mm wide
Y: 50mm (close) to 250mm (far)      = 200mm deep
```

**This works for most setups!** You can adjust later if needed.

#### **Method 3: Calibrate with 4 Points (Most Accurate)**

1. **Place 4 markers** (coins, small objects) at corners of workspace
2. **Measure their positions** with ruler:
   - Top-Left: X = -100mm, Y = 200mm
   - Top-Right: X = +100mm, Y = 200mm
   - Bottom-Left: X = -100mm, Y = 50mm
   - Bottom-Right: X = +100mm, Y = 50mm
3. **Use Calibration page** to map these to camera pixels

## Camera Position - Does It Matter?

### ✅ **YES! Camera Position Matters**

**Best Position: Above Workspace, Pointing Down**

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

**Why This Works Best:**
- ✅ Sees entire workspace
- ✅ No occlusion from arm
- ✅ Easy to calibrate
- ✅ Consistent view

**Important:**
- **Lock camera position** after mounting
- **Don't move camera** after calibration
- **If you move camera, recalibrate!**

### Camera Setup Steps

1. **Mount camera** 30-50cm above workspace
2. **Point straight down** at workspace
3. **Center camera** over workspace
4. **Lock in place** (tape, screws, etc.)
5. **Calibrate** with camera in final position

## ToF Sensor on Claw - How It Works

### Your Setup: ToF on Claw

**Important:** Your ToF sensor is **on top of the claw**, which means:
- ✅ ToF **moves with the arm**
- ✅ When arm approaches tomato, ToF reads **distance from claw to tomato**
- ✅ This is **actually better** for picking!

### How It Works

**Pick Sequence:**
1. **Arm moves to approach position** (above tomato)
2. **ToF sensor reads distance** from claw to tomato surface
3. **Use this distance directly** for picking
4. **No need to calculate** - ToF already measures what we need!

**Example:**
```
Arm at approach position:
  Claw (with ToF)
     ↓
     ↓ 50mm (ToF reading)
     ↓
  Tomato
```

**Depth = 50mm** (use ToF reading directly, maybe subtract 10-20mm for tomato radius)

### Code Updated

The code has been updated to handle ToF on claw:
- Reads ToF distance when arm is at approach position
- Uses distance directly (with small adjustment for tomato radius)
- Works correctly with moving ToF sensor

## Quick Setup Guide

### Step 1: Set Up Workspace

1. **Place workspace** in front of arm
2. **Measure workspace size:**
   - Width (left-right): e.g., 300mm
   - Depth (forward): e.g., 200mm
3. **Use default coordinates:**
   - X: -150 to +150mm
   - Y: 50 to 250mm

### Step 2: Position Camera

1. **Mount camera** above workspace
2. **Point down** at workspace
3. **Lock camera** in place
4. **Don't move** after calibration

### Step 3: Calibrate (Optional but Recommended)

**Option A: Use Default (Easiest)**
- System uses fallback scaling
- Works but less accurate
- Good for testing

**Option B: Calibrate with 4 Points (Best)**
1. Go to **Calibrate** page
2. Place 4 markers at known positions
3. Click on markers in camera image
4. Enter real-world coordinates (mm)
5. Save calibration

### Step 4: Test

1. **Place a tomato** in workspace
2. **Enable automatic mode**
3. **System should detect and pick**
4. **Check if arm reaches tomato correctly**

## Coordinate System Visual Guide

### Top-Down View

```
                    Y (Forward)
                    ↑
                    |
        ┌───────────┼───────────┐
        │           │           │
    -X  │           │           │  +X
  (Left)│           │           │(Right)
        │           │           │
        └───────────┴───────────┘
                    │
                    │
                  Arm Base
                 (Origin)
```

### Side View

```
                    Z (Up)
                    ↑
                    │
        ┌───────────┼───────────┐
        │           │           │
        │  Workspace│           │
        │           │           │
        └───────────┴───────────┘
                    │
                    │
                  Arm Base
                 (Origin)
```

## Default Values (Use These If Unsure)

### Workspace Bounds
```python
X: -150mm to +150mm  (300mm wide)
Y: 50mm to 250mm     (200mm deep)
Z: 20mm to 150mm     (height)
```

### Coordinate Origin
- **X = 0, Y = 0:** Center of workspace (or front-left corner)
- **Adjust based on your setup**

## Summary

### Real-World Coordinates:
- **X, Y in millimeters** - actual physical positions
- **Measure with ruler** or use defaults
- **Calibrate** for best accuracy

### Camera Position:
- **Above workspace** is best
- **Point straight down**
- **Lock in place** after mounting
- **Don't move** after calibration

### ToF on Claw:
- ✅ **Actually better** for picking!
- ✅ Reads distance from claw to tomato
- ✅ Use distance directly
- ✅ Code updated to handle this

**You're all set!** Use the default coordinates to start, then calibrate for better accuracy.

