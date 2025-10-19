# Arduino Robotic Arm Controller
## AI Tomato Sorter - Arduino Firmware

This Arduino code controls a 3-DOF robotic arm for automatic tomato sorting. It receives commands from a Raspberry Pi and executes precise movements to pick and sort tomatoes based on AI classification.

## 🤖 Hardware Requirements

### **Arduino Board**
- Arduino Uno, Nano, or Mega
- USB cable for programming and communication

### **Servo Motors (3x)**
- **Servo 1**: Base rotation (SG90 or similar)
- **Servo 2**: Arm joint (SG90 or similar)  
- **Servo 3**: Gripper (SG90 or similar)

### **Power Supply**
- 5V power supply (2A minimum)
- Servo motors require significant current

### **Connections**
- USB cable to Raspberry Pi
- Jumper wires for servo connections

## 🔌 Connection Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │    Arduino      │    │  Robotic Arm    │
│                 │    │                 │    │                 │
│  USB Port       │◄──►│  USB Port       │    │                 │
│                 │    │                 │    │                 │
│  GPIO Pins      │    │  Digital Pins   │    │  Servo Motors   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │   Servo 1    │ │   Servo 2    │ │   Servo 3    │
            │  (Base)     │ │  (Arm)      │ │ (Gripper)   │
            │   Pin 3     │ │   Pin 5     │ │   Pin 6     │
            └─────────────┘ └─────────────┘ └─────────────┘
```

## 📋 Pin Connections

| Arduino Pin | Component | Description |
|-------------|-----------|-------------|
| **Pin 3** | Servo 1 | Base rotation motor |
| **Pin 5** | Servo 2 | Arm joint motor |
| **Pin 6** | Servo 3 | Gripper motor |
| **5V** | Servo Power | Power for all servos |
| **GND** | Servo Ground | Ground for all servos |
| **USB** | Raspberry Pi | Serial communication |

## 🔧 Wiring Diagram

```
                    Arduino Uno
                   ┌─────────────┐
                   │             │
    Servo 1 ──────►│ Pin 3       │
    (Base)         │             │
                   │             │
    Servo 2 ──────►│ Pin 5       │
    (Arm)          │             │
                   │             │
    Servo 3 ──────►│ Pin 6       │
    (Gripper)      │             │
                   │             │
    Power + ──────►│ 5V          │
                   │             │
    Power - ──────►│ GND         │
                   │             │
    Raspberry Pi──►│ USB         │
                   └─────────────┘
```

## 🚀 Setup Instructions

### **1. Hardware Assembly**
1. Connect servos to Arduino pins as shown above
2. Connect power supply (5V, 2A minimum)
3. Connect USB cable to Raspberry Pi
4. Ensure all connections are secure

### **2. Software Setup**
1. Open Arduino IDE
2. Load `tomato_sorter_arduino.ino`
3. Select correct Arduino board and port
4. Upload the code to Arduino
5. Open Serial Monitor (115200 baud) to verify

### **3. Testing**
1. Power on Arduino
2. Check Serial Monitor for "Tomato Sorter Arduino - Ready"
3. Send test commands:
   - `STATUS` - Check system status
   - `HOME` - Move to home position
   - `ANGLE 90 90 90` - Set all servos to 90°

## 📡 Communication Protocol

### **Commands from Raspberry Pi**

| Command | Format | Description |
|---------|--------|-------------|
| **MOVE** | `MOVE X Y CLASS` | Move to coordinates and sort by class |
| **ANGLE** | `ANGLE A1 A2 A3` | Set servo angles directly |
| **GRIP** | `GRIP OPEN/CLOSE` | Control gripper |
| **HOME** | `HOME` | Return to home position |
| **STOP** | `STOP` | Emergency stop |
| **STATUS** | `STATUS` | Get system status |

### **Example Commands**
```
MOVE 100 150 1        // Move to (100,150) and sort as class 1 (Ready)
ANGLE 90 45 0        // Set servos to 90°, 45°, 0°
GRIP CLOSE           // Close gripper
HOME                 // Return to home position
```

## 🎯 Sorting Logic

The system sorts tomatoes into 3 bins based on AI classification:

| Class | Description | Bin Position | Servo Angles |
|-------|-------------|--------------|--------------|
| **0** | Not Ready | Bin 1 | (0°, 45°) |
| **1** | Ready | Bin 2 | (90°, 45°) |
| **2** | Spoilt | Bin 3 | (180°, 45°) |

## ⚙️ Configuration

### **Servo Limits**
```cpp
const int SERVO1_MIN = 0;    // Base rotation minimum
const int SERVO1_MAX = 180;  // Base rotation maximum
const int SERVO2_MIN = 0;    // Arm joint minimum
const int SERVO2_MAX = 180;  // Arm joint maximum
const int SERVO3_MIN = 0;    // Gripper minimum
const int SERVO3_MAX = 180;  // Gripper maximum
```

### **Movement Parameters**
```cpp
const int MOVEMENT_DELAY = 50;        // Delay between movements (ms)
const int MAX_MOVEMENT_SPEED = 5;    // Max degrees per step
const int GRIPPER_OPEN = 0;          // Gripper open position
const int GRIPPER_CLOSE = 180;       // Gripper close position
```

### **Arm Dimensions**
```cpp
const float ARM_LENGTH1 = 100.0;  // First arm segment (mm)
const float ARM_LENGTH2 = 80.0;   // Second arm segment (mm)
```

## 🔒 Safety Features

### **Emergency Stop**
- `STOP` command immediately halts all movement
- Emergency stop flag prevents new movements
- Use `HOME` command to reset emergency stop

### **Movement Limits**
- All servo angles constrained to 0-180°
- Smooth movement prevents jerky motion
- Speed limiting prevents damage

### **Error Handling**
- Invalid commands are rejected
- Unreachable positions are detected
- Serial communication errors are handled

## 🐛 Troubleshooting

### **Common Issues**

| Problem | Solution |
|---------|----------|
| **Servos not moving** | Check power supply (5V, 2A) |
| **Jittery movement** | Check connections, reduce speed |
| **Serial communication fails** | Check USB cable, baud rate (115200) |
| **Servos move to wrong positions** | Calibrate servo mounting |
| **Arduino not responding** | Check power, reset Arduino |

### **Debug Commands**
```
STATUS              // Check system status
ANGLE 90 90 90     // Test all servos
HOME               // Return to safe position
```

## 📊 Status Information

The `STATUS` command returns:
- Emergency stop status
- Current servo angles
- Servo range limits
- System configuration

## 🔄 Workflow

1. **Initialize**: Arduino starts in home position (90°, 90°, 90°)
2. **Receive Command**: Pi sends `MOVE X Y CLASS`
3. **Calculate Angles**: Inverse kinematics converts coordinates
4. **Move to Position**: Smooth movement to target
5. **Pick Tomato**: Close gripper
6. **Sort**: Move to appropriate bin based on class
7. **Drop Tomato**: Open gripper
8. **Return Home**: Move back to home position

## 📝 Notes

- **Power**: Servos require significant current - use external power supply
- **Calibration**: Adjust servo mounting and limits for your specific arm
- **Safety**: Always test movements in safe area first
- **Maintenance**: Check connections regularly

## 🔗 Integration with Raspberry Pi

The Arduino works seamlessly with the Raspberry Pi:

1. **Pi detects tomatoes** using AI camera system
2. **Pi calculates coordinates** from camera pixels
3. **Pi sends commands** to Arduino via serial
4. **Arduino executes movements** and sorts tomatoes
5. **Arduino reports status** back to Pi

This creates a complete autonomous tomato sorting system! 🍅🤖✨
