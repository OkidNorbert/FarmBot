# Arduino Robotic Arm Controller
## AI Tomato Sorter - Arduino Firmware

This Arduino code controls a 5-DOF robotic arm for automatic tomato sorting. It receives commands from a Raspberry Pi and executes precise movements to pick and sort tomatoes based on AI classification, including automatic distance compensation with an ultrasonic sensor.

## 🤖 Hardware Requirements

### **Arduino Board**
- Arduino Uno, Nano, or Mega
- USB cable for programming and communication

### **Servo Motors (5x)**
- **Servo 1**: Base rotation (Pin 3)
- **Servo 2**: Shoulder / main arm joint (Pin 5)  
- **Servo 3**: Elbow / secondary arm joint (Pin 6)
- **Servo 4**: Wrist pitch (Pin 9)
- **Servo 5**: Gripper open/close (Pin 10)

### **Distance Sensor (optional but recommended)**
- HC-SR04 ultrasonic sensor (TRIG → Pin 11, ECHO → Pin 12)
- Provides accurate distance to the tomato to fine-tune the wrist height

### **Power Supply**
- 5V power supply (2A minimum)
- Servo motors require significant current

### **Connections**
- USB cable to Raspberry Pi
- Jumper wires for servo connections

## 🔌 Connection Diagram

```
┌─────────────────┐    USB     ┌─────────────────┐     PWM          ┌────────────────────┐
│   Raspberry Pi  │◄──────────►│    Arduino      │◄────────────────►│  Robotic Arm       │
│                 │            │                 │                   │                    │
│  Camera Input   │            │  PWM Pins       │                   │  Servo 1 (Base)    │
│  Web Interface  │            │  (3,5,6,9,10)   │                   │  Servo 2 (Shoulder)│
│  AI Processing  │            │                 │                   │  Servo 3 (Elbow)   │
│                 │            │  IO Pins        │          ┌──────► │  Servo 4 (Wrist)   │
│  GPIO (optional)│            │  (11,12)        │          │        │  Servo 5 (Gripper) │
└─────────────────┘            └─────────────────┘          │        └────────────────────┘
                                                           │
                                                           │ HC-SR04 Ultrasonic Sensor
                                                           ▼
                                                    TRIG (Pin 11) / ECHO (Pin 12)
```

## 📋 Pin Connections

| Arduino Pin | Component | Description |
|-------------|-----------|-------------|
| **Pin 3** | Servo 1 | Base rotation |
| **Pin 5** | Servo 2 | Shoulder joint |
| **Pin 6** | Servo 3 | Elbow joint |
| **Pin 9** | Servo 4 | Wrist pitch |
| **Pin 10** | Servo 5 | Gripper open/close |
| **Pin 11** | Ultrasonic TRIG | Distance trigger pulse |
| **Pin 12** | Ultrasonic ECHO | Distance measurement |
| **5V (external)** | Servo Power | 5V, 2A+ dedicated supply |
| **GND (common)** | Ground | Tie external supply, servos, and Arduino GND |
| **USB** | Raspberry Pi | Serial communication & Arduino power/logic |

## 🔧 Wiring Diagram

```
                   Arduino Uno / Nano
                  ┌───────────────────┐
                  │                   │
 Servo 1 (Base) ─►│ Pin 3   (PWM)     │
 Servo 2 (Should)│ Pin 5   (PWM)     │◄─ External 5V (+) to servo red wires
 Servo 3 (Elbow) │ Pin 6   (PWM)     │
 Servo 4 (Wrist) │ Pin 9   (PWM)     │
 Servo 5 (Grip) ─►│ Pin 10  (PWM)     │
 Ultrasonic TRIG │ Pin 11 (Digital)  │
 Ultrasonic ECHO │ Pin 12 (Digital)  │
 Common Ground ─►│ GND               │◄─ External 5V ground & servo grounds
 Raspberry Pi ──►│ USB               │
                  │                   │
                  └───────────────────┘
```

## 🚀 Setup Instructions

### **1. Hardware Assembly**
1. Connect all five servos to the designated PWM pins (3, 5, 6, 9, 10)
2. Wire the HC-SR04 ultrasonic sensor (TRIG → 11, ECHO → 12)
3. Power the servos from a dedicated 5V / 2A (or higher) supply and tie grounds together
4. Connect USB cable to Raspberry Pi
5. Double-check that every ground (Arduino, servos, ultrasonic, Pi) is common

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
   - `ANGLE 90 90 90 90 30` - Set all servos to their neutral positions

## 📡 Communication Protocol

### **Commands from Raspberry Pi**

| Command | Format | Description |
|---------|--------|-------------|
| **MOVE** | `MOVE X Y CLASS` | Move to coordinates and sort by class |
| **ANGLE** | `ANGLE A1 A2 A3 A4 A5` | Set servo angles directly (`-1` keeps current) |
| **GRIP** | `GRIP OPEN/CLOSE` | Control gripper |
| **HOME** | `HOME` | Return to home position |
| **STOP** | `STOP` | Emergency stop |
| **STATUS** | `STATUS` | Get system status |

### **Example Commands**
```
MOVE 100 150 1           // Move to (100,150) and sort as class 1 (Ready)
ANGLE 90 60 120 95 150   // Set joints manually (base, shoulder, elbow, wrist, gripper)
ANGLE -1 -1 -1 85 30     // Adjust wrist only, keep other joints unchanged
GRIP CLOSE               // Close gripper
HOME                      // Return to home position
```

## 🎯 Sorting Logic

The system sorts tomatoes into 3 bins based on AI classification:

| Class | Description | Bin Pose (Base, Shoulder, Elbow, Wrist, Gripper) |
|-------|-------------|-----------------------------------------------|
| **0** | Not Ready | (20°, 55°, 120°, 80°, 150°) |
| **1** | Ready | (100°, 50°, 110°, 80°, 150°) |
| **2** | Spoilt | (160°, 60°, 115°, 80°, 150°) |

## ⚙️ Configuration

### **Servo Limits**
```cpp
const int SERVO_PINS[5] = {3, 5, 6, 9, 10};
const int SERVO_MIN[5]  = {0, 10, 0, 0, 20};    // Base, Shoulder, Elbow, Wrist, Gripper
const int SERVO_MAX[5]  = {180, 170, 180, 180, 160};
```

### **Movement Parameters**
```cpp
const int MOVEMENT_DELAY = 40;           // Delay between movement steps (ms)
const int MAX_MOVEMENT_SPEED = 5;        // Max degrees per step per update
const int GRIPPER_OPEN = 30;             // Gripper open position
const int GRIPPER_CLOSE = 150;           // Gripper close position
const int WRIST_NEUTRAL = 90;            // Default wrist position
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
| **Servos move to wrong positions** | Calibrate servo mounting and update limits |
| **Arduino not responding** | Check power, reset Arduino |

### **Debug Commands**
```
STATUS                    // Check system status
ANGLE 90 90 90 90 30      // Test all servos (base, shoulder, elbow, wrist, gripper)
HOME                      // Return to safe position
```

## 📊 Status Information

The `STATUS` command returns:
- Emergency stop status
- Current servo angles
- Servo range limits
- System configuration

## 🔄 Workflow

1. **Initialize**: Arduino starts in home position (90°, 90°, 90°, 90°, gripper open)
2. **Receive Command**: Pi sends `MOVE X Y CLASS`
3. **Calculate Angles**: Inverse kinematics converts coordinates
4. **Measure Distance**: Ultrasonic sensor refines wrist height
5. **Move to Position**: Smooth movement to target
6. **Pick Tomato**: Close gripper
7. **Sort**: Move to appropriate bin based on class
8. **Drop Tomato**: Open gripper
9. **Return Home**: Move back to home position

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
