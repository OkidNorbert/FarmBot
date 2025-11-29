# System Architecture: How Web Communicates with Arduino

## Overview

The Tomato Sorter system uses a **dual communication protocol** to connect the web interface with the Arduino controller:

1. **Bluetooth Low Energy (BLE)** - Primary method
2. **Serial/USB** - Fallback method for debugging

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Python)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Flask Web Server (web_interface.py)                  │  │
│  │  - REST API endpoints                                  │  │
│  │  - WebSocket server (SocketIO)                        │  │
│  │  - HTML/JavaScript frontend                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Hardware Controller (hardware_controller.py)         │  │
│  │  - BLE connection manager                              │  │
│  │  - Serial connection manager                           │  │
│  │  - Command translator                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ BLE / Serial
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Arduino Uno (Firmware)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  BLE Service: "FarmBot"                               │  │
│  │  - Command Characteristic (UUID: 19B10001-...)       │  │
│  │  - Receives commands from web                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Command Processor (processCommand)                   │  │
│  │  - Parses text commands                               │  │
│  │  - Executes servo movements                            │  │
│  │  - Controls robotic arm                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Hardware Control                                     │  │
│  │  - 6x Servo motors (Base, Shoulder, Elbow, etc.)     │  │
│  │  - VL53L0X Distance Sensor                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Communication Flow

### 1. **Web Interface → Arduino (Commands)**

#### Step 1: User Action in Web Browser
- User clicks button or triggers action in web interface
- JavaScript sends request to Flask backend via REST API or WebSocket

#### Step 2: Flask Backend Processing
```python
# In web_interface.py
@app.route('/api/control/move', methods=['POST'])
def move_arm():
    # Get command from web request
    data = request.json
    x = data['x']
    y = data['y']
    class_id = data['class']
    
    # Send to hardware controller
    hw_controller.send_command(f"MOVE {x} {y} {class_id}")
```

#### Step 3: Hardware Controller Translation
```python
# In hardware_controller.py
def send_command(self, command):
    if self.connection_type == 'bluetooth':
        # Send via BLE
        self.ble_client.write_characteristic(
            characteristic_uuid="19B10001-E8F2-537E-4F6C-D104768A1214",
            value=command
        )
    elif self.connection_type == 'serial':
        # Send via Serial
        self.serial_connection.write(f"{command}\n")
```

#### Step 4: Arduino Receives Command
```cpp
// In tomato_sorter_arduino.ino
void loop() {
    // Poll for BLE events
    BLE.poll();
    
    // Check for BLE commands
    if (commandCharacteristic.written()) {
        String command = commandCharacteristic.value();
        command.trim();
        processCommand(command);  // Execute command
    }
    
    // Also check Serial (for debugging)
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
}
```

#### Step 5: Command Execution
```cpp
void processCommand(String command) {
    if (command.startsWith("MOVE")) {
        processMoveCommand(command);  // Move arm to coordinates
    }
    else if (command.startsWith("ANGLE")) {
        processAngleCommand(command);  // Set servo angles
    }
    else if (command.startsWith("PICK")) {
        processPickCommand(command);  // Pick and sort tomato
    }
    // ... other commands
}
```

---

### 2. **Arduino → Web Interface (Status/Telemetry)**

#### Arduino Sends Status
```cpp
// Arduino periodically sends status via Serial
void reportStatusBrief() {
    Serial.println("=== STATUS REPORT ===");
    Serial.print("Servo Angles: Base=");
    Serial.print(current_angles[SERVO_BASE]);
    // ... more status info
}
```

#### Web Interface Reads Status
```python
# In hardware_controller.py
def read_status(self):
    if self.connection_type == 'serial':
        # Read from serial port
        status = self.serial_connection.readline()
        return parse_status(status)
    # BLE status reading (if implemented)
```

---

## Command Protocol

### Command Format

The Arduino uses **text-based commands** (not JSON). Commands are simple strings:

#### Available Commands

1. **MOVE** - Move arm to world coordinates
   ```
   MOVE X Y CLASS
   Example: MOVE 100 150 1
   ```
   - `X, Y`: World coordinates (mm)
   - `CLASS`: Tomato class (0=not ready, 1=ready, 2=spoilt)

2. **PICK** - Pick tomato at coordinates
   ```
   PICK X Y Z CLASS
   Example: PICK 100 150 50 1
   ```
   - `X, Y, Z`: 3D coordinates (mm)
   - `CLASS`: Tomato class

3. **ANGLE** - Set servo angles directly
   ```
   ANGLE A1 A2 A3 A4 A5 A6
   Example: ANGLE 90 90 90 90 90 30
   ```
   - `A1-A6`: Servo angles (0-180°)
   - Use `-1` to keep current angle

4. **HOME** - Return to home position
   ```
   HOME
   ```

5. **STOP** - Emergency stop
   ```
   STOP
   ```

6. **GRIP** or **GRIPPER** - Control gripper
   ```
   GRIP OPEN
   GRIP CLOSE
   ```

7. **STATUS** - Get current status
   ```
   STATUS
   ```

8. **DISTANCE** - Get distance sensor reading
   ```
   DISTANCE
   ```

---

## Communication Methods

### Method 1: Bluetooth Low Energy (BLE) - Primary

**Arduino Side:**
- Arduino advertises as **"FarmBot"**
- BLE Service UUID: `19B10000-E8F2-537E-4F6C-D104768A1214`
- Command Characteristic UUID: `19B10001-E8F2-537E-4F6C-D104768A1214`
- Commands sent as **text strings** (not JSON)

**Web Side:**
```python
# In hardware_controller.py
hw_controller = HardwareController(
    connection_type='bluetooth',
    ble_device_name="FarmBot"
)
```

**Advantages:**
- Wireless connection
- No USB cable needed
- Works from distance

**Limitations:**
- Range limited (~10 meters)
- Requires BLE-capable Arduino (Arduino R4 WiFi/UNO R4 WiFi)

---

### Method 2: Serial/USB - Fallback

**Arduino Side:**
- Serial communication at **115200 baud**
- Commands sent as text strings ending with `\n`

**Web Side:**
```python
# In hardware_controller.py
hw_controller = HardwareController(
    connection_type='serial',
    serial_port='/dev/ttyUSB0'  # or COM3 on Windows
)
```

**Advantages:**
- Works with any Arduino
- Reliable connection
- Good for debugging

**Limitations:**
- Requires USB cable
- Limited range

---

## Example: Complete Pick Sequence

### 1. Web Interface Receives Detection
```python
# YOLO detects tomato at pixel (320, 240)
detection = {
    'x': 320,
    'y': 240,
    'class': 'ripe',
    'confidence': 0.92
}
```

### 2. Convert Pixel to World Coordinates
```python
# Convert camera pixel to robot coordinates
world_x, world_y = pixel_to_world(detection['x'], detection['y'])
# Result: world_x=100mm, world_y=150mm
```

### 3. Send Command to Arduino
```python
# Send via hardware controller
command = f"MOVE {world_x} {world_y} {class_id}"
hw_controller.send_command(command)
# Command: "MOVE 100 150 1"
```

### 4. Arduino Processes Command
```cpp
// Arduino receives: "MOVE 100 150 1"
void processMoveCommand(String command) {
    float x = 100.0;  // Parsed from command
    float y = 150.0;
    int class_id = 1;
    
    // Convert to servo angles using inverse kinematics
    float baseAngle, shoulderAngle, elbowAngle;
    inverseKinematics(x, y, baseAngle, shoulderAngle, elbowAngle);
    
    // Move arm to position
    performPickAndSort(baseAngle, shoulderAngle, elbowAngle, class_id);
}
```

### 5. Arm Executes Movement
- Base rotates to calculated angle
- Shoulder and elbow move to reach coordinates
- Gripper opens, picks tomato
- Arm moves to sorting bin
- Gripper closes, releases tomato
- Arm returns to home position

---

## Status Reporting

### Arduino → Web (Periodic Updates)

Arduino sends status every 5 seconds via Serial:
```
=== STATUS REPORT ===
Servos Attached: YES
Emergency Stop: INACTIVE
Servo Angles: Base=90°, Shoulder=90°, Elbow=90°, ...
BLE Connected: YES
===================
```

Web interface can parse this and update dashboard.

---

## Error Handling

### Connection Failures
- If BLE connection fails, web interface can fall back to Serial
- If Serial connection fails, web interface shows error message

### Command Failures
- Arduino validates all commands
- Invalid commands are rejected with error message
- Emergency stop can be triggered at any time

### Safety Features
- Servos don't attach until first command (prevents unwanted movement)
- All angles constrained to safe limits
- Emergency stop immediately halts all movement

---

## Configuration

### Arduino Configuration
```cpp
// In tomato_sorter_arduino.ino
// BLE Service Name
BLE.setLocalName("FarmBot");

// Servo Pins
const int SERVO_PINS[6] = {7, 6, 5, 4, 3, 2};
// Base, Shoulder, Elbow, WristYaw, WristPitch, Gripper
```

### Web Interface Configuration
```python
# In web_interface.py
hw_controller = HardwareController(
    connection_type='auto',  # 'auto', 'bluetooth', or 'serial'
    ble_device_name="FarmBot",
    serial_port='/dev/ttyUSB0'  # For serial fallback
)
```

---

## Summary

**Communication Path:**
1. **Web Browser** → User clicks button
2. **Flask Backend** → Processes request
3. **Hardware Controller** → Translates to Arduino command
4. **BLE/Serial** → Sends text command to Arduino
5. **Arduino** → Parses and executes command
6. **Servos** → Move robotic arm
7. **Arduino** → Sends status back via Serial
8. **Web Interface** → Updates dashboard

**Key Points:**
- Commands are **text strings**, not JSON
- BLE is primary method, Serial is fallback
- Arduino processes commands sequentially
- Status is sent periodically via Serial
- All commands are validated and constrained for safety

