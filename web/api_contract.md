# API Contract: Hybrid Tomato-Picking System

## Overview
This document defines the communication interface between the Web Backend (Python/Flask) and the Arduino Field Controller, as well as the internal API for the Web Frontend.

## 1. WebSocket Protocol (Arduino <-> Web)
**Endpoint**: `/ws/arduino`
**Namespace**: `/arduino`

### 1.1. Arduino -> Web (Telemetry)
The Arduino sends status updates to the Web Backend.

**Event**: `telemetry`
**Payload**:
```json
{
  "battery_voltage": 12.4,
  "status": "IDLE", // IDLE, MOVING, PICKING, ERROR, CALIBRATING
  "last_action": "HOME",
  "error_code": 0,
  "timestamp": 1638300000
}
```

**Event**: `pick_result`
**Payload**:
```json
{
  "id": "det123",
  "status": "SUCCESS", // SUCCESS, FAILED, ABORTED
  "result": "ripe",
  "to_bin": "right",
  "duration_ms": 4500
}
```

### 1.2. Web -> Arduino (Commands)
The Web Backend sends commands to the Arduino.

**Event**: `command`
**Payloads**:

*   **Pick Request** (Auto Mode):
    ```json
    {
      "cmd": "pick",
      "id": "det123",
      "x": 320, // Pixel X
      "y": 240, // Pixel Y
      "z_depth": 0, // Optional: estimated depth if available
      "class": "ripe", // "ripe" or "unripe"
      "confidence": 0.92
    }
    ```

*   **Manual Move** (Joint Angles):
    ```json
    {
      "cmd": "move_joints",
      "base": 90,
      "shoulder": 45,
      "elbow": 90,
      "forearm": 90,
      "pitch": 90,
      "claw": 0
    }
    ```

*   **System Commands**:
    ```json
    { "cmd": "home" }
    { "cmd": "stop" } // Emergency stop
    { "cmd": "calibrate" }
    { "cmd": "set_mode", "mode": "AUTO" } // AUTO or MANUAL
    ```

## 2. REST API (Frontend <-> Backend)

### 2.1. Control
*   `POST /api/control/mode`: Set system mode (AUTO/MANUAL).
*   `POST /api/control/emergency_stop`: Trigger emergency stop.
*   `POST /api/manual/move`: Send manual joint angles.

### 2.2. Vision
*   `POST /api/vision/detection`: (Internal) YOLO service pushes detection to backend.
    ```json
    {
      "bbox": {"x": 320, "y": 240, "w": 100, "h": 100},
      "class": "ripe",
      "confidence": 0.95,
      "timestamp": "..."
    }
    ```

## 3. Data Types

### 3.1. Servo Config
```json
{
  "pin": 3,
  "min_pulse": 600,
  "max_pulse": 2400,
  "min_angle": 0,
  "max_angle": 180,
  "home_angle": 90
}
```
