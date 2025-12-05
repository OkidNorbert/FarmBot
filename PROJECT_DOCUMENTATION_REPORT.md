# Software Architecture Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Web UI     │  │  Mobile App  │  │  API Client  │          │
│  │  (Browser)   │  │   (Future)    │  │   (Future)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Flask Web Server│
                    │  (web_interface.py)│
                    │  Port: 5000/5001 │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼─────────┐ ┌──────▼──────┐ ┌────────▼─────────┐
│  Application Layer │ │  AI/ML Layer│ │ Hardware Layer   │
│                    │ │             │ │                  │
│ ┌────────────────┐│ │ ┌──────────┐│ │ ┌──────────────┐│
│ │ Pi Controller  ││ │ │ YOLO     ││ │ │ Camera       ││
│ │ (pi_controller)││ │ │ Service  ││ │ │ Interface    ││
│ └────────┬───────┘│ │ └────┬─────┘│ │ └──────┬───────┘│
│          │        │ │      │      │ │        │        │
│ ┌────────▼───────┐│ │ ┌────▼─────┐│ │ ┌──────▼───────┐│
│ │ Hardware       ││ │ │ Model    ││ │ │ Arduino      ││
│ │ Controller     ││ │ │ Inference││ │ │ Communication││
│ │ (hardware_ctrl)││ │ └──────────┘│ │ └──────┬───────┘│
│ └────────┬───────┘│ │             │ │        │        │
└──────────┼────────┘ └─────────────┘ └────────┼────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Communication      │
                 │  (WebSocket/Serial) │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Arduino Firmware   │
                 │  (main_firmware.ino)│
                 └──────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
┌─────────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
│  Servo Manager   │ │ Motion      │ │ Sensor Manager │
│  (servo_manager) │ │ Planner     │ │ (tof_vl53)     │
└─────────┬────────┘ │(motion_plnr)│ └─────────────────┘
          │          └──────┬──────┘
          │                 │
          └─────────┬───────┘
                    │
          ┌─────────▼─────────┐
          │   6-DOF Robotic   │
          │       Arm         │
          │  (Hardware)       │
          └───────────────────┘
```

## Component Details

### 1. User Interface Layer
- **Web UI**: Browser-based interface (Flask templates)
- **Mobile App**: Future mobile application
- **API Client**: RESTful API for external integration

### 2. Application Layer (Raspberry Pi)
- **Flask Web Server**: Main web application server
- **Pi Controller**: Main control logic and coordination
- **Hardware Controller**: Abstraction for hardware communication

### 3. AI/ML Layer
- **YOLO Service**: Real-time object detection service
- **Model Inference**: PyTorch-based classification

### 4. Hardware Layer
- **Camera Interface**: OpenCV camera capture
- **Arduino Communication**: WebSocket/Serial/BLE interface

### 5. Arduino Firmware Layer
- **Main Firmware**: Core control loop and command processing
- **Servo Manager**: Servo motor control
- **Motion Planner**: Pick sequence state machine
- **Sensor Manager**: ToF sensor interface

## Data Flow

```
Camera → YOLO Detection → Web Backend → Arduino → Servo Control → Robotic Arm
   ↑                                                                    │
   └─────────────────── Feedback/Telemetry ───────────────────────────┘
```

## Communication Protocols

- **WebSocket**: Real-time bidirectional (Primary)
- **Serial/USB**: Direct connection (Fallback)
- **BLE**: Bluetooth Low Energy (Wireless option)
- **HTTP/REST**: Web interface and API

