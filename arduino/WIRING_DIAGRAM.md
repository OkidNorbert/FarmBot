# Wiring Diagram: Hybrid Tomato-Picking System

## Arduino UNO R4 WiFi Pinout

### Servo Connections (D2-D7)

| Pin | Servo | Type | Description | Power |
|-----|-------|------|-------------|-------|
| **D2** | Claw | SG90 | Gripper open/close | 5V Logic |
| **D3** | Pitch | SG90 | Wrist pitch control | 5V Logic |
| **D4** | Elbow | SG90 | Elbow joint | 5V Logic |
| **D5** | Forearm | MG99x/MG996R | Forearm rotation | External 5-6V |
| **D6** | Shoulder | MG99x/MG996R | Shoulder joint | External 5-6V |
| **D7** | Base | MG99x/MG996R | Base rotation | External 5-6V |

### Sensor Connections

| Component | Pin | Description |
|-----------|-----|-------------|
| **VL53L0X** | SDA (I2C) | Data line |
| **VL53L0X** | SCL (I2C) | Clock line |
| **VL53L0X** | VCC | 3.3V or 5V (check module) |
| **VL53L0X** | GND | Ground |

### Safety & Control

| Pin | Component | Description |
|-----|-----------|-------------|
| **D8** | Emergency Stop | Hardware kill switch (Active LOW) |
| **A0** | Battery Monitor | Voltage divider input (optional) |

## Power Supply Requirements

### Arduino Power
- **USB Power**: 5V @ 500mA (for Arduino only)
- **OR External**: 7-12V DC via barrel jack

### Servo Power (CRITICAL)
- **External Supply**: 5-6V DC @ 2-5A (depending on load)
- **Must be separate from Arduino power**
- **Common Ground**: Tie external supply GND to Arduino GND

### Power Decoupling
- **1000 µF electrolytic** capacitor across servo power rails
- **0.1 µF ceramic** capacitor for high-frequency decoupling
- Place capacitors near servo power input

## Wiring Diagram (Text)

```
┌─────────────────────────────────────────────────────────┐
│              Arduino UNO R4 WiFi                         │
│                                                           │
│  D2 ──────► Claw Servo (SG90) ────► 5V External         │
│  D3 ──────► Pitch Servo (SG90) ────► 5V External        │
│  D4 ──────► Elbow Servo (SG90) ────► 5V External        │
│  D5 ──────► Forearm Servo (MG99x) ─► 5-6V External      │
│  D6 ──────► Shoulder Servo (MG99x) ─► 5-6V External      │
│  D7 ──────► Base Servo (MG99x) ────► 5-6V External      │
│                                                           │
│  SDA ─────► VL53L0X SDA                                  │
│  SCL ─────► VL53L0X SCL                                  │
│  3.3V ────► VL53L0X VCC                                  │
│  GND ─────► VL53L0X GND                                  │
│                                                           │
│  D8 ──────► Emergency Stop Switch ──► GND (when pressed)│
│  A0 ──────► Battery Voltage Divider (optional)           │
│                                                           │
│  GND ─────► Common Ground (all components)                │
└─────────────────────────────────────────────────────────┘

External Power Supply (5-6V, 2-5A)
  ├──► +5V ──► Servos (Red wires)
  └──► GND ──► Common Ground ──► Arduino GND
```

## Servo Pulse Width Configuration

### SG90 Servos (Claw, Pitch, Elbow)
- **Default Range**: 500-2400 microseconds
- **Standard**: `attach(pin, 500, 2400)`

### MG99x/MG996R Servos (Forearm, Shoulder, Base)
- **Calibrated Range**: 600-2400 microseconds
- **Use**: `attach(pin, 600, 2400)` for better control

## Safety Notes

1. **Power Sequencing**: 
   - Power on Arduino FIRST
   - Wait for boot (servos will home to 90°)
   - THEN enable external servo power supply

2. **Ground Connection**:
   - **CRITICAL**: All grounds must be connected
   - Arduino GND ↔ External Supply GND ↔ Servo GNDs ↔ Sensor GNDs

3. **Current Requirements**:
   - 3x SG90: ~300mA total
   - 3x MG99x: ~1.5-2A peak (per servo under load)
   - **Total**: 5A supply recommended for safety margin

4. **Voltage Levels**:
   - Arduino logic: 5V
   - Servos: 5-6V (check servo specifications)
   - VL53L0X: 3.3V or 5V (check module)

## Emergency Stop

- **Hardware**: Connect normally-open switch between D8 and GND
- **Software**: Sends "stop" command via WebSocket
- **Behavior**: Detaches all servos immediately

## I2C Bus

- **VL53L0X** uses I2C (SDA/SCL)
- Default address: 0x29
- Pull-up resistors usually included on module
- If multiple I2C devices, check address conflicts

