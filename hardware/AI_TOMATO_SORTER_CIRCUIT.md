# AI Tomato Sorter - Full Circuit Diagram (Text Schematic + Pinout)

This document describes the complete wiring for the specified components:

- Raspberry Pi 4 (4GB) or Raspberry Pi 5 + Hailo AI Hat
- Raspberry Pi Power Supply (5V 3A for Pi 4 / 5V 5A for Pi 5)
- microSD Card 32–128GB
- FT5330M 35kg Digital RC Servo Motors ×5 (Base, Shoulder, Elbow, Wrist, Gripper)
- Raspberry Pi Camera (Pi Cam v2 / HQ) or USB Camera
- Arduino Uno or Nano (USB connection to Pi)
- 5V DC Power Supply (5–8A) for servos
- HC-SR04 Ultrasonic Sensor ×1
- VL53L0X ToF Sensor ×1
- Limit Switches ×3 (home/safety)
- Breadboard / Terminal Block, Jumper Wires, Servo Extensions

> Important: 35kg servos have high stall current. A single 5–8A supply may be insufficient under simultaneous high load. If you see brownouts, upgrade to 5V 10–15A and keep wires short/thick (≥18AWG) on the servo power rail.

---

## 1) System Block Diagram (ASCII)

```
           ┌─────────────────────────┐        CSI (Pi Cam) / USB (Webcam)
           │     Raspberry Pi 4/5    │◄───────────────────────────┐
           │  + Hailo AI Hat (on Pi) │                            │
           │                         │                            │
USB (data) │  USB ► Arduino (Uno/Nano)                            │
           │                         │                            │
           └───────────┬─────────────┘                            │
                       │ USB 5V (logic power)                     │
                       ▼                                          │
               ┌───────────────┐                                  │
               │   Arduino     │                                  │
               │  Uno / Nano   │                                  │
               │               │                                  │
               │ D3  → Servo1 (Base)                              │
               │ D5  → Servo2 (Shoulder)                          │
               │ D6  → Servo3 (Elbow)                             │
               │ D9  → Servo4 (Wrist)                             │
               │ D10 → Servo5 (Gripper)                           │
               │ D11 → HC-SR04 TRIG                               │
               │ D12 ← HC-SR04 ECHO                               │
               │ A4  ↔ VL53L0X SDA                                │
               │ A5  ↔ VL53L0X SCL                                │
               │ D2  ← Limit Switch 1 (INT, pull-up)              │
               │ D4  ← Limit Switch 2 (pull-up)                   │
               │ D7  ← Limit Switch 3 (pull-up)                   │
               │ 5V,3V3,Vin,GND                                   │
               └───────┬────────┘
                       │ GND (common reference)
  Common GND ┌─────────┴────────────┐
             ▼                      ▼
      ┌───────────────┐       ┌───────────────┐
      │ 5V 5–8A (10A+)│  +5V  │ Servo Power   │  (distribution: terminal block)
      │   PSU         ├──────►│ Rail +        │─► five FT5330M Servos (+, red)
      │ (servos only) │  GND  │ Rail GND      │─► five FT5330M Servos (−, black)
      └─────┬─────────┘       └──────┬────────┘
            │  Fuse 2–5A              │  Bulk cap 1000–2200µF across +/− near servos
            └──────────────────────────┴─────────────> GND tied to Arduino GND & Pi GND
```

Notes:
- The Raspberry Pi powers the Arduino via USB (logic only). Do NOT power servos from Arduino 5V or Pi 5V.
- Servo 5V rail is powered from the dedicated high-current PSU. Tie GND of servo PSU, Arduino GND, and Pi GND together (common ground).
- Keep high-current servo wiring separate from Pi/Arduino wiring; star-ground at the servo power distribution point.

---

## 2) Power Topology

- Raspberry Pi: official PSU (5V 3A for Pi 4 or 5V 5A for Pi 5). Hailo Hat is powered via Pi.
- Servos (FT5330M ×5): Dedicated 5V PSU (5–8A minimum; 10–15A recommended for safety margin). Power ONLY servo red/black from this PSU.
- Arduino: via USB from Pi (recommended) or external 5V; if external, ensure GND is common with servo PSU and Pi.
- Decoupling: 1000–2200µF electrolytic across +5V/GND at the servo rail plus 0.1µF ceramic near each servo connector.
- Protection: Inline 2–5A fuse from the servo PSU to the power rail; consider main toggle switch on servo rail.

---

## 3) Arduino Signal Pin Mapping

Servos (PWM outputs):
- D3  → Servo 1 (Base, FT5330M) signal (yellow/white)
- D5  → Servo 2 (Shoulder) signal
- D6  → Servo 3 (Elbow) signal
- D9  → Servo 4 (Wrist) signal
- D10 → Servo 5 (Gripper) signal

Sensors:
- HC-SR04: VCC → Arduino 5V, GND → Arduino GND, TRIG → D11, ECHO → D12 (Arduino Uno ECHO is 5V tolerant)
- VL53L0X ToF: Use module rated for 5V input (with onboard regulator/level shifting) if possible.  
  - VCC → 5V (or 3.3V if your module requires it)  
  - GND → Arduino GND  
  - SDA → A4, SCL → A5 (I²C)  
  - If using multiple VL53L0X, set unique I²C addresses.

Limit Switches (Normally Closed recommended for fail-safe):
- D2  ← Switch 1 NC to GND (use INPUT_PULLUP in firmware)  
- D4  ← Switch 2 NC to GND (use INPUT_PULLUP)  
- D7  ← Switch 3 NC to GND (use INPUT_PULLUP)

Other Connections:
- Arduino GND must connect to Servo PSU GND and Pi GND (common ground).
- Pi ↔ Arduino: USB cable for data and Arduino 5V logic power.
- Camera: CSI ribbon to Pi (for Pi Cam) or USB (for USB camera). Ensure Hailo Hat/CSI stacking clearance.

---

## 4) Servo Power Rail Wiring

- Use a terminal block or power distribution board: route PSU +5V to servo + (red) leads and PSU GND to servo − (black) leads.
- Wire gauge: ≥18AWG for PSU→rail; servo pigtails are typically smaller—keep lengths short.
- Place a 1000–2200µF cap across +5V/GND at the rail to mitigate inrush and brownouts.
- Route servo signal wires (from Arduino) away from high-current power lines if possible.

---

## 5) Hailo AI Hat Considerations

- Ensure mechanical stacking doesn’t block the CSI camera connector/cable.
- Hailo Hat uses the Pi’s 5V/3V3 rails—no extra wiring needed here.
- Avoid drawing servo power from the Pi 5V. Keep the servo PSU electrically separate except for GND.

---

## 6) Suggested Grounding Strategy

```
         [Servo PSU GND]────┐
                            ├───┐ Star point on servo rail / terminal block
 [Arduino GND]──────────────┘   │
                                └────[Raspberry Pi GND]
```

- One star ground point reduces ground bounce. Keep high-current returns localized.

---

## 7) Minimal Text Schematic (Netlist-style)

Power:
- PSU_Servo +5V → Servo_Rail +5V
- PSU_Servo GND → Servo_Rail GND → Arduino GND → Pi GND (common)
- Pi PSU → Pi 5V (official adapter)
- Pi USB → Arduino 5V (logic), GND

Servos (x5):
- ServoX +5V → Servo_Rail +5V
- ServoX GND → Servo_Rail GND
- Servo1 SIG → D3
- Servo2 SIG → D5
- Servo3 SIG → D6
- Servo4 SIG → D9
- Servo5 SIG → D10

HC-SR04:
- VCC → Arduino 5V
- GND → Arduino GND
- TRIG → D11
- ECHO → D12

VL53L0X (5V module recommended):
- VCC → Arduino 5V (or 3.3V per module spec)
- GND → Arduino GND
- SDA → A4
- SCL → A5

Limit Switches (NC):
- SW1: D2 ↔ NC; COM ↔ GND; Arduino pin mode INPUT_PULLUP
- SW2: D4 ↔ NC; COM ↔ GND; INPUT_PULLUP
- SW3: D7 ↔ NC; COM ↔ GND; INPUT_PULLUP

Camera:
- Pi Cam: CSI ribbon to Pi CSI port (ensure Hailo Hat compatibility/spacing)  
  or USB Cam: to Pi USB port.

---

## 8) Practical Tips

- Test each joint independently with low-speed motion first; confirm current draw and PSU stability.
- If any brownouts reset the Arduino/Pi, increase bulk capacitance and/or PSU rating, and verify ground topology.
- Keep servo wires short; heavy joints (shoulder/elbow/wrist) benefit from MG/metal-gear servos and robust brackets.
- Add ferrules or crimped ends for terminal blocks to avoid stray strands.
- Label cables and keep a wiring map near the rig for maintenance.


