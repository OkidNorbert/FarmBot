# Embedded System Components (Bill of Materials)

This file lists all embedded/robotics hardware required for the AI Tomato Sorter (5‑DOF arm + ultrasonic sensing), with suggested quantities and brief notes.

## Core Compute
- Raspberry Pi 4 (4GB+) or Pi 5 (1)
- Official Raspberry Pi PSU (Pi 4: 5V 3A, Pi 5: 5V 5A) (1)
- microSD card 32–128GB, Class 10 (1)
- HDMI cable + monitor/keyboard (initial setup, optional)

## Vision
- Camera: Raspberry Pi Camera Module (v2/HQ) with ribbon OR USB webcam (1)
- Camera mount/bracket (1)

## Motion Control
- Arduino Uno or Nano (1)
- Servos (5 total):
  - Base: SG90/MG90S (1)
  - Shoulder: MG90S (metal gear recommended) (1)
  - Elbow: MG90S (metal gear recommended) (1)
  - Wrist: MG90S (metal gear recommended) (1)
  - Gripper: SG90/MG90S (1)
- Servo gripper mechanism (servo‑driven) (1)
- External 5V DC power supply for servos (1): 5V 2–5A (choose by torque/load)
- Large electrolytic capacitor 1000–2200 µF, ≥10V across servo 5V/GND (1)
- 0.1 µF ceramic decoupling capacitors near servo connectors (3–5)
- Inline fuse holder + 2–5A fuse for servo 5V rail (1 set, recommended)
- Main DC toggle switch for servo PSU (1, recommended)
- USB cable Pi↔Arduino (A–B or A–Micro/Mini as applicable) (1)

## Sensors
- HC‑SR04 Ultrasonic Distance Sensor (1)  (TRIG→D11, ECHO→D12)
- Optional: VL53L0X ToF distance sensor (1) for higher accuracy at short range
- Optional: Limit switches (2–4) for homing/safety

## Wiring & Prototyping
- Breadboard or screw terminal blocks (1)
- Male–male and male–female Dupont jumper wires (assorted)
- Servo extension leads (as needed)
- Crimp kit + housings (optional, for robust connectors)
- Heat‑shrink tubing / electrical tape, zip ties

## Mechanical
- 5‑DOF robotic arm frame (metal/acrylic/3D‑printed) with brackets & servo horns
- Fastener kit: M2/M3 screws, nuts, standoffs (assorted)
- Base plate and mounting hardware (1)
- Sorting bins/containers (3)

## Power & Conversion (Optional/As Needed)
- 5V buck converter (if stepping down from higher voltage) (1)
- Powered USB hub (if using multiple USB devices on Pi) (1)

## Safety & Tools
- Emergency stop switch (1)
- Multimeter (1)
- Soldering iron + solder (optional but useful)

## Networking (Optional)
- Ethernet cable (1) or reliable Wi‑Fi

## Pin Allocation (Reference)
- Arduino PWM servo pins: D3 (Base), D5 (Shoulder), D6 (Elbow), D9 (Wrist), D10 (Gripper)
- HC‑SR04: TRIG→D11, ECHO→D12
- Power: Servos on dedicated 5V PSU; tie servo GND to Arduino GND and Raspberry Pi GND (common ground is mandatory)

## Notes
- Prefer MG90S (metal gear) for joints bearing load (shoulder/elbow/wrist). SG90 is acceptable for base/gripper on light builds.
- Do NOT power servos from Arduino 5V. Use a dedicated 5V PSU sized for peak stall currents; add bulk capacitance near the servo rail.
- If you later need more stable timing or more channels, consider a PCA9685 16‑ch servo driver (optional).


