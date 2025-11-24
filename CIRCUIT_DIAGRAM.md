# Final Circuit Diagram - AI Tomato Sorter

This document details the wiring for the complete system, including the Raspberry Pi, Arduino R4, 6DOF Arm, and Sensors.

## System Overview

```mermaid
graph TD
    subgraph Power_Supply
        PSU[5V 5A External Power Supply]
    end

    subgraph Raspberry_Pi_4
        Pi[Raspberry Pi 4]
        Cam[USB Camera]
    end

    subgraph Arduino_System
        R4[Arduino R4 WiFi/Minima]
        
        subgraph Servos
            S1[Base Servo (Pin 3)]
            S2[Shoulder Servo (Pin 5)]
            S3[Elbow Servo (Pin 6)]
            S4[Wrist Ver Servo (Pin 9)]
            S5[Wrist Rot Servo (Pin 10)]
            S6[Gripper Servo (Pin 11)]
        end
        
        subgraph Sensors
            TOF[VL53L0X Distance Sensor]
        end
    end

    %% Connections
    PSU -->|VCC (5V)| S1 & S2 & S3 & S4 & S5 & S6
    PSU -->|GND| S1 & S2 & S3 & S4 & S5 & S6
    PSU -->|GND| R4
    PSU -->|Common GND| Pi
    
    Pi -->|USB Data| R4
    Pi -->|USB| Cam
    
    R4 -->|PWM Pin 3| S1
    R4 -->|PWM Pin 5| S2
    R4 -->|PWM Pin 6| S3
    R4 -->|PWM Pin 9| S4
    R4 -->|PWM Pin 10| S5
    R4 -->|PWM Pin 11| S6
    
    R4 -->|I2C SDA (A4)| TOF
    R4 -->|I2C SCL (A5)| TOF
    R4 -->|3.3V| TOF
    R4 -->|GND| TOF
```

## Pinout Table

### Arduino R4 Connections

| Component | Arduino Pin | Description |
| :--- | :--- | :--- |
| **Base Servo** | Digital 3 (~) | PWM Signal |
| **Shoulder Servo** | Digital 5 (~) | PWM Signal |
| **Elbow Servo** | Digital 6 (~) | PWM Signal |
| **Wrist Ver Servo** | Digital 9 (~) | PWM Signal |
| **Wrist Rot Servo** | Digital 10 (~) | PWM Signal |
| **Gripper Servo** | Digital 11 (~) | PWM Signal |
| **VL53L0X SDA** | SDA (or A4) | I2C Data |
| **VL53L0X SCL** | SCL (or A5) | I2C Clock |
| **VL53L0X VCC** | 3.3V | Power (Check sensor voltage!) |
| **VL53L0X GND** | GND | Ground |

### Power Connections (CRITICAL)

> [!WARNING]
> **DO NOT power the servos directly from the Arduino 5V pin.**
> The 6 servos draw too much current and will reset or damage the Arduino.

1.  **External Power Supply**: Use a 5V 5A Power Supply for the servos.
2.  **Servo Power**: Connect all Servo **Red (+) wires** directly to the PSU Positive (5V).
3.  **Servo Ground**: Connect all Servo **Brown/Black (-) wires** directly to the PSU Negative (GND).
4.  **Common Ground**: Connect the PSU Negative (GND) to:
    - Arduino **GND** pin (required for PWM signals to work)
    - Raspberry Pi **GND** (for common reference)
5.  **Arduino Power**: Power the Arduino via the USB cable connected to the Raspberry Pi (USB Data connection).

## Raspberry Pi Connections

| Component | Connection | Description |
| :--- | :--- | :--- |
| **Arduino R4** | USB Port | Serial Communication & Power |
| **USB Camera** | USB Port | Video Feed |
| **Power** | USB-C | Official Pi Power Supply |

## Connection Summary

### Power Distribution
- **5V External Power Supply (5A)**: Powers all 6 servos directly (red wires to 5V, black/brown wires to GND)
- **Arduino R4**: Powered via USB from Raspberry Pi
- **Raspberry Pi**: Powered via USB-C with official Pi power supply
- **VL53L0X Sensor**: Powered from Arduino 3.3V pin

### Data/Control Connections
- **Raspberry Pi ↔ Arduino**: USB cable for serial communication
- **Raspberry Pi ↔ USB Camera**: USB connection for video feed
- **Arduino ↔ Servos**: PWM signals on pins D3, D5, D6, D9, D10, D11
- **Arduino ↔ VL53L0X**: I2C communication (SDA on A4, SCL on A5)

## Notes
- **VL53L0X**: Most modules work on 3.3V. If your module has a voltage regulator, 5V might be okay, but 3.3V is safer for the I2C lines on some boards. The Arduino R4 is 5V logic but often tolerant. Use the 3.3V output from Arduino to power the sensor.
- **Common Ground**: All components must share a common ground reference. Connect the external PSU GND to both Arduino GND and Raspberry Pi GND.
- **Cable Management**: Ensure servo cables are long enough for the arm to move fully without pulling them out.
- **Power Isolation**: Servos are powered directly from the external 5V supply, NOT from the Arduino 5V pin, to prevent brownouts and resets.
