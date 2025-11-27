/*
 * Tomato Sorter Robotic Arm Firmware
 * Board: Arduino R4 WiFi/Minima
 * 
 * Controls 6 Servos for a 6DOF Robotic Arm
 * VL53L0X Distance Sensor for precise depth measurement
 * Communicates via Serial (115200 baud)
 * 
 * Commands:
 * - MOVE <x> <y> <z> <class> : Move to coordinates (Inverse Kinematics or mapped)
 * - PICK <x> <y> <z> <class> : Execute pick sequence
 * - HOME                     : Move to home position
 * - GRIPPER <OPEN/CLOSE>     : Control gripper
 * - DISTANCE                  : Read distance from VL53L0X sensor (mm)
 * - STATUS                   : Report status
 * 
 * Wiring:
 * - Servos: Base(D3), Shoulder(D5), Elbow(D6), Wrist Vert(D9), Wrist Rot(D10), Gripper(D11)
 * - VL53L0X: SDA(A4), SCL(A5), VCC(3.3V), GND(GND)
 * - Power: Servos powered from external 5V 5A supply, Arduino via USB
 */

#include <Servo.h>
#include "Adafruit_VL53L0X.h"

// Servo Pin Definitions (Adjust based on your wiring)
#define PIN_BASE       3
#define PIN_SHOULDER   5
#define PIN_ELBOW      6
#define PIN_WRIST_VER  9
#define PIN_WRIST_ROT  10
#define PIN_GRIPPER    11

// Servo Objects
Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_ver;
Servo wrist_rot;
Servo gripper;

// VL53L0X Distance Sensor
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
bool distance_sensor_available = false;

// Home Positions (Angles 0-180)
const int HOME_BASE = 90;
const int HOME_SHOULDER = 90;
const int HOME_ELBOW = 90;
const int HOME_WRIST_VER = 90;
const int HOME_WRIST_ROT = 90;
const int HOME_GRIPPER = 0; // Open

// Bin Positions (Base Angles)
const int BIN_UNRIPE = 30;
const int BIN_RIPE = 150;

void setup() {
  Serial.begin(115200);
  
  // Initialize VL53L0X Distance Sensor
  if (!lox.begin()) {
    Serial.println("WARNING: VL53L0X sensor not found");
    distance_sensor_available = false;
  } else {
    Serial.println("VL53L0X sensor initialized");
    distance_sensor_available = true;
  }
  
  // Attach Servos
  base.attach(PIN_BASE);
  shoulder.attach(PIN_SHOULDER);
  elbow.attach(PIN_ELBOW);
  wrist_ver.attach(PIN_WRIST_VER);
  wrist_rot.attach(PIN_WRIST_ROT);
  gripper.attach(PIN_GRIPPER);
  
  // Move to Home
  moveHome();
  
  Serial.println("STATUS: READY");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }
}

void processCommand(String cmd) {
  if (cmd.startsWith("HOME")) {
    moveHome();
    Serial.println("OK: HOME");
  }
  else if (cmd.startsWith("GRIPPER")) {
    if (cmd.indexOf("OPEN") > 0) {
      gripper.write(0);
      Serial.println("OK: GRIPPER OPEN");
    } else {
      gripper.write(180); // Closed
      Serial.println("OK: GRIPPER CLOSED");
    }
  }
  else if (cmd.startsWith("PICK")) {
    // Format: PICK <x> <y> <z> <class_id>
    // For now, we'll implement a simplified pick sequence based on class ID
    // In a real implementation, you'd use IK to move to x,y,z
    
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    int thirdSpace = cmd.indexOf(' ', secondSpace + 1);
    int fourthSpace = cmd.indexOf(' ', thirdSpace + 1);
    
    if (fourthSpace > 0) {
      int classId = cmd.substring(fourthSpace + 1).toInt();
      executePickSequence(classId);
      Serial.println("OK: PICK COMPLETE");
    } else {
      Serial.println("ERROR: INVALID ARGS");
    }
  }
  else if (cmd.startsWith("DISTANCE")) {
    if (distance_sensor_available) {
      VL53L0X_RangingMeasurementData_t measure;
      lox.rangingTest(&measure, false);
      
      if (measure.RangeStatus != 4) {  // Valid reading
        Serial.print("DISTANCE: ");
        Serial.println(measure.RangeMilliMeter);
      } else {
        Serial.println("DISTANCE: OUT_OF_RANGE");
      }
    } else {
      Serial.println("DISTANCE: SENSOR_NOT_AVAILABLE");
    }
  }
  else if (cmd.startsWith("STATUS")) {
    Serial.print("STATUS: READY");
    if (distance_sensor_available) {
      Serial.println(" | VL53L0X: OK");
    } else {
      Serial.println(" | VL53L0X: NOT_FOUND");
    }
  }
}

void moveHome() {
  base.write(HOME_BASE);
  delay(200);
  shoulder.write(HOME_SHOULDER);
  delay(200);
  elbow.write(HOME_ELBOW);
  delay(200);
  wrist_ver.write(HOME_WRIST_VER);
  delay(200);
  wrist_rot.write(HOME_WRIST_ROT);
  delay(200);
  gripper.write(HOME_GRIPPER);
}

void executePickSequence(int classId) {
  // 1. Move to Pick Position (Simplified: Center)
  base.write(90);
  shoulder.write(60); // Lower arm
  elbow.write(60);
  delay(1000);
  
  // 2. Measure distance and adjust wrist height if sensor available
  if (distance_sensor_available) {
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);
    
    if (measure.RangeStatus != 4 && measure.RangeMilliMeter > 0) {
      int distance_mm = measure.RangeMilliMeter;
      // Adjust wrist angle based on distance (closer = lower wrist)
      // Adjust these values based on your arm geometry
      // Map: 20mm (close) -> 60° (low), 200mm (far) -> 120° (high)
      int wrist_angle = map(constrain(distance_mm, 20, 200), 20, 200, 60, 120);
      wrist_ver.write(wrist_angle);
      Serial.print("Distance: ");
      Serial.print(distance_mm);
      Serial.print("mm, Wrist angle: ");
      Serial.println(wrist_angle);
      delay(300);
    }
  }
  
  // 3. Close Gripper
  gripper.write(180);
  delay(500);
  
  // 4. Lift
  shoulder.write(90);
  elbow.write(90);
  wrist_ver.write(90); // Reset wrist
  delay(1000);
  
  // 5. Move to Bin
  if (classId == 1) { // Ripe
    base.write(BIN_RIPE);
  } else { // Unripe
    base.write(BIN_UNRIPE);
  }
  delay(1000);
  
  // 6. Drop
  gripper.write(0);
  delay(500);
  
  // 7. Return Home
  moveHome();
}
