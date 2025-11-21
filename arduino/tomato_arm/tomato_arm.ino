/*
 * Tomato Sorter Robotic Arm Firmware
 * Board: Arduino R4 / Uno
 * 
 * Controls 6 Servos for a 6DOF Robotic Arm
 * Communicates via Serial (115200 baud)
 * 
 * Commands:
 * - MOVE <x> <y> <z> <class> : Move to coordinates (Inverse Kinematics or mapped)
 * - PICK <x> <y> <z> <class> : Execute pick sequence
 * - HOME                     : Move to home position
 * - GRIPPER <OPEN/CLOSE>     : Control gripper
 * - STATUS                   : Report status
 */

#include <Servo.h>

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
  else if (cmd.startsWith("STATUS")) {
    Serial.println("STATUS: READY");
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
  
  // 2. Close Gripper
  gripper.write(180);
  delay(500);
  
  // 3. Lift
  shoulder.write(90);
  elbow.write(90);
  delay(1000);
  
  // 4. Move to Bin
  if (classId == 1) { // Ripe
    base.write(BIN_RIPE);
  } else { // Unripe
    base.write(BIN_UNRIPE);
  }
  delay(1000);
  
  // 5. Drop
  gripper.write(0);
  delay(500);
  
  // 6. Return Home
  moveHome();
}
