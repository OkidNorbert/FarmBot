/*
 * AI Tomato Sorter - Arduino Firmware
 * Controls 3-DOF robotic arm for tomato sorting
 * 
 * Hardware:
 * - 3x Servo motors (SG90 or similar)
 * - Servo 1: Base rotation (Pin 3)
 * - Servo 2: Arm joint (Pin 5) 
 * - Servo 3: Gripper (Pin 6)
 * 
 * Communication:
 * - Serial commands from Raspberry Pi
 * - Commands: MOVE X Y CLASS, ANGLE A1 A2 A3, STOP
 */

#include <Servo.h>

// Servo objects
Servo servo1;  // Base rotation
Servo servo2;  // Arm joint
Servo servo3;  // Gripper

// Servo pins
const int SERVO1_PIN = 3;
const int SERVO2_PIN = 5;
const int SERVO3_PIN = 6;

// Servo limits (degrees)
const int SERVO1_MIN = 0;
const int SERVO1_MAX = 180;
const int SERVO2_MIN = 0;
const int SERVO2_MAX = 180;
const int SERVO3_MIN = 0;
const int SERVO3_MAX = 180;

// Current servo positions
int current_angle1 = 90;
int current_angle2 = 90;
int current_angle3 = 90;

// Movement parameters
const int MOVEMENT_DELAY = 50;  // Delay between servo movements (ms)
const int GRIPPER_OPEN = 0;
const int GRIPPER_CLOSE = 180;

// Safety parameters
const int MAX_MOVEMENT_SPEED = 5;  // Max degrees per step
bool emergency_stop = false;

// Sorting bins positions (servo angles)
const int BIN_NOT_READY_ANGLE1 = 0;
const int BIN_NOT_READY_ANGLE2 = 45;
const int BIN_READY_ANGLE1 = 90;
const int BIN_READY_ANGLE2 = 45;
const int BIN_SPOILT_ANGLE1 = 180;
const int BIN_SPOILT_ANGLE2 = 45;

// Arm dimensions (mm)
const float ARM_LENGTH1 = 100.0;  // First arm segment
const float ARM_LENGTH2 = 80.0;   // Second arm segment

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Tomato Sorter Arduino - Ready");
  
  // Attach servos
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  
  // Initialize servos to home position
  moveToHome();
  
  // Wait for servos to settle
  delay(1000);
  
  Serial.println("Servos initialized to home position");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      processCommand(command);
    }
  }
  
  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}

void processCommand(String command) {
  Serial.println("Received: " + command);
  
  if (command.startsWith("MOVE")) {
    // MOVE X Y CLASS - Move to world coordinates and sort by class
    processMoveCommand(command);
  }
  else if (command.startsWith("ANGLE")) {
    // ANGLE A1 A2 A3 - Set servo angles directly
    processAngleCommand(command);
  }
  else if (command.startsWith("STOP")) {
    // Emergency stop
    emergencyStop();
  }
  else if (command.startsWith("HOME")) {
    // Return to home position
    moveToHome();
  }
  else if (command.startsWith("GRIP")) {
    // GRIP OPEN/CLOSE - Control gripper
    processGripCommand(command);
  }
  else if (command.startsWith("STATUS")) {
    // Return current status
    sendStatus();
  }
  else {
    Serial.println("Unknown command: " + command);
  }
}

void processMoveCommand(String command) {
  // Parse MOVE X Y CLASS command
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ', firstSpace + 1);
  int thirdSpace = command.indexOf(' ', secondSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1) {
    Serial.println("Invalid MOVE command format");
    return;
  }
  
  float x = command.substring(firstSpace + 1, secondSpace).toFloat();
  float y = command.substring(secondSpace + 1, thirdSpace).toFloat();
  int class_id = command.substring(thirdSpace + 1).toInt();
  
  Serial.println("Moving to: X=" + String(x) + ", Y=" + String(y) + ", Class=" + String(class_id));
  
  // Convert world coordinates to servo angles using inverse kinematics
  float angle1, angle2;
  if (inverseKinematics(x, y, angle1, angle2)) {
    // Move to position
    moveToPosition(angle1, angle2);
    delay(500);  // Wait for movement to complete
    
    // Sort based on class
    sortTomato(class_id);
  } else {
    Serial.println("Position unreachable");
  }
}

void processAngleCommand(String command) {
  // Parse ANGLE A1 A2 A3 command
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ', firstSpace + 1);
  int thirdSpace = command.indexOf(' ', secondSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1) {
    Serial.println("Invalid ANGLE command format");
    return;
  }
  
  int angle1 = command.substring(firstSpace + 1, secondSpace).toInt();
  int angle2 = command.substring(secondSpace + 1, thirdSpace).toInt();
  int angle3 = command.substring(thirdSpace + 1).toInt();
  
  // Validate angles
  angle1 = constrain(angle1, SERVO1_MIN, SERVO1_MAX);
  angle2 = constrain(angle2, SERVO2_MIN, SERVO2_MAX);
  angle3 = constrain(angle3, SERVO3_MIN, SERVO3_MAX);
  
  // Move servos
  moveServos(angle1, angle2, angle3);
  
  Serial.println("Angles set: " + String(angle1) + ", " + String(angle2) + ", " + String(angle3));
}

void processGripCommand(String command) {
  // Parse GRIP OPEN/CLOSE command
  int spaceIndex = command.indexOf(' ');
  if (spaceIndex == -1) {
    Serial.println("Invalid GRIP command format");
    return;
  }
  
  String action = command.substring(spaceIndex + 1);
  action.trim();
  
  if (action.equals("OPEN")) {
    servo3.write(GRIPPER_OPEN);
    Serial.println("Gripper opened");
  }
  else if (action.equals("CLOSE")) {
    servo3.write(GRIPPER_CLOSE);
    Serial.println("Gripper closed");
  }
  else {
    Serial.println("Invalid gripper action: " + action);
  }
}

void emergencyStop() {
  emergency_stop = true;
  Serial.println("EMERGENCY STOP ACTIVATED");
  
  // Stop all movement immediately
  // In a real system, you might want to cut power to servos
  // or move to a safe position
}

void moveToHome() {
  Serial.println("Moving to home position");
  moveServos(90, 90, 90);  // Home position
  current_angle1 = 90;
  current_angle2 = 90;
  current_angle3 = 90;
  emergency_stop = false;
}

void moveServos(int angle1, int angle2, int angle3) {
  if (emergency_stop) {
    Serial.println("Movement blocked - emergency stop active");
    return;
  }
  
  // Smooth movement to prevent jerky motion
  smoothMove(servo1, current_angle1, angle1);
  smoothMove(servo2, current_angle2, angle2);
  smoothMove(servo3, current_angle3, angle3);
  
  current_angle1 = angle1;
  current_angle2 = angle2;
  current_angle3 = angle3;
}

void smoothMove(Servo &servo, int current_angle, int target_angle) {
  int steps = abs(target_angle - current_angle);
  int step_size = min(MAX_MOVEMENT_SPEED, steps);
  
  if (current_angle < target_angle) {
    for (int angle = current_angle; angle <= target_angle; angle += step_size) {
      if (emergency_stop) return;
      servo.write(angle);
      delay(MOVEMENT_DELAY);
    }
  } else {
    for (int angle = current_angle; angle >= target_angle; angle -= step_size) {
      if (emergency_stop) return;
      servo.write(angle);
      delay(MOVEMENT_DELAY);
    }
  }
  
  // Final position
  servo.write(target_angle);
  delay(MOVEMENT_DELAY);
}

void moveToPosition(float angle1, float angle2) {
  // Convert float angles to int and constrain
  int servo1_angle = constrain((int)angle1, SERVO1_MIN, SERVO1_MAX);
  int servo2_angle = constrain((int)angle2, SERVO2_MIN, SERVO2_MAX);
  
  // Move to position
  moveServos(servo1_angle, servo2_angle, current_angle3);
  
  Serial.println("Moved to position: " + String(servo1_angle) + ", " + String(servo2_angle));
}

void sortTomato(int class_id) {
  Serial.println("Sorting tomato - Class: " + String(class_id));
  
  // Close gripper to pick up tomato
  servo3.write(GRIPPER_CLOSE);
  delay(500);
  
  // Move to appropriate bin based on class
  switch (class_id) {
    case 0:  // Not ready
      moveServos(BIN_NOT_READY_ANGLE1, BIN_NOT_READY_ANGLE2, current_angle3);
      break;
    case 1:  // Ready
      moveServos(BIN_READY_ANGLE1, BIN_READY_ANGLE2, current_angle3);
      break;
    case 2:  // Spoilt
      moveServos(BIN_SPOILT_ANGLE1, BIN_SPOILT_ANGLE2, current_angle3);
      break;
    default:
      Serial.println("Unknown class: " + String(class_id));
      return;
  }
  
  delay(500);  // Wait for movement
  
  // Open gripper to drop tomato
  servo3.write(GRIPPER_OPEN);
  delay(500);
  
  // Return to home position
  moveToHome();
  
  Serial.println("Tomato sorted successfully");
}

bool inverseKinematics(float x, float y, float &angle1, float &angle2) {
  // Simple 2D inverse kinematics for 2-link arm
  // This is a simplified version - adjust based on your arm geometry
  
  float distance = sqrt(x*x + y*y);
  
  // Check if position is reachable
  if (distance > (ARM_LENGTH1 + ARM_LENGTH2) || distance < abs(ARM_LENGTH1 - ARM_LENGTH2)) {
    return false;
  }
  
  // Calculate angles using law of cosines
  float cos_angle2 = (ARM_LENGTH1*ARM_LENGTH1 + ARM_LENGTH2*ARM_LENGTH2 - distance*distance) / 
                     (2 * ARM_LENGTH1 * ARM_LENGTH2);
  
  if (cos_angle2 < -1 || cos_angle2 > 1) {
    return false;
  }
  
  angle2 = acos(cos_angle2);
  
  float beta = atan2(y, x);
  float alpha = acos((ARM_LENGTH1*ARM_LENGTH1 + distance*distance - ARM_LENGTH2*ARM_LENGTH2) / 
                     (2 * ARM_LENGTH1 * distance));
  
  angle1 = beta - alpha;
  
  // Convert to degrees
  angle1 = angle1 * 180.0 / PI;
  angle2 = angle2 * 180.0 / PI;
  
  // Adjust for servo mounting and coordinate system
  angle1 = angle1 + 90;  // Adjust based on your servo mounting
  
  return true;
}

void sendStatus() {
  Serial.println("=== STATUS ===");
  Serial.println("Emergency Stop: " + String(emergency_stop ? "ACTIVE" : "INACTIVE"));
  Serial.println("Current Angles: " + String(current_angle1) + ", " + String(current_angle2) + ", " + String(current_angle3));
  Serial.println("Servo 1 Range: " + String(SERVO1_MIN) + " - " + String(SERVO1_MAX));
  Serial.println("Servo 2 Range: " + String(SERVO2_MIN) + " - " + String(SERVO2_MAX));
  Serial.println("Servo 3 Range: " + String(SERVO3_MIN) + " - " + String(SERVO3_MAX));
  Serial.println("==============");
}

// Utility functions
void blinkLED(int pin, int times, int delay_ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, HIGH);
    delay(delay_ms);
    digitalWrite(pin, LOW);
    delay(delay_ms);
  }
}

void resetEmergencyStop() {
  emergency_stop = false;
  Serial.println("Emergency stop reset");
}
