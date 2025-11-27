/*
 * AI Tomato Sorter - Arduino Firmware
 * Controls 5-DOF robotic arm for tomato sorting
 * 
 * Hardware:
 * - 6x Servo motors
 *   Servo 1: Base rotation (Pin 7)
 *   Servo 2: Shoulder/Forearm (Pin 6)
 *   Servo 3: Elbow joint (Pin 5)
 *   Servo 4: Wrist Yaw (Pin 4)
 *   Servo 5: Wrist Pitch (Pin 3)
 *   Servo 6: Gripper (Pin 2)
 * - VL53L0X Time-of-Flight Distance Sensor (SDA A4, SCL A5)
 * 
 * Communication:
 * - Serial commands from Raspberry Pi
 * - Commands: MOVE X Y CLASS, ANGLE A1 A2 A3 A4 A5 A6, STOP
 */

#include <Servo.h>
#include <Wire.h>
#include "Adafruit_VL53L0X.h"
#include <ArduinoBLE.h>

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

// BLE Service and Characteristic
BLEService farmBotService("19B10000-E8F2-537E-4F6C-D104768A1214"); // Custom UUID
BLEStringCharacteristic commandCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite, 50);


// Servo indices for readability
enum ServoIndex {
  SERVO_BASE = 0,
  SERVO_SHOULDER = 1,
  SERVO_ELBOW = 2,
  SERVO_WRIST_YAW = 3,
  SERVO_WRIST_PITCH = 4,
  SERVO_GRIPPER = 5,
  SERVO_COUNT = 6
};

// Servo objects
Servo servos[SERVO_COUNT];

// Servo pins (Base, Shoulder, Elbow, WristYaw, WristPitch, Gripper)
const int SERVO_PINS[SERVO_COUNT] = {7, 6, 5, 4, 3, 2};

// Servo limits (degrees)
const int SERVO_MIN[SERVO_COUNT] = {0, 10, 0, 0, 0, 20};
const int SERVO_MAX[SERVO_COUNT] = {180, 170, 180, 180, 180, 160};

// Movement parameters
const int MOVEMENT_DELAY = 40;  // Delay between servo movements (ms)
const int MAX_MOVEMENT_SPEED = 5;  // Max degrees per step
const int GRIPPER_OPEN = 30;
const int GRIPPER_CLOSE = 150;

// Wrist presets
// Wrist presets
const int WRIST_PITCH_NEUTRAL = 90;
const int WRIST_PITCH_PICK = 110;
const int WRIST_PITCH_BIN = 80;
const int WRIST_YAW_NEUTRAL = 90;

// Current servo positions
int current_angles[SERVO_COUNT] = {90, 90, 90, WRIST_YAW_NEUTRAL, WRIST_PITCH_NEUTRAL, GRIPPER_OPEN};
int target_angles[SERVO_COUNT]  = {90, 90, 90, WRIST_YAW_NEUTRAL, WRIST_PITCH_NEUTRAL, GRIPPER_OPEN};

bool emergency_stop = false;

// Sorting bins positions (servo angles)
struct ServoPose {
  int base;
  int shoulder;
  int elbow;
  int wrist_yaw;
  int wrist_pitch;
};

const ServoPose BIN_NOT_READY_POSE = {20, 55, 120, WRIST_YAW_NEUTRAL, WRIST_PITCH_BIN};
const ServoPose BIN_READY_POSE     = {100, 50, 110, WRIST_YAW_NEUTRAL, WRIST_PITCH_BIN};
const ServoPose BIN_SPOILT_POSE    = {160, 60, 115, WRIST_YAW_NEUTRAL, WRIST_PITCH_BIN};

ServoPose getBinPose(int class_id);
void performPickAndSort(float baseAngle, float shoulderAngle, float elbowAngle, int class_id);

// Arm dimensions (mm)
const float ARM_LENGTH1 = 100.0;  // First arm segment
const float ARM_LENGTH2 = 80.0;   // Second arm segment

// VL53L0X Sensor
// No specific pins needed as it uses I2C (SDA, SCL)

// Pick height configuration
const float MIN_PICK_DISTANCE_CM = 6.0;
const float MAX_PICK_DISTANCE_CM = 25.0;
const int WRIST_MIN_PICK_ANGLE = 70;
const int WRIST_MAX_PICK_ANGLE = 130;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Tomato Sorter Arduino - Ready");
  
  // Attach servos
  for (int i = 0; i < SERVO_COUNT; i++) {
    servos[i].attach(SERVO_PINS[i]);
  }

  // Initialize VL53L0X
  if (!lox.begin()) {
    Serial.println(F("Failed to boot VL53L0X"));
    // Continue anyway, but distance sensing won't work
  } else {
    Serial.println(F("VL53L0X ready"));
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("FarmBot");
  BLE.setAdvertisedService(farmBotService);
  farmBotService.addCharacteristic(commandCharacteristic);
  BLE.addService(farmBotService);
  commandCharacteristic.writeValue("");

  BLE.advertise();
  Serial.println("BLE FarmBot Ready");
  
  // Initialize servos to home position
  moveToHome();
  
  // Wait for servos to settle
  delay(1000);
  
  Serial.println("Servos initialized to home position");
}

void loop() {
  // Poll for BLE events
  BLE.poll();

  // Check for BLE commands
  if (commandCharacteristic.written()) {
    String command = commandCharacteristic.value();
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }

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
  else if (command.startsWith("PICK")) {
    // PICK X Y Z CLASS - Pick from coordinates with depth and sort by class
    processPickCommand(command);
  }
  else if (command.startsWith("ANGLE")) {
    // ANGLE A1 A2 A3 A4 A5 A6 - Set servo angles directly (-1 to keep current)
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
  else if (command.startsWith("GRIPPER")) {
    // GRIPPER OPEN/CLOSE - Alternative gripper command
    processGripCommand(command);
  }
  else if (command.startsWith("STATUS")) {
    // Return current status
    sendStatus();
  }
  else if (command.startsWith("DISTANCE")) {
    // Return distance sensor reading
    sendDistance();
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
  float baseAngle, shoulderAngle, elbowAngle;
  if (inverseKinematics(x, y, baseAngle, shoulderAngle, elbowAngle)) {
    performPickAndSort(baseAngle, shoulderAngle, elbowAngle, class_id);
  } else {
    Serial.println("Position unreachable");
  }
}

void processPickCommand(String command) {
  // Parse PICK X Y Z CLASS command
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ', firstSpace + 1);
  int thirdSpace = command.indexOf(' ', secondSpace + 1);
  int fourthSpace = command.indexOf(' ', thirdSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1 || fourthSpace == -1) {
    Serial.println("Invalid PICK command format");
    return;
  }
  
  float x = command.substring(firstSpace + 1, secondSpace).toFloat();
  float y = command.substring(secondSpace + 1, thirdSpace).toFloat();
  float z = command.substring(thirdSpace + 1, fourthSpace).toFloat();
  int class_id = command.substring(fourthSpace + 1).toInt();
  
  Serial.println("Picking at: X=" + String(x) + ", Y=" + String(y) + ", Z=" + String(z) + ", Class=" + String(class_id));
  
  // Convert world coordinates to servo angles using inverse kinematics
  float baseAngle, shoulderAngle, elbowAngle;
  if (inverseKinematics(x, y, baseAngle, shoulderAngle, elbowAngle)) {
    performPickAndSort(baseAngle, shoulderAngle, elbowAngle, class_id);
  } else {
    Serial.println("Position unreachable");
  }
}

void sendDistance() {
  // Read distance sensor and send response
  float distance = measureDistanceCm();
  
  if (distance < 0) {
    Serial.println("DISTANCE: OUT_OF_RANGE");
  } else if (distance > 400) {
    Serial.println("DISTANCE: OUT_OF_RANGE");
  } else {
    // Convert cm to mm
    int distance_mm = (int)(distance * 10);
    Serial.println("DISTANCE: " + String(distance_mm));
  }
}

void processAngleCommand(String command) {
  // ANGLE A1 A2 A3 A4 A5 (-1 to keep current angle)
  int parsedAngles[SERVO_COUNT];
  for (int i = 0; i < SERVO_COUNT; i++) {
    parsedAngles[i] = current_angles[i];
  }

  int startIdx = command.indexOf(' ');
  if (startIdx == -1) {
    Serial.println("Invalid ANGLE command format");
    return;
  }
  command = command.substring(startIdx + 1);

  for (int i = 0; i < SERVO_COUNT; i++) {
    int spaceIdx = command.indexOf(' ');
    String token;
    if (spaceIdx == -1) {
      token = command;
    } else {
      token = command.substring(0, spaceIdx);
    }

    if (token.length() == 0) {
      Serial.println("Invalid ANGLE command parameters");
      return;
    }

    int val = token.toInt();
    if (val >= 0) {
      parsedAngles[i] = constrain(val, SERVO_MIN[i], SERVO_MAX[i]);
    }

    if (spaceIdx == -1) {
      if (i < SERVO_COUNT - 1) {
        Serial.println("Not enough ANGLE parameters provided");
        return;
      }
      break;
    }
    command = command.substring(spaceIdx + 1);
  }

  moveServos(parsedAngles);

  Serial.print("Angles set: ");
  for (int i = 0; i < SERVO_COUNT; i++) {
    Serial.print(parsedAngles[i]);
    if (i < SERVO_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
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
    setGripper(true);
    Serial.println("Gripper opened");
  }
  else if (action.equals("CLOSE")) {
    setGripper(false);
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
  int homeAngles[SERVO_COUNT] = {90, 90, 90, WRIST_YAW_NEUTRAL, WRIST_PITCH_NEUTRAL, GRIPPER_OPEN};
  moveServos(homeAngles);
  emergency_stop = false;
}

void setGripper(bool open) {
  target_angles[SERVO_GRIPPER] = constrain(open ? GRIPPER_OPEN : GRIPPER_CLOSE,
                                           SERVO_MIN[SERVO_GRIPPER],
                                           SERVO_MAX[SERVO_GRIPPER]);
  smoothMoveServo(SERVO_GRIPPER, target_angles[SERVO_GRIPPER]);
  current_angles[SERVO_GRIPPER] = target_angles[SERVO_GRIPPER];
}

void smoothMoveServo(int servoIndex, int targetAngle) {
  // Safety check: validate servo index
  if (servoIndex < 0 || servoIndex >= SERVO_COUNT) {
    Serial.println("ERROR: Invalid servo index");
    return;
  }
  
  int currentAngle = current_angles[servoIndex];
  int constrainedTarget = constrain(targetAngle, SERVO_MIN[servoIndex], SERVO_MAX[servoIndex]);

  // Safety check: prevent extreme movements
  int maxSingleMove = 90; // Max degrees in one command
  if (abs(constrainedTarget - currentAngle) > maxSingleMove) {
    Serial.println("WARNING: Large movement requested, limiting to " + String(maxSingleMove) + " degrees");
    if (constrainedTarget > currentAngle) {
      constrainedTarget = currentAngle + maxSingleMove;
    } else {
      constrainedTarget = currentAngle - maxSingleMove;
    }
  }

  if (currentAngle == constrainedTarget) {
    return;
  }

  int stepDirection = (constrainedTarget > currentAngle) ? 1 : -1;
  int remaining = abs(constrainedTarget - currentAngle);
  int stepSize = min(MAX_MOVEMENT_SPEED, remaining);

  int angle = currentAngle;
  while (angle != constrainedTarget) {
    // Emergency stop check
    if (emergency_stop) {
      Serial.println("Movement stopped - emergency stop active");
      return;
    }
    
    angle += stepDirection * stepSize;
    if ((stepDirection > 0 && angle > constrainedTarget) ||
        (stepDirection < 0 && angle < constrainedTarget)) {
      angle = constrainedTarget;
    }
    
    // Double-check bounds
    angle = constrain(angle, SERVO_MIN[servoIndex], SERVO_MAX[servoIndex]);
    servos[servoIndex].write(angle);
    delay(MOVEMENT_DELAY);
  }
  delay(MOVEMENT_DELAY);
}

void moveServos(const int desiredAngles[SERVO_COUNT]) {
  if (emergency_stop) {
    Serial.println("Movement blocked - emergency stop active");
    return;
  }

  for (int i = 0; i < SERVO_COUNT; i++) {
    target_angles[i] = constrain(desiredAngles[i], SERVO_MIN[i], SERVO_MAX[i]);
  }

  // Move base/shoulder/elbow first for positioning
  smoothMoveServo(SERVO_BASE, target_angles[SERVO_BASE]);
  current_angles[SERVO_BASE] = target_angles[SERVO_BASE];

  smoothMoveServo(SERVO_SHOULDER, target_angles[SERVO_SHOULDER]);
  current_angles[SERVO_SHOULDER] = target_angles[SERVO_SHOULDER];

  smoothMoveServo(SERVO_ELBOW, target_angles[SERVO_ELBOW]);
  current_angles[SERVO_ELBOW] = target_angles[SERVO_ELBOW];

  // Wrist adjustments
  smoothMoveServo(SERVO_WRIST_YAW, target_angles[SERVO_WRIST_YAW]);
  current_angles[SERVO_WRIST_YAW] = target_angles[SERVO_WRIST_YAW];

  smoothMoveServo(SERVO_WRIST_PITCH, target_angles[SERVO_WRIST_PITCH]);
  current_angles[SERVO_WRIST_PITCH] = target_angles[SERVO_WRIST_PITCH];

  // Gripper moves only when explicitly requested
  if (target_angles[SERVO_GRIPPER] != current_angles[SERVO_GRIPPER]) {
    smoothMoveServo(SERVO_GRIPPER, target_angles[SERVO_GRIPPER]);
    current_angles[SERVO_GRIPPER] = target_angles[SERVO_GRIPPER];
  }
}

void moveToPose(const ServoPose &pose, int gripperAngle = -1) {
  int desired[SERVO_COUNT] = {
    pose.base,
    pose.shoulder,
    pose.elbow,
    pose.wrist_yaw,
    pose.wrist_pitch,
    (gripperAngle >= 0) ? gripperAngle : current_angles[SERVO_GRIPPER]
  };
  moveServos(desired);
}

void setWristPitchAngle(int angle) {
  int constrained = constrain(angle, SERVO_MIN[SERVO_WRIST_PITCH], SERVO_MAX[SERVO_WRIST_PITCH]);
  target_angles[SERVO_WRIST_PITCH] = constrained;
  smoothMoveServo(SERVO_WRIST_PITCH, constrained);
  current_angles[SERVO_WRIST_PITCH] = constrained;
}

float measureDistanceCm() {
  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false); 

  if (measure.RangeStatus != 4) {  // phase failures have incorrect data
    return measure.RangeMilliMeter / 10.0;
  } else {
    return -1.0;
  }
}

int computeWristAngleForDistance(float distanceCm) {
  if (distanceCm < 0) {
    return WRIST_PITCH_PICK;
  }
  float clamped = constrain(distanceCm, MIN_PICK_DISTANCE_CM, MAX_PICK_DISTANCE_CM);
  long scaled = (long)(clamped * 10.0);  // improve precision for map
  long minScaled = (long)(MIN_PICK_DISTANCE_CM * 10.0);
  long maxScaled = (long)(MAX_PICK_DISTANCE_CM * 10.0);
  long mapped = map(scaled, minScaled, maxScaled,
                    (long)WRIST_MAX_PICK_ANGLE, (long)WRIST_MIN_PICK_ANGLE);
  return constrain((int)mapped, WRIST_MIN_PICK_ANGLE, WRIST_MAX_PICK_ANGLE);
}

ServoPose getBinPose(int class_id) {
  switch (class_id) {
    case 0:
      return BIN_NOT_READY_POSE;
    case 1:
      return BIN_READY_POSE;
    case 2:
      return BIN_SPOILT_POSE;
    default:
      return BIN_READY_POSE;
  }
}

void performPickAndSort(float baseAngle, float shoulderAngle, float elbowAngle, int class_id) {
  Serial.println("Beginning pick sequence");

  // Approach tomato
  int approach[SERVO_COUNT] = {
    constrain((int)baseAngle, SERVO_MIN[SERVO_BASE], SERVO_MAX[SERVO_BASE]),
    constrain((int)shoulderAngle, SERVO_MIN[SERVO_SHOULDER], SERVO_MAX[SERVO_SHOULDER]),
    constrain((int)elbowAngle, SERVO_MIN[SERVO_ELBOW], SERVO_MAX[SERVO_ELBOW]),
    WRIST_YAW_NEUTRAL,
    WRIST_PITCH_PICK,
    GRIPPER_OPEN
  };

  moveServos(approach);
  delay(300);

  // Measure distance and adjust wrist height
  float distance = measureDistanceCm();
  if (distance > 0) {
    int wristAngle = computeWristAngleForDistance(distance);
    Serial.print("Distance detected: ");
    Serial.print(distance);
    Serial.print(" cm -> Wrist angle ");
    Serial.println(wristAngle);
    setWristPitchAngle(wristAngle);
  } else {
    Serial.println("Distance measurement failed, using default wrist angle");
    setWristPitchAngle(WRIST_PITCH_PICK);
  }
  delay(200);

  // Close gripper to pick tomato
  setGripper(false);
  delay(500);

  // Move to bin
  ServoPose binPose = getBinPose(class_id);
  moveToPose(binPose, current_angles[SERVO_GRIPPER]);
  delay(500);

  // Release tomato
  setWristPitchAngle(binPose.wrist_pitch);
  delay(150);
  setGripper(true);
  delay(400);

  // Return to home
  moveToHome();

  Serial.println("Tomato sorted successfully");
}

void moveToPosition(float baseAngle, float shoulderAngle, float elbowAngle, float wristAngle) {
  int desired[SERVO_COUNT] = {
    constrain((int)baseAngle, SERVO_MIN[SERVO_BASE], SERVO_MAX[SERVO_BASE]),
    constrain((int)shoulderAngle, SERVO_MIN[SERVO_SHOULDER], SERVO_MAX[SERVO_SHOULDER]),
    constrain((int)elbowAngle, SERVO_MIN[SERVO_ELBOW], SERVO_MAX[SERVO_ELBOW]),
    WRIST_YAW_NEUTRAL,
    constrain((int)wristAngle, SERVO_MIN[SERVO_WRIST_PITCH], SERVO_MAX[SERVO_WRIST_PITCH]),
    current_angles[SERVO_GRIPPER]
  };

  moveServos(desired);

  Serial.print("Moved to position: ");
  Serial.print(desired[SERVO_BASE]);
  Serial.print(", ");
  Serial.print(desired[SERVO_SHOULDER]);
  Serial.print(", ");
  Serial.print(desired[SERVO_ELBOW]);
  Serial.print(", wrist pitch ");
  Serial.println(desired[SERVO_WRIST_PITCH]);
}


bool inverseKinematics(float x, float y, float &baseAngle, float &shoulderAngle, float &elbowAngle) {
  // Convert target to polar for base rotation
  float planarDistance = sqrt(x * x + y * y);

  // Base rotation assumes x-axis is horizontal offset, y-axis is forward reach
  float baseRadians = atan2(x, y);  // rotate around vertical axis
  baseAngle = (baseRadians * 180.0 / PI) + 90.0; // convert to servo range

  // Check reachability for planar 2-link arm
  if (planarDistance > (ARM_LENGTH1 + ARM_LENGTH2) || planarDistance < abs(ARM_LENGTH1 - ARM_LENGTH2)) {
    return false;
  }

  float cosElbow = (ARM_LENGTH1 * ARM_LENGTH1 + ARM_LENGTH2 * ARM_LENGTH2 - planarDistance * planarDistance) /
                   (2 * ARM_LENGTH1 * ARM_LENGTH2);
  cosElbow = constrain(cosElbow, -1.0f, 1.0f);

  float elbowRadians = acos(cosElbow);
  elbowAngle = 180.0 - (elbowRadians * 180.0 / PI);  // convert for servo orientation

  float cosShoulder = (ARM_LENGTH1 * ARM_LENGTH1 + planarDistance * planarDistance - ARM_LENGTH2 * ARM_LENGTH2) /
                      (2 * ARM_LENGTH1 * planarDistance);
  cosShoulder = constrain(cosShoulder, -1.0f, 1.0f);

  float shoulderOffset = acos(cosShoulder);
  float planarAngle = atan2(y, x);
  float shoulderRadians = planarAngle - shoulderOffset;
  shoulderAngle = (shoulderRadians * 180.0 / PI) + 90.0;

  return true;
}

void sendStatus() {
  Serial.println("=== STATUS ===");
  Serial.println("Emergency Stop: " + String(emergency_stop ? "ACTIVE" : "INACTIVE"));
  Serial.print("Current Angles: ");
  for (int i = 0; i < SERVO_COUNT; i++) {
    Serial.print(current_angles[i]);
    if (i < SERVO_COUNT - 1) Serial.print(", ");
  }
  Serial.println();
  for (int i = 0; i < SERVO_COUNT; i++) {
    Serial.print("Servo ");
    Serial.print(i + 1);
    Serial.print(" Range: ");
    Serial.print(SERVO_MIN[i]);
    Serial.print(" - ");
    Serial.println(SERVO_MAX[i]);
  }
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
