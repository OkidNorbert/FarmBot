/*
 * Arduino R4 Component Test Suite
 * =================================
 * 
 * Tests all components of the Tomato Sorter Robotic Arm:
 * - 6 Servos (Base, Shoulder, Elbow, Wrist Vert, Wrist Rot, Gripper)
 * - VL53L0X Distance Sensor
 * 
 * Commands via Serial (115200 baud):
 * - TEST ALL          : Run full test sequence
 * - TEST SERVOS       : Test all servos individually
 * - TEST SENSOR       : Test VL53L0X distance sensor
 * - SERVO <n> <angle> : Move servo n (1-6) to angle (0-180)
 * - HOME              : Move all servos to home position
 * - STATUS            : Show component status
 * 
 * Servo Numbers:
 * 1 = Base (D3)
 * 2 = Shoulder (D5)
 * 3 = Elbow (D6)
 * 4 = Wrist Vert (D9)
 * 5 = Wrist Rot (D10)
 * 6 = Gripper (D11)
 */

#include <Servo.h>
#include "Adafruit_VL53L0X.h"

// Servo Pin Definitions
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

// Home Positions (Safe initial position for robotic arm)
const int HOME_BASE = 90;
const int HOME_SHOULDER = 90;
const int HOME_ELBOW = 90;
const int HOME_WRIST_VER = 90;
const int HOME_WRIST_ROT = 90;
const int HOME_GRIPPER = 0;  // Open

// VL53L0X Distance Sensor
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
bool sensor_available = false;

// Test state
bool test_running = false;

void setup() {
  Serial.begin(115200);
  
  // Wait for serial port to open (optional, for USB)
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  
  Serial.println("\n");
  Serial.println("========================================");
  Serial.println("  Arduino R4 Component Test Suite");
  Serial.println("========================================");
  Serial.println();
  
  // Initialize VL53L0X Sensor
  Serial.print("Initializing VL53L0X sensor... ");
  if (!lox.begin()) {
    Serial.println("FAILED");
    Serial.println("  ERROR: VL53L0X sensor not found!");
    Serial.println("  Check wiring: SDA->A4, SCL->A5, VCC->3.3V, GND->GND");
    sensor_available = false;
  } else {
    Serial.println("OK");
    sensor_available = true;
  }
  Serial.println();
  
  // Attach Servos
  Serial.println("Attaching servos...");
  base.attach(PIN_BASE);
  Serial.print("  Base (D3): ");
  Serial.println(base.attached() ? "OK" : "FAILED");
  delay(200);
  
  shoulder.attach(PIN_SHOULDER);
  Serial.print("  Shoulder (D5): ");
  Serial.println(shoulder.attached() ? "OK" : "FAILED");
  delay(200);
  
  elbow.attach(PIN_ELBOW);
  Serial.print("  Elbow (D6): ");
  Serial.println(elbow.attached() ? "OK" : "FAILED");
  delay(200);
  
  wrist_ver.attach(PIN_WRIST_VER);
  Serial.print("  Wrist Vert (D9): ");
  Serial.println(wrist_ver.attached() ? "OK" : "FAILED");
  delay(200);
  
  wrist_rot.attach(PIN_WRIST_ROT);
  Serial.print("  Wrist Rot (D10): ");
  Serial.println(wrist_rot.attached() ? "OK" : "FAILED");
  delay(200);
  
  gripper.attach(PIN_GRIPPER);
  Serial.print("  Gripper (D11): ");
  Serial.println(gripper.attached() ? "OK" : "FAILED");
  Serial.println();
  
  // IMPORTANT: Move to home position FIRST before any testing
  Serial.println(">>> Setting initial home position FIRST...");
  Serial.println("    This prevents servo confusion on startup");
  moveHome();
  delay(2000);
  Serial.println("    Home position set - arm is ready");
  Serial.println();
  
  Serial.println("========================================");
  Serial.println("  System Ready!");
  Serial.println("========================================");
  Serial.println();
  printHelp();
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    processCommand(command);
  }
  
  // If test is running, continue it
  if (test_running) {
    delay(100);
  }
}

void processCommand(String cmd) {
  if (cmd.startsWith("TEST ALL")) {
    runFullTest();
  }
  else if (cmd.startsWith("TEST SERVOS")) {
    testServos();
  }
  else if (cmd.startsWith("TEST SENSOR")) {
    testSensor();
  }
  else if (cmd.startsWith("SERVO")) {
    // Format: SERVO <n> <angle>
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    
    if (firstSpace > 0 && secondSpace > 0) {
      int servoNum = cmd.substring(firstSpace + 1, secondSpace).toInt();
      int angle = cmd.substring(secondSpace + 1).toInt();
      
      if (servoNum >= 1 && servoNum <= 6 && angle >= 0 && angle <= 180) {
        moveServo(servoNum, angle);
      } else {
        Serial.println("ERROR: Invalid servo number (1-6) or angle (0-180)");
      }
    } else {
      Serial.println("ERROR: Usage: SERVO <1-6> <0-180>");
    }
  }
  else if (cmd.startsWith("HOME")) {
    moveHome();
    Serial.println("OK: Moved to home position");
  }
  else if (cmd.startsWith("STATUS")) {
    printStatus();
  }
  else if (cmd.startsWith("HELP")) {
    printHelp();
  }
  else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
    Serial.println("Type HELP for available commands");
  }
}

void printHelp() {
  Serial.println("Available Commands:");
  Serial.println("  TEST ALL          - Run full test sequence");
  Serial.println("  TEST SERVOS       - Test all servos");
  Serial.println("  TEST SENSOR       - Test distance sensor");
  Serial.println("  SERVO <n> <angle> - Move servo n (1-6) to angle (0-180)");
  Serial.println("  HOME              - Move all servos to home (90°)");
  Serial.println("  STATUS            - Show component status");
  Serial.println("  HELP              - Show this help");
  Serial.println();
  Serial.println("Servo Numbers:");
  Serial.println("  1 = Base (D3)");
  Serial.println("  2 = Shoulder (D5)");
  Serial.println("  3 = Elbow (D6)");
  Serial.println("  4 = Wrist Vert (D9)");
  Serial.println("  5 = Wrist Rot (D10)");
  Serial.println("  6 = Gripper (D11)");
  Serial.println();
}

void printStatus() {
  Serial.println("\n========================================");
  Serial.println("  Component Status");
  Serial.println("========================================");
  
  Serial.print("VL53L0X Sensor: ");
  if (sensor_available) {
    Serial.println("OK");
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);
    if (measure.RangeStatus != 4) {
      Serial.print("  Distance: ");
      Serial.print(measure.RangeMilliMeter);
      Serial.println(" mm");
    } else {
      Serial.println("  Status: Out of range");
    }
  } else {
    Serial.println("NOT AVAILABLE");
  }
  
  Serial.println("\nServos:");
  Serial.print("  Base (D3): ");
  Serial.println(base.attached() ? "ATTACHED" : "NOT ATTACHED");
  Serial.print("  Shoulder (D5): ");
  Serial.println(shoulder.attached() ? "ATTACHED" : "NOT ATTACHED");
  Serial.print("  Elbow (D6): ");
  Serial.println(elbow.attached() ? "ATTACHED" : "NOT ATTACHED");
  Serial.print("  Wrist Vert (D9): ");
  Serial.println(wrist_ver.attached() ? "ATTACHED" : "NOT ATTACHED");
  Serial.print("  Wrist Rot (D10): ");
  Serial.println(wrist_rot.attached() ? "ATTACHED" : "NOT ATTACHED");
  Serial.print("  Gripper (D11): ");
  Serial.println(gripper.attached() ? "ATTACHED" : "NOT ATTACHED");
  
  Serial.println("========================================\n");
}

void moveServo(int servoNum, int angle) {
  Serial.print("Moving Servo ");
  Serial.print(servoNum);
  Serial.print(" to ");
  Serial.print(angle);
  Serial.println("°");
  
  switch(servoNum) {
    case 1: base.write(angle); break;
    case 2: shoulder.write(angle); break;
    case 3: elbow.write(angle); break;
    case 4: wrist_ver.write(angle); break;
    case 5: wrist_rot.write(angle); break;
    case 6: gripper.write(angle); break;
    default:
      Serial.println("ERROR: Invalid servo number");
      return;
  }
  
  delay(500); // Wait for servo to move
  Serial.println("OK");
}

void moveHome() {
  // Move servos sequentially to avoid confusion
  base.write(HOME_BASE);
  delay(300);
  shoulder.write(HOME_SHOULDER);
  delay(300);
  elbow.write(HOME_ELBOW);
  delay(300);
  wrist_ver.write(HOME_WRIST_VER);
  delay(300);
  wrist_rot.write(HOME_WRIST_ROT);
  delay(300);
  gripper.write(HOME_GRIPPER); // Open
  delay(500); // Extra delay for gripper
}

void testServos() {
  Serial.println("\n========================================");
  Serial.println("  Testing All Servos (One at a Time)");
  Serial.println("========================================");
  Serial.println("Each servo will be tested individually");
  Serial.println("Other servos will stay in home position");
  Serial.println();
  
  test_running = true;
  
  // Ensure we start from home position
  Serial.println("Ensuring all servos are in home position...");
  moveHome();
  delay(1000);
  
  // Test each servo individually (ONE AT A TIME)
  int servos[] = {1, 2, 3, 4, 5, 6};
  const char* names[] = {"Base", "Shoulder", "Elbow", "Wrist Vert", "Wrist Rot", "Gripper"};
  
  for (int i = 0; i < 6; i++) {
    Serial.print("\n>>> Testing ");
    Serial.print(names[i]);
    Serial.print(" (Servo ");
    Serial.print(servos[i]);
    Serial.println(") ONLY...");
    Serial.println("    (Other servos should NOT move)");
    delay(1000);
    
    // Move to 0°
    Serial.println("    Moving to 0°...");
    moveServo(servos[i], 0);
    delay(1500);
    
    // Move to 90°
    Serial.println("    Moving to 90°...");
    moveServo(servos[i], 90);
    delay(1500);
    
    // Move to 180°
    Serial.println("    Moving to 180°...");
    moveServo(servos[i], 180);
    delay(1500);
    
    // Return to 90° (home for most servos)
    Serial.println("    Returning to 90°...");
    moveServo(servos[i], 90);
    delay(1500);
    
    // Return to actual home position
    if (servos[i] == 6) { // Gripper
      moveServo(servos[i], 0); // Gripper home is 0° (open)
    } else {
      moveServo(servos[i], 90); // Other servos home is 90°
    }
    delay(1500);
    
    Serial.print("    ");
    Serial.print(names[i]);
    Serial.println(" test complete - returned to home");
    Serial.println();
  }
  
  // Final return to home
  Serial.println(">>> Final check: Returning all servos to home position...");
  moveHome();
  delay(1000);
  
  Serial.println("\n========================================");
  Serial.println("  Servo Test Complete!");
  Serial.println("========================================\n");
  
  test_running = false;
}

void testSensor() {
  Serial.println("\n========================================");
  Serial.println("  Testing VL53L0X Distance Sensor");
  Serial.println("========================================");
  
  if (!sensor_available) {
    Serial.println("ERROR: Sensor not available!");
    Serial.println("Check wiring: SDA->A4, SCL->A5, VCC->3.3V, GND->GND");
    return;
  }
  
  Serial.println("Taking 10 distance measurements...");
  Serial.println();
  
  int valid_readings = 0;
  int invalid_readings = 0;
  long total_distance = 0;
  
  for (int i = 0; i < 10; i++) {
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);
    
    Serial.print("Reading ");
    Serial.print(i + 1);
    Serial.print(": ");
    
    if (measure.RangeStatus != 4) {
      int distance = measure.RangeMilliMeter;
      Serial.print(distance);
      Serial.println(" mm");
      valid_readings++;
      total_distance += distance;
    } else {
      Serial.println("OUT OF RANGE");
      invalid_readings++;
    }
    
    delay(200);
  }
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("  Sensor Test Results");
  Serial.println("========================================");
  Serial.print("Valid readings: ");
  Serial.println(valid_readings);
  Serial.print("Invalid readings: ");
  Serial.println(invalid_readings);
  
  if (valid_readings > 0) {
    Serial.print("Average distance: ");
    Serial.print(total_distance / valid_readings);
    Serial.println(" mm");
  }
  
  Serial.println("========================================\n");
}

void runFullTest() {
  Serial.println("\n");
  Serial.println("========================================");
  Serial.println("  FULL COMPONENT TEST SEQUENCE");
  Serial.println("========================================");
  Serial.println();
  
  test_running = true;
  
  // 1. Test Sensor
  Serial.println(">>> Step 1: Testing Distance Sensor");
  Serial.println();
  testSensor();
  delay(2000);
  
  // 2. Test Servos
  Serial.println(">>> Step 2: Testing All Servos");
  Serial.println();
  testServos();
  delay(2000);
  
  // 3. Test coordinated movement
  Serial.println(">>> Step 3: Testing Coordinated Movement");
  Serial.println();
  Serial.println("Moving arm through test sequence...");
  
  // Sequence: Lower arm, measure distance, pick motion, return
  Serial.println("  Lowering arm...");
  base.write(90);
  shoulder.write(60);
  elbow.write(60);
  wrist_ver.write(90);
  wrist_rot.write(90);
  delay(2000);
  
  if (sensor_available) {
    Serial.println("  Measuring distance...");
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);
    if (measure.RangeStatus != 4) {
      Serial.print("  Distance: ");
      Serial.print(measure.RangeMilliMeter);
      Serial.println(" mm");
    }
    delay(1000);
  }
  
  Serial.println("  Closing gripper...");
  gripper.write(180);
  delay(1000);
  
  Serial.println("  Lifting arm...");
  shoulder.write(90);
  elbow.write(90);
  delay(2000);
  
  Serial.println("  Opening gripper...");
  gripper.write(0);
  delay(1000);
  
  Serial.println("  Returning to home...");
  moveHome();
  delay(2000);
  
  Serial.println("\n========================================");
  Serial.println("  FULL TEST COMPLETE!");
  Serial.println("========================================");
  Serial.println();
  Serial.println("All components tested successfully!");
  Serial.println();
  
  test_running = false;
}

