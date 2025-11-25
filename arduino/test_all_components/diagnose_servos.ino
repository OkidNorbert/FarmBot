/*
 * Servo Diagnostic Test
 * =====================
 * 
 * Tests each servo individually to identify wiring/power issues
 * Sets initial home position FIRST, then tests one servo at a time
 */

#include <Servo.h>

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

// Home Position (Safe initial position)
const int HOME_BASE = 90;
const int HOME_SHOULDER = 90;
const int HOME_ELBOW = 90;
const int HOME_WRIST_VER = 90;
const int HOME_WRIST_ROT = 90;
const int HOME_GRIPPER = 0;  // Open

bool initialized = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) delay(10);
  
  Serial.println("\n");
  Serial.println("========================================");
  Serial.println("  SERVO DIAGNOSTIC TEST");
  Serial.println("========================================");
  Serial.println();
  
  // STEP 1: Initialize all servos and set home position FIRST
  Serial.println(">>> STEP 1: Initializing all servos...");
  initializeAllServos();
  
  Serial.println();
  Serial.println(">>> STEP 2: Moving to safe home position...");
  moveToHome();
  delay(2000);
  
  Serial.println();
  Serial.println(">>> Arm is now in home position");
  Serial.println(">>> Ready to test servos one at a time");
  Serial.println();
  Serial.println("Press any key in Serial Monitor to start testing...");
  Serial.println();
  
  initialized = true;
}

void loop() {
  if (!initialized) {
    return;
  }
  
  // Wait for user input to start testing
  if (Serial.available() > 0) {
    Serial.readString(); // Clear input buffer
    
    Serial.println("\n========================================");
    Serial.println("  STARTING INDIVIDUAL SERVO TESTS");
    Serial.println("========================================");
    Serial.println();
    Serial.println("Each servo will be tested ONE AT A TIME");
    Serial.println("Watch carefully which servos move!");
    Serial.println();
    delay(2000);
    
    // Test each servo individually
    Serial.println("\n>>> Testing Base Servo (D3) ONLY...");
    testServoIndividually(base, PIN_BASE, "Base", HOME_BASE);
    delay(2000);
    
    Serial.println("\n>>> Testing Shoulder Servo (D5) ONLY...");
    testServoIndividually(shoulder, PIN_SHOULDER, "Shoulder", HOME_SHOULDER);
    delay(2000);
    
    Serial.println("\n>>> Testing Elbow Servo (D6) ONLY...");
    testServoIndividually(elbow, PIN_ELBOW, "Elbow", HOME_ELBOW);
    delay(2000);
    
    Serial.println("\n>>> Testing Wrist Vert Servo (D9) ONLY...");
    testServoIndividually(wrist_ver, PIN_WRIST_VER, "Wrist Vert", HOME_WRIST_VER);
    delay(2000);
    
    Serial.println("\n>>> Testing Wrist Rot Servo (D10) ONLY...");
    testServoIndividually(wrist_rot, PIN_WRIST_ROT, "Wrist Rot", HOME_WRIST_ROT);
    delay(2000);
    
    Serial.println("\n>>> Testing Gripper Servo (D11) ONLY...");
    testServoIndividually(gripper, PIN_GRIPPER, "Gripper", HOME_GRIPPER);
    delay(2000);
    
    // Return all to home
    Serial.println("\n>>> Returning all servos to home position...");
    moveToHome();
    delay(2000);
    
    Serial.println("\n========================================");
    Serial.println("  DIAGNOSTIC COMPLETE");
    Serial.println("========================================");
    Serial.println();
    Serial.println("All servos returned to home position");
    Serial.println();
    Serial.println("Press any key to run tests again...");
    Serial.println();
  }
  
  delay(100);
}

void initializeAllServos() {
  Serial.println("  Attaching all servos...");
  
  base.attach(PIN_BASE);
  Serial.print("    Base (D3): ");
  Serial.println(base.attached() ? "OK" : "FAILED");
  delay(200);
  
  shoulder.attach(PIN_SHOULDER);
  Serial.print("    Shoulder (D5): ");
  Serial.println(shoulder.attached() ? "OK" : "FAILED");
  delay(200);
  
  elbow.attach(PIN_ELBOW);
  Serial.print("    Elbow (D6): ");
  Serial.println(elbow.attached() ? "OK" : "FAILED");
  delay(200);
  
  wrist_ver.attach(PIN_WRIST_VER);
  Serial.print("    Wrist Vert (D9): ");
  Serial.println(wrist_ver.attached() ? "OK" : "FAILED");
  delay(200);
  
  wrist_rot.attach(PIN_WRIST_ROT);
  Serial.print("    Wrist Rot (D10): ");
  Serial.println(wrist_rot.attached() ? "OK" : "FAILED");
  delay(200);
  
  gripper.attach(PIN_GRIPPER);
  Serial.print("    Gripper (D11): ");
  Serial.println(gripper.attached() ? "OK" : "FAILED");
  delay(200);
  
  Serial.println("  All servos attached");
}

void moveToHome() {
  Serial.println("  Moving to home position (90° for joints, 0° for gripper)...");
  
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
  gripper.write(HOME_GRIPPER);
  delay(500);
  
  Serial.println("  Home position set");
}

void testServoIndividually(Servo &servo, int pin, const char* name, int homeAngle) {
  Serial.print("  Testing ");
  Serial.print(name);
  Serial.print(" on pin D");
  Serial.println(pin);
  Serial.println("  (Other servos should NOT move)");
  delay(1000);
  
  // Test sequence: home -> 0° -> 90° -> 180° -> 90° -> home
  Serial.println("    Moving to 0°...");
  servo.write(0);
  delay(1500);
  
  Serial.println("    Moving to 90°...");
  servo.write(90);
  delay(1500);
  
  Serial.println("    Moving to 180°...");
  servo.write(180);
  delay(1500);
  
  Serial.println("    Returning to 90°...");
  servo.write(90);
  delay(1500);
  
  Serial.println("    Returning to home position...");
  servo.write(homeAngle);
  delay(1500);
  
  Serial.print("    ");
  Serial.print(name);
  Serial.println(" test complete - returned to home");
  Serial.println();
}

