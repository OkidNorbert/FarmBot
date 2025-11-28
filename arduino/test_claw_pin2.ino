/*
 * Claw/Gripper Servo Test - Pin 2
 * SG90 Micro Servo Test
 * Claw opens and closes within safe limits (0-90°)
 * 
 * SG90 Specifications:
 * - Operating voltage: 4.8V - 6.0V
 * - Rotation range: ~180 degrees (0-180°)
 * - Torque: 1.8kg/cm @ 4.8V
 * - Speed: 0.1 sec/60° @ 4.8V
 * - Weight: 9g
 * 
 * Claw Behavior:
 * - CLOSED Position: 90 degrees
 * - OPEN Position: 0 degrees
 * - Range: 0° to 90° (NEVER EXCEED 90°)
 * - Movement: Only moves between 0° (open) and 90° (closed)
 * 
 * Safety Limits:
 * - MIN: 0 degrees (hard limit - open position)
 * - MAX: 90 degrees (hard limit - closed position, NEVER EXCEED)
 * - Power-On Position: 90 degrees (closed/safe position)
 * 
 * IMPORTANT - Power-On Safety and Stop Method:
 * - Power-On Position: 90 degrees (moved to IMMEDIATELY on boot - closed/safe)
 * - This is the ONLY reference position - no assumptions about claw position
 * - Stop Sequence:
 *   1. Send 90 degrees (closed/safe position)
 *   2. Detach servo (clawServo.detach())
 *   3. Set pin to INPUT mode (pinMode(CLAW_PIN, INPUT))
 * - Power-On Sequence:
 *   1. Set pin to OUTPUT
 *   2. Attach servo
 *   3. IMMEDIATELY write 90° (closed position, no delay before this!)
 * - This prevents unwanted movement during initialization
 */

#include <Servo.h>

const int CLAW_PIN = 2;        // Claw/Gripper servo on pin 2
const int MIN_ANGLE = 0;      // Minimum safe angle (open position)
const int MAX_ANGLE = 90;      // Maximum safe angle (closed position - NEVER EXCEED)
const int CLAW_CLOSED = 90;    // Claw closed position
const int CLAW_OPEN = 0;       // Claw open position

Servo clawServo;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  
  Serial.println("========================================");
  Serial.println("SG90 Claw/Gripper Servo Test");
  Serial.println("Pin: 2");
  Serial.println("========================================");
  Serial.print("Safe Range: ");
  Serial.print(MIN_ANGLE);
  Serial.print(" - ");
  Serial.print(MAX_ANGLE);
  Serial.println(" degrees");
  Serial.print("Closed Position: ");
  Serial.print(CLAW_CLOSED);
  Serial.println(" degrees");
  Serial.print("Open Position: ");
  Serial.print(CLAW_OPEN);
  Serial.println(" degrees");
  Serial.println("Movement: 90° (closed) ↔ 0° (open)");
  Serial.println("");
  
  // CRITICAL: Set servo to closed position (90°) IMMEDIATELY on power-on
  // This is the ONLY position we assume - no other assumptions about claw position
  Serial.println("Setting pin to OUTPUT mode...");
  pinMode(CLAW_PIN, OUTPUT);
  
  Serial.print("Attaching servo and moving to closed position (");
  Serial.print(CLAW_CLOSED);
  Serial.println("°) IMMEDIATELY...");
  clawServo.attach(CLAW_PIN);
  // Write 90 degrees IMMEDIATELY - no delay before this!
  clawServo.write(CLAW_CLOSED);  // Closed/safe position
  delay(1000);  // Give servo time to reach 90 degrees
  
  Serial.print("Claw positioned at closed position: ");
  Serial.print(CLAW_CLOSED);
  Serial.println(" degrees");
  delay(500);
  
  // Test opening: 90° → 0° (closed to open)
  Serial.println("Opening claw: 90° → 0° (closed to open)...");
  for (int angle = CLAW_CLOSED; angle >= CLAW_OPEN; angle -= 1) {
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    clawServo.write(safeAngle);
    delay(15);  // Smooth movement (adjust for speed: lower = faster)
  }
  Serial.print("Claw OPEN at ");
  Serial.print(CLAW_OPEN);
  Serial.println(" degrees");
  delay(500);
  
  // Test closing: 0° → 90° (open to closed)
  Serial.println("Closing claw: 0° → 90° (open to closed)...");
  for (int angle = CLAW_OPEN; angle <= CLAW_CLOSED; angle += 1) {
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    clawServo.write(safeAngle);
    delay(15);  // Smooth movement
  }
  Serial.print("Claw CLOSED at ");
  Serial.print(CLAW_CLOSED);
  Serial.println(" degrees");
  delay(500);
  
  // Repeat open/close cycle
  Serial.println("Repeating open/close cycle...");
  for (int cycle = 0; cycle < 3; cycle++) {
    Serial.print("Cycle ");
    Serial.print(cycle + 1);
    Serial.println(": Opening...");
    for (int angle = CLAW_CLOSED; angle >= CLAW_OPEN; angle -= 1) {
      int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
      clawServo.write(safeAngle);
      delay(15);
    }
    delay(500);
    
    Serial.print("Cycle ");
    Serial.print(cycle + 1);
    Serial.println(": Closing...");
    for (int angle = CLAW_OPEN; angle <= CLAW_CLOSED; angle += 1) {
      int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
      clawServo.write(safeAngle);
      delay(15);
    }
    delay(500);
  }
  
  // Stop the servo - send closed position (90 degrees)
  Serial.print("Stopping servo at closed position (");
  Serial.print(CLAW_CLOSED);
  Serial.println(" degrees)...");
  clawServo.write(CLAW_CLOSED);  // Closed/stop position
  delay(1000);
  
  // Detach servo to completely stop sending signals
  Serial.println("Detaching servo to stop all signals...");
  clawServo.detach();
  
  // Set pin to INPUT to ensure no signal
  pinMode(CLAW_PIN, INPUT);
  
  Serial.println("");
  Serial.println("========================================");
  Serial.println("Claw servo test complete!");
  Serial.print("Servo stopped at closed position (");
  Serial.print(CLAW_CLOSED);
  Serial.println(" degrees)");
  Serial.print("Range tested: ");
  Serial.print(CLAW_OPEN);
  Serial.print(" (open) - ");
  Serial.print(CLAW_CLOSED);
  Serial.println(" (closed) degrees");
  Serial.println("========================================");
}

void loop() {
  // Do nothing - rotation complete
  // Servo is detached and will not receive any signals
  delay(1000);
}

