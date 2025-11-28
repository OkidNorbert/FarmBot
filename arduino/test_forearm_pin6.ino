/*
 * Forearm/Shoulder Servo Test - Pin 6
 * Tower Pro MG996R Servo Test
 * Rotates servo within safe limits (0-180°) and stops
 * 
 * MG996R Specifications:
 * - Operating voltage: 4.8V - 7.2V
 * - Rotation range: ~180 degrees (0-180°)
 * - Torque: 10kg/cm @ 4.8V, 11kg/cm @ 6V
 * - Speed: 0.2 sec/60° @ 4.8V
 * 
 * Safety Limits:
 * - MIN: 0 degrees (hard limit)
 * - MAX: 180 degrees (hard limit - NEVER EXCEED)
 * - Initial/Reference Position: 90 degrees (set on power-on)
 * 
 * Movement Logic:
 * - All movements are relative to 90° (power-on position)
 * - Moving LEFT (negative): 90° - offset (minimum 0°)
 * - Moving RIGHT (positive): 90° + offset (maximum 180°)
 * 
 * IMPORTANT - Power-On Safety and Stop Method:
 * - Power-On Position: 90 degrees (moved to IMMEDIATELY on boot)
 * - This is the ONLY reference position - no assumptions about arm position
 * - Stop Sequence:
 *   1. Send 90 degrees (power-on/reference position)
 *   2. Detach servo (forearmServo.detach())
 *   3. Set pin to INPUT mode (pinMode(FOREARM_PIN, INPUT))
 * - Power-On Sequence:
 *   1. Set pin to OUTPUT
 *   2. Attach servo
 *   3. IMMEDIATELY write 90° (reference position, no delay before this!)
 * - This prevents unwanted movement during initialization
 */

#include <Servo.h>

const int FOREARM_PIN = 6;  // Forearm/Shoulder servo on pin 6
const int MIN_ANGLE = 0;     // Minimum safe angle
const int MAX_ANGLE = 180;   // Maximum safe angle - NEVER EXCEED
const int REFERENCE_ANGLE = 90; // Reference position (set on power-on)

Servo forearmServo;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  
  Serial.println("========================================");
  Serial.println("Tower Pro MG996R Forearm Servo Test");
  Serial.println("Pin: 6");
  Serial.println("========================================");
  Serial.print("Safe Range: ");
  Serial.print(MIN_ANGLE);
  Serial.print(" - ");
  Serial.print(MAX_ANGLE);
  Serial.println(" degrees");
  Serial.print("Reference Position: ");
  Serial.print(REFERENCE_ANGLE);
  Serial.println(" degrees (set on power-on)");
  Serial.println("Movement: Relative to reference position (90°)");
  Serial.println("  LEFT (negative): 90° - offset");
  Serial.println("  RIGHT (positive): 90° + offset");
  Serial.println("");
  
  // CRITICAL: Set servo to reference position (90°) IMMEDIATELY on power-on
  // This is the ONLY position we assume - no other assumptions about arm position
  Serial.println("Setting pin to OUTPUT mode...");
  pinMode(FOREARM_PIN, OUTPUT);
  
  Serial.print("Attaching servo and moving to reference position (");
  Serial.print(REFERENCE_ANGLE);
  Serial.println("°) IMMEDIATELY...");
  forearmServo.attach(FOREARM_PIN);
  // Write 90 degrees IMMEDIATELY - no delay before this!
  forearmServo.write(REFERENCE_ANGLE);  // Reference position (middle of range)
  delay(1000);  // Give servo time to reach 90 degrees
  
  Serial.print("Servo positioned at reference ");
  Serial.print(REFERENCE_ANGLE);
  Serial.println("° position");
  delay(500);
  
  // Test movement LEFT (negative from reference) - 90° down to 0°
  Serial.println("Moving LEFT: 90° -> 0° (90° - 90° = 0°)...");
  for (int offset = 0; offset <= 90; offset += 1) {
    int angle = REFERENCE_ANGLE - offset;  // Move left (negative from reference)
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    forearmServo.write(safeAngle);
    delay(20);  // Smooth movement (adjust for speed: lower = faster)
  }
  Serial.print("Reached minimum: ");
  Serial.print(MIN_ANGLE);
  Serial.println(" degrees");
  delay(500);
  
  // Return to reference position
  Serial.println("Returning to reference: 0° -> 90°...");
  for (int offset = 90; offset >= 0; offset -= 1) {
    int angle = REFERENCE_ANGLE - offset;
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    forearmServo.write(safeAngle);
    delay(20);
  }
  Serial.print("Reached reference: ");
  Serial.print(REFERENCE_ANGLE);
  Serial.println(" degrees");
  delay(500);
  
  // Test movement RIGHT (positive from reference) - 90° up to 180° (90° + 90°)
  Serial.println("Moving RIGHT: 90° -> 180° (90° + 90° = 180°)...");
  for (int offset = 0; offset <= 90; offset += 1) {
    int angle = REFERENCE_ANGLE + offset;  // Move right (positive from reference)
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    forearmServo.write(safeAngle);
    delay(20);  // Smooth movement
  }
  Serial.print("Reached maximum: ");
  Serial.print(MAX_ANGLE);
  Serial.println(" degrees");
  delay(500);
  
  // Return to reference position
  Serial.println("Returning to reference: 180° -> 90°...");
  for (int offset = 90; offset >= 0; offset -= 1) {
    int angle = REFERENCE_ANGLE + offset;
    int safeAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    forearmServo.write(safeAngle);
    delay(20);
  }
  Serial.print("Reached reference: ");
  Serial.print(REFERENCE_ANGLE);
  Serial.println(" degrees");
  delay(500);
  
  // Stop the servo - send reference position (90 degrees)
  Serial.print("Stopping servo at reference position (");
  Serial.print(REFERENCE_ANGLE);
  Serial.println(" degrees)...");
  forearmServo.write(REFERENCE_ANGLE);  // Reference/stop position
  delay(1000);
  
  // Detach servo to completely stop sending signals
  Serial.println("Detaching servo to stop all signals...");
  forearmServo.detach();
  
  // Set pin to INPUT to ensure no signal
  pinMode(FOREARM_PIN, INPUT);
  
  Serial.println("");
  Serial.println("========================================");
  Serial.println("Forearm servo test complete!");
  Serial.print("Servo stopped at reference position (");
  Serial.print(REFERENCE_ANGLE);
  Serial.println(" degrees)");
  Serial.print("Range tested: ");
  Serial.print(MIN_ANGLE);
  Serial.print(" - ");
  Serial.print(MAX_ANGLE);
  Serial.print(" degrees (relative to ");
  Serial.print(REFERENCE_ANGLE);
  Serial.println("°)");
  Serial.println("========================================");
}

void loop() {
  // Do nothing - rotation complete
  // Servo is detached and will not receive any signals
  delay(1000);
}

