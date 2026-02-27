#include <Servo.h>

/**
 * Standalone Test Script for SunFounder SF006FM Digital Servo
 * 
 * Specs for SF006FM:
 * - Pulse Width Range: 500us - 2500us
 * - Angle Range: 180 degrees
 */

// --- Configuration ---
const int SERVO_PIN = 3;  // <--- CHANGE THIS to your actual output pin
const int TARGET_ANGLE = 90;

Servo myServo;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- SunFounder SF006FM Digital Servo Test ---");
  
  // Attach with specific pulse widths for SF006FM (500us to 2500us)
  // This ensures the 90 degree position is dead-center.
  myServo.attach(SERVO_PIN, 500, 2500);
  
  Serial.print("Moving to ");
  Serial.print(TARGET_ANGLE);
  Serial.println(" degrees...");
  
  myServo.write(TARGET_ANGLE);
  
  Serial.println("Position set. Listening on Serial...");
  Serial.println("Tip: You can type an angle in the Serial Monitor (0-180) to move it.");
}

void loop() {
  // Optional: Allow manual testing via Serial Monitor
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();
    if (angle >= 0 && angle <= 180) {
      Serial.print("Moving to custom angle: ");
      Serial.println(angle);
      myServo.write(angle);
    }
  }
}
