/*
 * Base Servo Test - Pin 7
 * Tower Pro MG996R Servo Test
 * Rotates servo 360 degrees (0->180->0) and stops
 * 
 * MG996R Specifications:
 * - Operating voltage: 4.8V - 7.2V
 * - Rotation range: ~180 degrees (0-180°)
 * - Torque: 10kg/cm @ 4.8V, 11kg/cm @ 6V
 * - Speed: 0.2 sec/60° @ 4.8V
 * 
 * IMPORTANT - Neutral Position and Stop Method:
 * - Neutral/Stop Position: 90 degrees
 * - Stop Sequence:
 *   1. Send 90 degrees (neutral position)
 *   2. Detach servo (baseServo.detach())
 *   3. Set pin to INPUT mode (pinMode(BASE_PIN, INPUT))
 * - This method prevents continuous rotation after movement
 */

#include <Servo.h>

const int BASE_PIN = 7;  // Base servo on pin 7
Servo baseServo;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  
  Serial.println("========================================");
  Serial.println("Tower Pro MG996R Base Servo Test");
  Serial.println("Pin: 7");
  Serial.println("========================================");
  Serial.println("Rotating 360 degrees (0->180->0)...");
  Serial.println("");
  
  // Attach servo to pin 7
  baseServo.attach(BASE_PIN);
  delay(500);  // Give servo time to initialize
  
  // Start at 0 degrees
  Serial.println("Starting position: 0 degrees");
  baseServo.write(0);
  delay(1000);
  
  // Rotate from 0 to 180 degrees (first half of rotation - 180°)
  Serial.println("Rotating: 0 -> 180 degrees...");
  for (int angle = 0; angle <= 180; angle += 1) {
    baseServo.write(angle);
    delay(20);  // Smooth movement (adjust for speed: lower = faster)
  }
  Serial.println("Reached 180 degrees");
  delay(500);
  
  // Rotate from 180 back to 0 degrees (second half of rotation - 180°)
  Serial.println("Rotating: 180 -> 0 degrees...");
  for (int angle = 180; angle >= 0; angle -= 1) {
    baseServo.write(angle);
    delay(20);  // Smooth movement
  }
  Serial.println("Reached 0 degrees");
  delay(500);
  
  // Stop the servo - send neutral position (90 degrees)
  // For continuous rotation servos, 90 is the stop position
  Serial.println("Stopping servo at neutral position (90 degrees)...");
  baseServo.write(90);  // Neutral/stop position
  delay(1000);
  
  // Detach servo to completely stop sending signals
  Serial.println("Detaching servo to stop all signals...");
  baseServo.detach();
  
  // Set pin to INPUT to ensure no signal
  pinMode(BASE_PIN, INPUT);
  
  Serial.println("");
  Serial.println("========================================");
  Serial.println("360 degree rotation complete!");
  Serial.println("Servo stopped and detached");
  Serial.println("========================================");
}

void loop() {
  // Do nothing - rotation complete
  // Servo is detached and will not receive any signals
  delay(1000);
}

