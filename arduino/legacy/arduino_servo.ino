// arduino_servo.ino
#include <Servo.h>

Servo s1, s2, s3;
const int pin1 = 3;
const int pin2 = 5;
const int pin3 = 6;

int cur1 = 90, cur2 = 90, cur3 = 90;
int tgt1 = 90, tgt2 = 90, tgt3 = 90;
unsigned long lastMove = 0;
const int stepDelay = 20; // ms between servo steps
const int stepSize = 1; // degrees per step

void setup() {
  Serial.begin(115200);
  s1.attach(pin1);
  s2.attach(pin2);
  s3.attach(pin3);
  s1.write(cur1);
  s2.write(cur2);
  s3.write(cur3);
  delay(500);
  Serial.println("Servo controller ready");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      if (line.startsWith("ANGLE")) {
        int a1, a2, a3;
        int n = sscanf(line.c_str(), "ANGLE %d %d %d", &a1, &a2, &a3);
        if (n >= 3) {
          tgt1 = constrain(a1, 0, 180);
          tgt2 = constrain(a2, 0, 180);
          tgt3 = constrain(a3, 0, 180);
          Serial.print("New targets: ");
          Serial.print(tgt1); Serial.print(" ");
          Serial.print(tgt2); Serial.print(" ");
          Serial.println(tgt3);
        }
      } else if (line.startsWith("STOP")) {
        // bring to safe pose
        tgt1 = 90; tgt2 = 90; tgt3 = 90;
      }
    }
  }

  // Interpolate servo positions
  if (millis() - lastMove > stepDelay) {
    bool moved = false;
    if (cur1 < tgt1) { cur1 = min(cur1 + stepSize, tgt1); moved = true; }
    else if (cur1 > tgt1) { cur1 = max(cur1 - stepSize, tgt1); moved = true; }

    if (cur2 < tgt2) { cur2 = min(cur2 + stepSize, tgt2); moved = true; }
    else if (cur2 > tgt2) { cur2 = max(cur2 - stepSize, tgt2); moved = true; }

    if (cur3 < tgt3) { cur3 = min(cur3 + stepSize, tgt3); moved = true; }
    else if (cur3 > tgt3) { cur3 = max(cur3 - stepSize, tgt3); moved = true; }

    if (moved) {
      s1.write(cur1);
      s2.write(cur2);
      s3.write(cur3);
    }
    lastMove = millis();
  }
}
