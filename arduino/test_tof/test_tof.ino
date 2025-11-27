/*
 * Test Sketch for VL53L0X Time-of-Flight Sensor
 * 
 * Instructions:
 * 1. Install "Adafruit_VL53L0X" library via Arduino Library Manager.
 * 2. Connect VCC -> 3.3V (or 5V if your module supports it)
 * 3. Connect GND -> GND
 * 4. Connect SDA -> SDA (A4 on Uno/Nano, SDA on R4)
 * 5. Connect SCL -> SCL (A5 on Uno/Nano, SCL on R4)
 */

#include "Adafruit_VL53L0X.h"

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

void setup() {
  Serial.begin(115200);

  // Wait for serial port to open
  while (!Serial) {
    delay(1);
  }
  
  Serial.println("Adafruit VL53L0X test");

  if (!lox.begin()) {
    Serial.println(F("Failed to boot VL53L0X"));
    while(1);
  }
  // power 
  Serial.println(F("VL53L0X API Simple Ranging example\n\n")); 
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;
    
  Serial.print("Reading a measurement... ");
  lox.rangingTest(&measure, false); // pass in 'true' to get debug data printout!

  if (measure.RangeStatus != 4) {  // phase failures have incorrect data
    Serial.print("Distance (mm): "); Serial.println(measure.RangeMilliMeter);
  } else {
    Serial.println(" out of range ");
  }
    
  delay(100);
}
