#include "tof_vl53.h"

ToFManager::ToFManager() {
    _initialized = false;
}

bool ToFManager::begin() {
    Serial.println(F("ToF: Starting I2C Scan..."));
    
    // Initialize standard I2C bus (A4/A5)
    Wire.begin();
    
    // Try to initialize on standard Wire bus
    if (lox.begin(VL53L0X_I2C_ADDR, false, &Wire)) {
        Serial.println(F("ToF: VL53L0X found on Wire (Standard pins A4/A5)"));
        _initialized = true;
        return true;
    }
    
    // If that fails, try Wire1 (often used for Qwiic/STEMMA QT connector on R4 WiFi)
    Serial.println(F("ToF: Not found on Wire. Trying Wire1 (Qwiic/D26/D27)..."));
    Wire1.begin();
    if (lox.begin(VL53L0X_I2C_ADDR, false, &Wire1)) {
        Serial.println(F("ToF: VL53L0X found on Wire1"));
        _initialized = true;
        return true;
    }

    Serial.println(F("ToF: ERROR - Sensor not detected on any I2C bus."));
    Serial.println(F("Troubleshooting:"));
    Serial.println(F("1. Check SDA/SCL wiring (swap the lines to test)"));
    Serial.println(F("2. Ensure sensor has 3.3V or 5V power and GND"));
    Serial.println(F("3. If using A4/A5, ensure they are not used for other sensors"));
    
    _initialized = false;
    return false;
}

int ToFManager::getDistance() {
    if (!_initialized) return -1;
    
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false); // pass false to get data
    
    if (measure.RangeStatus != 4) {  // phase failures have incorrect data
        return measure.RangeMilliMeter;
    } else {
        return -1; // Out of range
    }
}

bool ToFManager::isRangeValid(int distance) {
    return (distance > 20 && distance < 1200); // Valid range for this application
}
