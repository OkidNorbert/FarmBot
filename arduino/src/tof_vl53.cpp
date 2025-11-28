#include "tof_vl53.h"

ToFManager::ToFManager() {
    _initialized = false;
}

bool ToFManager::begin() {
    if (!lox.begin()) {
        Serial.println(F("Failed to boot VL53L0X"));
        return false;
    }
    _initialized = true;
    return true;
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
