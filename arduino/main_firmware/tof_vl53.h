#ifndef TOF_VL53_H
#define TOF_VL53_H

#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_VL53L0X.h"

class ToFManager {
public:
    ToFManager();
    bool begin();
    bool isInitialized() { return _initialized; }
    int getDistance(); // Returns distance in mm
    bool isRangeValid(int distance);

private:
    Adafruit_VL53L0X lox;
    bool _initialized;
};

#endif // TOF_VL53_H
