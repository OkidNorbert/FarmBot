#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <Arduino.h>
#include <EEPROM.h>

// EEPROM Address Map
#define EEPROM_MAGIC_ADDR      0
#define EEPROM_MAGIC_VALUE     0xAB
#define EEPROM_SERVO_TRIMS     10  // 6 servos * 2 bytes = 12 bytes (10-21)
#define EEPROM_BIN_POSES       30  // 2 bins * 6 servos * 2 bytes = 24 bytes (30-53)
#define EEPROM_PIXEL_MAP       60  // Calibration points (variable size)

struct ServoTrim {
    int16_t offset;  // Trim offset from 90° center
    int16_t min_angle;
    int16_t max_angle;
};

struct CalibrationData {
    ServoTrim servo_trims[6];
    int16_t bin_ripe[6];
    int16_t bin_unripe[6];
    bool valid;
};

class CalibrationManager {
public:
    CalibrationManager();
    
    // Load/Save
    bool load();
    bool save();
    void reset();
    
    // Servo Trims
    void setServoTrim(uint8_t servo_id, int16_t offset);
    void setServoLimits(uint8_t servo_id, int16_t min_angle, int16_t max_angle);
    int16_t getServoTrim(uint8_t servo_id);
    ServoTrim getServoTrimData(uint8_t servo_id);
    
    // Bin Poses
    void setBinPose(String bin_type, int16_t angles[6]);
    void getBinPose(String bin_type, int16_t angles[6]);
    
    // Status
    bool isValid();
    
private:
    CalibrationData _data;
    bool _dirty; // Needs saving
    
    void initDefaults();
};

#endif // CALIBRATION_H

