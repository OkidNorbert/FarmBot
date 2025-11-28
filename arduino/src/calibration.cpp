#include "calibration.h"

CalibrationManager::CalibrationManager() {
    _dirty = false;
    _data.valid = false;
    initDefaults();
}

bool CalibrationManager::load() {
    // Check magic value
    uint8_t magic = EEPROM.read(EEPROM_MAGIC_ADDR);
    if (magic != EEPROM_MAGIC_VALUE) {
        Serial.println("No calibration data found, using defaults");
        return false;
    }
    
    // Load servo trims
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_SERVO_TRIMS + (i * 4);
        _data.servo_trims[i].offset = (int16_t)(EEPROM.read(addr) | (EEPROM.read(addr + 1) << 8));
        _data.servo_trims[i].min_angle = (int16_t)(EEPROM.read(addr + 2) | (EEPROM.read(addr + 3) << 8));
        _data.servo_trims[i].max_angle = (int16_t)(EEPROM.read(addr + 4) | (EEPROM.read(addr + 5) << 8));
    }
    
    // Load bin poses
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_BIN_POSES + (i * 2);
        _data.bin_ripe[i] = (int16_t)(EEPROM.read(addr) | (EEPROM.read(addr + 1) << 8));
    }
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_BIN_POSES + 12 + (i * 2);
        _data.bin_unripe[i] = (int16_t)(EEPROM.read(addr) | (EEPROM.read(addr + 1) << 8));
    }
    
    _data.valid = true;
    Serial.println("Calibration data loaded from EEPROM");
    return true;
}

bool CalibrationManager::save() {
    // Write magic value
    EEPROM.write(EEPROM_MAGIC_ADDR, EEPROM_MAGIC_VALUE);
    
    // Save servo trims
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_SERVO_TRIMS + (i * 6);
        EEPROM.write(addr, _data.servo_trims[i].offset & 0xFF);
        EEPROM.write(addr + 1, (_data.servo_trims[i].offset >> 8) & 0xFF);
        EEPROM.write(addr + 2, _data.servo_trims[i].min_angle & 0xFF);
        EEPROM.write(addr + 3, (_data.servo_trims[i].min_angle >> 8) & 0xFF);
        EEPROM.write(addr + 4, _data.servo_trims[i].max_angle & 0xFF);
        EEPROM.write(addr + 5, (_data.servo_trims[i].max_angle >> 8) & 0xFF);
    }
    
    // Save bin poses
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_BIN_POSES + (i * 2);
        EEPROM.write(addr, _data.bin_ripe[i] & 0xFF);
        EEPROM.write(addr + 1, (_data.bin_ripe[i] >> 8) & 0xFF);
    }
    for (int i = 0; i < 6; i++) {
        int addr = EEPROM_BIN_POSES + 12 + (i * 2);
        EEPROM.write(addr, _data.bin_unripe[i] & 0xFF);
        EEPROM.write(addr + 1, (_data.bin_unripe[i] >> 8) & 0xFF);
    }
    
    _dirty = false;
    Serial.println("Calibration data saved to EEPROM");
    return true;
}

void CalibrationManager::reset() {
    initDefaults();
    _dirty = true;
    Serial.println("Calibration reset to defaults");
}

void CalibrationManager::setServoTrim(uint8_t servo_id, int16_t offset) {
    if (servo_id >= 6) return;
    _data.servo_trims[servo_id].offset = offset;
    _dirty = true;
}

void CalibrationManager::setServoLimits(uint8_t servo_id, int16_t min_angle, int16_t max_angle) {
    if (servo_id >= 6) return;
    _data.servo_trims[servo_id].min_angle = min_angle;
    _data.servo_trims[servo_id].max_angle = max_angle;
    _dirty = true;
}

int16_t CalibrationManager::getServoTrim(uint8_t servo_id) {
    if (servo_id >= 6) return 0;
    return _data.servo_trims[servo_id].offset;
}

ServoTrim CalibrationManager::getServoTrimData(uint8_t servo_id) {
    ServoTrim trim = {0, 0, 180};
    if (servo_id < 6) {
        trim = _data.servo_trims[servo_id];
    }
    return trim;
}

void CalibrationManager::setBinPose(String bin_type, int16_t angles[6]) {
    int16_t* target = (bin_type == "ripe") ? _data.bin_ripe : _data.bin_unripe;
    for (int i = 0; i < 6; i++) {
        target[i] = angles[i];
    }
    _dirty = true;
}

void CalibrationManager::getBinPose(String bin_type, int16_t angles[6]) {
    int16_t* source = (bin_type == "ripe") ? _data.bin_ripe : _data.bin_unripe;
    for (int i = 0; i < 6; i++) {
        angles[i] = source[i];
    }
}

bool CalibrationManager::isValid() {
    return _data.valid;
}

void CalibrationManager::initDefaults() {
    // Initialize with default values
    for (int i = 0; i < 6; i++) {
        _data.servo_trims[i].offset = 0;
        _data.servo_trims[i].min_angle = 0;
        _data.servo_trims[i].max_angle = 180;
    }
    
    // Default bin poses
    int16_t ripe_default[6] = {150, 60, 110, 90, 80, 90};
    int16_t unripe_default[6] = {30, 60, 110, 90, 80, 90};
    
    for (int i = 0; i < 6; i++) {
        _data.bin_ripe[i] = ripe_default[i];
        _data.bin_unripe[i] = unripe_default[i];
    }
    
    _data.valid = true;
}

