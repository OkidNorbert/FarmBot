#include "hts16l_servo.h"

// HTS-16L Command Definitions
#define CMD_SERVO_MOVE       0x01
#define CMD_SERVO_STOP       0x02
#define CMD_SERVO_READ       0x03

HTS16LServo::HTS16LServo() {
    _serial = nullptr;
    _servo_id = 1;
    _current_angle = 90;
    _initialized = false;
}

bool HTS16LServo::begin(HardwareSerial* serial, uint8_t servo_id, uint32_t baud) {
    _serial = serial;
    _servo_id = servo_id;
    _current_angle = 90;
    
    if (_serial) {
        _serial->begin(baud);
        _initialized = true;
        delay(100); // Wait for serial to initialize
        return true;
    }
    return false;
}

bool HTS16LServo::setAngle(int angle) {
    return setAngle(angle, 500); // Default speed
}

bool HTS16LServo::setAngle(int angle, int speed) {
    if (!_initialized || !_serial) return false;
    
    // Constrain angle to 0-240 degrees (HTS-16L range)
    angle = constrain(angle, 0, 240);
    speed = constrain(speed, 0, 1000);
    
    // Convert angle to servo position (0-1000 for 0-240 degrees)
    // HTS-16L uses 0-1000 range for 0-240 degrees
    uint16_t position = map(angle, 0, 240, 0, 1000);
    
    // Prepare command data
    uint8_t cmd_data[5];
    cmd_data[0] = _servo_id;
    cmd_data[1] = (position >> 8) & 0xFF;  // High byte
    cmd_data[2] = position & 0xFF;          // Low byte
    cmd_data[3] = (speed >> 8) & 0xFF;      // Speed high byte
    cmd_data[4] = speed & 0xFF;             // Speed low byte
    
    sendCommand(CMD_SERVO_MOVE, cmd_data, 5);
    
    _current_angle = angle;
    return true;
}

int HTS16LServo::getAngle() {
    // Return last set angle (no feedback without reading from servo)
    return _current_angle;
}

void HTS16LServo::stop() {
    if (!_initialized || !_serial) return;
    
    uint8_t cmd_data[1];
    cmd_data[0] = _servo_id;
    sendCommand(CMD_SERVO_STOP, cmd_data, 1);
}

void HTS16LServo::sendCommand(uint8_t cmd, uint8_t* data, uint8_t len) {
    if (!_serial) return;
    
    // HTS-16L Protocol: [0xFF 0xFF] [ID] [Length] [Command] [Data...] [Checksum]
    _serial->write(0xFF);
    _serial->write(0xFF);
    _serial->write(_servo_id);
    _serial->write(len + 1); // Length = data length + command byte
    _serial->write(cmd);
    
    // Send data
    for (uint8_t i = 0; i < len; i++) {
        _serial->write(data[i]);
    }
    
    // Calculate and send checksum
    uint8_t checksum_data[64];
    checksum_data[0] = _servo_id;
    checksum_data[1] = len + 1;
    checksum_data[2] = cmd;
    for (uint8_t i = 0; i < len; i++) {
        checksum_data[3 + i] = data[i];
    }
    
    uint16_t checksum = calculateChecksum(checksum_data, len + 3);
    _serial->write(checksum & 0xFF);
    
    delay(1); // Small delay for transmission
}

uint16_t HTS16LServo::calculateChecksum(uint8_t* data, uint8_t len) {
    uint16_t sum = 0;
    for (uint8_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return ~sum; // Invert checksum
}

