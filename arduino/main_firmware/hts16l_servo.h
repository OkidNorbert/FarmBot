#ifndef HTS16L_SERVO_H
#define HTS16L_SERVO_H

#include <Arduino.h>

// HTS-16L Serial Servo Driver
// Uses UART communication at 115200 baud
// Servo ID: 1 (default, can be changed)

class HTS16LServo {
public:
    HTS16LServo();
    bool begin(HardwareSerial* serial, uint8_t servo_id = 1, uint32_t baud = 115200);
    bool setAngle(int angle);  // Set angle 0-240 degrees
    bool setAngle(int angle, int speed);  // Set angle with speed (0-1000)
    int getAngle();  // Get current angle (if feedback available)
    void stop();  // Stop servo movement
    
private:
    HardwareSerial* _serial;
    uint8_t _servo_id;
    int _current_angle;
    bool _initialized;
    
    void sendCommand(uint8_t cmd, uint8_t* data, uint8_t len);
    uint16_t calculateChecksum(uint8_t* data, uint8_t len);
};

#endif // HTS16L_SERVO_H

