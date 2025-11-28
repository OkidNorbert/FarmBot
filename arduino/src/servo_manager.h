#ifndef SERVO_MANAGER_H
#define SERVO_MANAGER_H

#include <Arduino.h>
#include <Servo.h>
#include "config.h"

struct ServoConfig {
    uint8_t pin;
    int min_pulse;
    int max_pulse;
    int min_angle;
    int max_angle;
    int current_angle;
    int target_angle;
    Servo servo;
};

class ServoManager {
public:
    ServoManager();
    void begin();
    void update(); // Call in loop for smooth motion
    
    // Motion Control
    bool setTarget(uint8_t servo_id, int angle);
    bool setTargets(int base, int shoulder, int forearm, int elbow, int pitch, int claw);
    void home();
    void emergencyStop();
    
    // Status
    int getAngle(uint8_t servo_id);
    bool isMoving();

private:
    ServoConfig servos[6];
    bool _emergency_stop;
    unsigned long _last_update;
    
    void attachServo(int id, int pin, int min_p, int max_p, int min_a, int max_a);
    int constrainAngle(int id, int angle);
};

#endif // SERVO_MANAGER_H
