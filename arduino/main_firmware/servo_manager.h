#ifndef SERVO_MANAGER_H
#define SERVO_MANAGER_H

#include <Arduino.h>
#include <Servo.h>
#include "config.h"
#if SHOULDER_USE_HTS16L
#include "hts16l_servo.h"
#include <HardwareSerial.h>
#endif

struct ServoConfig {
    uint8_t pin;
    int min_pulse;
    int max_pulse;
    int min_angle;
    int max_angle;
    int current_angle;
    int target_angle;
    bool is_continuous;  // True for continuous rotation servos
    bool is_serial;      // True for serial servos (HTS-16L)
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
    
    // Speed Control
    void setSpeed(int speed_deg_per_sec); // Set movement speed (degrees per second)
    int getSpeed(); // Get current speed
    
    // Status
    int getAngle(uint8_t servo_id);
    bool isMoving();

private:
    ServoConfig servos[6];
    bool _emergency_stop;
    unsigned long _last_update;
    int _current_speed; // Current speed in degrees per second
    
    // Continuous rotation tracking
    unsigned long _base_rotation_start_time;
    int _base_rotation_direction; // -1 = CCW, 0 = stop, 1 = CW
    int _base_virtual_angle; // Tracked virtual position for continuous rotation
    
    // Serial servo (HTS-16L) for shoulder
    #if SHOULDER_USE_HTS16L
    HTS16LServo hts16l_shoulder;
    HardwareSerial* hts16l_serial;
    #endif
    
    void attachServo(int id, int pin, int min_p, int max_p, int min_a, int max_a, bool continuous = false, bool serial = false);
    int constrainAngle(int id, int angle);
    void updateContinuousRotation(int id);
    void updateSerialServo(int id);
};

#endif // SERVO_MANAGER_H
