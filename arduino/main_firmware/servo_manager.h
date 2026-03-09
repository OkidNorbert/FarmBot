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
    float current_angle;
    int target_angle;
    bool is_continuous;  // True for continuous rotation servos
    Servo servo;
};

class ServoManager {
public:
    ServoManager();
    void begin();
    void update(); // Call in loop for smooth motion
    
    // Motion Control
    bool setTarget(uint8_t servo_id, int angle);
    bool setTargets(int waist, int shoulder, int elbow, int wrist_roll, int wrist_pitch, int claw);
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
    unsigned long _rotation_start_time[6];
    int _rotation_direction[6]; // -1 = CCW, 0 = stop, 1 = CW
    float _virtual_angle[6]; // Tracked virtual position for continuous rotation
    
    void attachServo(int id, int pin, int min_p, int max_p, int min_a, int max_a, bool continuous = false);
    int constrainAngle(int id, int angle);
    void updateContinuousRotation(int id);
    
    static const int UPDATE_INTERVAL_MS = 20; // 50Hz update frequency for smooth motion
};

#endif // SERVO_MANAGER_H
