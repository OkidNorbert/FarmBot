#include "servo_manager.h"

ServoManager::ServoManager() {
    _emergency_stop = false;
    _last_update = 0;
}

void ServoManager::begin() {
    // Initialize Servos with Config
    // ID mapping: 0=Base, 1=Shoulder, 2=Forearm, 3=Elbow, 4=Pitch, 5=Claw
    
    attachServo(0, PIN_SERVO_BASE,     PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_BASE_MIN,     LIMIT_BASE_MAX);
    attachServo(1, PIN_SERVO_SHOULDER, PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
    attachServo(2, PIN_SERVO_FOREARM,  PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_FOREARM_MIN,  LIMIT_FOREARM_MAX);
    attachServo(3, PIN_SERVO_ELBOW,    PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_ELBOW_MIN,    LIMIT_ELBOW_MAX);
    attachServo(4, PIN_SERVO_PITCH,    PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_PITCH_MIN,    LIMIT_PITCH_MAX);
    attachServo(5, PIN_SERVO_CLAW,     PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_CLAW_MIN,     LIMIT_CLAW_MAX);
    
    home();
}

void ServoManager::attachServo(int id, int pin, int min_p, int max_p, int min_a, int max_a) {
    servos[id].pin = pin;
    servos[id].min_pulse = min_p;
    servos[id].max_pulse = max_p;
    servos[id].min_angle = min_a;
    servos[id].max_angle = max_a;
    servos[id].current_angle = HOME_ANGLE;
    servos[id].target_angle = HOME_ANGLE;
    
    servos[id].servo.attach(pin, min_p, max_p);
    servos[id].servo.write(HOME_ANGLE);
}

void ServoManager::update() {
    if (_emergency_stop) return;
    
    unsigned long now = millis();
    if (now - _last_update < (1000 / DEFAULT_SPEED)) return; // Speed control
    
    bool moving = false;
    for (int i = 0; i < 6; i++) {
        if (servos[i].current_angle != servos[i].target_angle) {
            moving = true;
            if (servos[i].current_angle < servos[i].target_angle) {
                servos[i].current_angle++;
            } else {
                servos[i].current_angle--;
            }
            servos[i].servo.write(servos[i].current_angle);
        }
    }
    
    if (moving) _last_update = now;
}

bool ServoManager::setTarget(uint8_t servo_id, int angle) {
    if (servo_id >= 6) return false;
    if (_emergency_stop) return false;
    
    servos[servo_id].target_angle = constrainAngle(servo_id, angle);
    return true;
}

bool ServoManager::setTargets(int base, int shoulder, int forearm, int elbow, int pitch, int claw) {
    bool success = true;
    success &= setTarget(0, base);
    success &= setTarget(1, shoulder);
    success &= setTarget(2, forearm);
    success &= setTarget(3, elbow);
    success &= setTarget(4, pitch);
    success &= setTarget(5, claw);
    return success;
}

int ServoManager::constrainAngle(int id, int angle) {
    if (angle < servos[id].min_angle) return servos[id].min_angle;
    if (angle > servos[id].max_angle) return servos[id].max_angle;
    return angle;
}

void ServoManager::home() {
    for (int i = 0; i < 6; i++) {
        setTarget(i, HOME_ANGLE);
    }
    // Claw closed (0) or open (90)? Prompt says 90 on power up.
    // But usually home for claw is open. Let's stick to 90 as requested.
}

void ServoManager::emergencyStop() {
    _emergency_stop = true;
    for (int i = 0; i < 6; i++) {
        servos[i].servo.detach(); // Cut power signal
    }
}

int ServoManager::getAngle(uint8_t servo_id) {
    if (servo_id >= 6) return -1;
    return servos[servo_id].current_angle;
}

bool ServoManager::isMoving() {
    for (int i = 0; i < 6; i++) {
        if (servos[i].current_angle != servos[i].target_angle) return true;
    }
    return false;
}
