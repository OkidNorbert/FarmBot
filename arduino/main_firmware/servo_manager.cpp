#include "servo_manager.h"

ServoManager::ServoManager() {
    _emergency_stop = false;
    _last_update = 0;
    _current_speed = DEFAULT_SPEED; // Initialize with default speed
    
    _rotation_start_time[0] = 0; _rotation_direction[0] = 0; _virtual_angle[0] = HOME_BASE_ANGLE;
    _rotation_start_time[1] = 0; _rotation_direction[1] = 0; _virtual_angle[1] = HOME_SHOULDER_ANGLE;
    _rotation_start_time[2] = 0; _rotation_direction[2] = 0; _virtual_angle[2] = HOME_ELBOW_ANGLE;
    _rotation_start_time[3] = 0; _rotation_direction[3] = 0; _virtual_angle[3] = HOME_WRIST_ROLL_ANGLE;
    _rotation_start_time[4] = 0; _rotation_direction[4] = 0; _virtual_angle[4] = HOME_WRIST_PITCH_ANGLE;
    _rotation_start_time[5] = 0; _rotation_direction[5] = 0; _virtual_angle[5] = HOME_CLAW_ANGLE;
}

void ServoManager::begin() {
    // Initialize Servos with Config
    // ID mapping: 0=Waist, 1=Shoulder, 2=Elbow, 3=Wrist Roll, 4=Wrist Pitch, 5=Claw
    
    attachServo(0, PIN_SERVO_BASE,         PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_BASE_MIN,        LIMIT_BASE_MAX, BASE_CONTINUOUS_ROTATION);
    attachServo(1, PIN_SERVO_SHOULDER,     PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_SHOULDER_MIN,    LIMIT_SHOULDER_MAX, false);
    attachServo(2, PIN_SERVO_ELBOW,        PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_ELBOW_MIN,       LIMIT_ELBOW_MAX, false);
    attachServo(3, PIN_SERVO_WRIST_ROLL,   PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_WRIST_ROLL_MIN,  LIMIT_WRIST_ROLL_MAX, false);
    attachServo(4, PIN_SERVO_WRIST_PITCH,  PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_WRIST_PITCH_MIN, LIMIT_WRIST_PITCH_MAX, PITCH_CONTINUOUS_ROTATION);
    attachServo(5, PIN_SERVO_CLAW,         PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_CLAW_MIN,        LIMIT_CLAW_MAX, false);
    
    // Set claw to closed position immediately on power-on for safety
    // Use CLAW_CLOSED_POSITION (may be 0-5° depending on servo calibration)
    servos[5].servo.write(CLAW_CLOSED_POSITION);
    servos[5].current_angle = CLAW_CLOSED_POSITION;
    servos[5].target_angle = CLAW_CLOSED_POSITION;
    delay(500); // Give servo time to move to closed position
    
    home();
}

void ServoManager::attachServo(int id, int pin, int min_p, int max_p, int min_a, int max_a, bool continuous) {
    servos[id].pin = pin;
    servos[id].min_pulse = min_p;
    servos[id].max_pulse = max_p;
    servos[id].min_angle = min_a;
    servos[id].max_angle = max_a;
    servos[id].is_continuous = continuous;
    
    // Claw (id=5) starts at CLAW_CLOSED_POSITION (closed) for safety, all others at 90° (home)
    int initial_angle = (id == 5) ? CLAW_CLOSED_POSITION : HOME_ANGLE;
    servos[id].current_angle = initial_angle;
    servos[id].target_angle = initial_angle;
    
    servos[id].servo.attach(pin, min_p, max_p);
    
    if (continuous) {
        // For continuous rotation: 90° = stop, <90° = CCW, >90° = CW
        servos[id].servo.write(90); // Start stopped
        _virtual_angle[id] = initial_angle;
        _rotation_direction[id] = 0;
        _rotation_start_time[id] = millis();
    } else {
        servos[id].servo.write(initial_angle);
    }
}

void ServoManager::update() {
    if (_emergency_stop) return;
    
    unsigned long now = millis();
    if (now - _last_update < UPDATE_INTERVAL_MS) return;
    
    float dt = (float)(now - _last_update) / 1000.0f;
    _last_update = now;
    
    // Mode-based speed
    float speed = (_current_speed > 0) ? (float)_current_speed : (float)DEFAULT_SPEED;
    float max_step = speed * dt;
    
    for (int i = 0; i < 6; i++) {
        if (servos[i].is_continuous) {
            updateContinuousRotation(i);
        } else {
            // Standard position-based servo with smooth interpolation
            if (abs(servos[i].current_angle - (float)servos[i].target_angle) > 0.1f) {
                float diff = (float)servos[i].target_angle - servos[i].current_angle;
                
                if (abs(diff) <= max_step) {
                    servos[i].current_angle = (float)servos[i].target_angle;
                } else {
                    servos[i].current_angle += (diff > 0 ? 1.0f : -1.0f) * max_step;
                }
                
                servos[i].servo.write((int)round(servos[i].current_angle));
            }
        }
    }
}

bool ServoManager::setTarget(uint8_t servo_id, int angle) {
    if (servo_id >= 6) return false;
    if (_emergency_stop) return false;
    
    servos[servo_id].target_angle = constrainAngle(servo_id, angle);
    
    // For continuous rotation servo, if target changed significantly, reset rotation tracking
    if (servos[servo_id].is_continuous) {
        int diff = abs(servos[servo_id].target_angle - servos[servo_id].current_angle);
        if (diff > 5) { // Significant change
            _rotation_start_time[servo_id] = millis(); // Reset timer for new rotation
        }
    }
    
    return true;
}

bool ServoManager::setTargets(int waist, int shoulder, int elbow, int wrist_roll, int wrist_pitch, int claw) {
    bool success = true;
    success &= setTarget(0, waist);
    success &= setTarget(1, shoulder);
    success &= setTarget(2, elbow);
    success &= setTarget(3, wrist_roll);
    success &= setTarget(4, wrist_pitch);
    success &= setTarget(5, claw);
    return success;
}

int ServoManager::constrainAngle(int id, int angle) {
    if (angle < servos[id].min_angle) return servos[id].min_angle;
    if (angle > servos[id].max_angle) return servos[id].max_angle;
    return angle;
}

void ServoManager::home() {
    // Home all servos to their configured positions
    setTarget(0, HOME_BASE_ANGLE);
    setTarget(1, HOME_SHOULDER_ANGLE);
    setTarget(2, HOME_ELBOW_ANGLE);
    setTarget(3, HOME_WRIST_ROLL_ANGLE);
    setTarget(4, HOME_WRIST_PITCH_ANGLE);
    setTarget(5, HOME_CLAW_ANGLE);
}

void ServoManager::emergencyStop() {
    _emergency_stop = true;
    for (int i = 0; i < 6; i++) {
        servos[i].servo.detach(); // Cut power signal
    }
}

int ServoManager::getAngle(uint8_t servo_id) {
    if (servo_id >= 6) return -1;
    return (int)round(servos[servo_id].current_angle);
}

bool ServoManager::isMoving() {
    for (int i = 0; i < 6; i++) {
        if (abs(servos[i].current_angle - (float)servos[i].target_angle) > 0.1f) return true;
    }
    return false;
}

void ServoManager::setSpeed(int speed_deg_per_sec) {
    // Constrain speed to absolute hardware limits (1-180 degrees per second)
    // Mode-based limits are applied in main_firmware.ino before calling this
    _current_speed = constrain(speed_deg_per_sec, MIN_SPEED, MAX_SPEED);
}

int ServoManager::getSpeed() {
    return _current_speed;
}

void ServoManager::updateContinuousRotation(int id) {
    if (!servos[id].is_continuous) return;
    
    unsigned long now = millis();
    int target = servos[id].target_angle;
    int current = servos[id].current_angle;
    
    // Select constants based on ID
    float rot_speed = (id == 0) ? BASE_ROTATION_SPEED : PITCH_ROTATION_SPEED;
    int coast_time = (id == 0) ? BASE_COASTING_TIME_MS : PITCH_COASTING_TIME_MS;
    int stop_tol = (id == 0) ? BASE_STOP_TOLERANCE : PITCH_STOP_TOLERANCE;
    
    // Calculate shortest rotation path (handle 0-180 wrap-around)
    int diff = target - current;
    if (diff > 90) {
        diff = diff - 180; // Go the other way
    } else if (diff < -90) {
        diff = diff + 180; // Go the other way
    }
    
    // Determine rotation direction
    int new_direction = (diff > 0) ? 1 : -1;
    
    // If direction changed, restart rotation timer
    if (new_direction != _rotation_direction[id]) {
        _rotation_direction[id] = new_direction;
        _rotation_start_time[id] = now;
    }
    
    int servo_angle;
    int speed = (_current_speed > 0) ? _current_speed : DEFAULT_SPEED;
    int speed_percent = map(constrain(speed, MIN_SPEED, MAX_SPEED), MIN_SPEED, MAX_SPEED, 0, 100);
    
    if (_rotation_direction[id] == 1) {
        servo_angle = map(speed_percent, 0, 100, 90, 120);
    } else if (_rotation_direction[id] == -1) {
        servo_angle = map(speed_percent, 0, 100, 90, 60);
    } else {
        servo_angle = 90; // Stop
    }
    
    float speed_factor;
    if (_rotation_direction[id] == 1) {
        speed_factor = (float)(servo_angle - 90) / 30.0;
    } else if (_rotation_direction[id] == -1) {
        speed_factor = (float)(90 - servo_angle) / 30.0;
    } else {
        speed_factor = 0.0;
    }
    
    float actual_rotation_speed = rot_speed * speed_factor;
    float coasting_distance = (actual_rotation_speed * coast_time) / 1000.0;
    float stop_threshold = stop_tol + coasting_distance;
    
    if (abs(diff) < stop_threshold) {
        if (_rotation_direction[id] != 0) {
            servos[id].servo.write(90);
            int coast_direction = _rotation_direction[id];
            _rotation_direction[id] = 0;
            
            if (abs(diff) > stop_tol) {
                if (coast_direction == 1) {
                    _virtual_angle[id] = target - coasting_distance;
                } else if (coast_direction == -1) {
                    _virtual_angle[id] = target + coasting_distance;
                }
            } else {
                _virtual_angle[id] = target;
            }
            servos[id].current_angle = (int)_virtual_angle[id];
        }
        return;
    }
    
    servos[id].servo.write(servo_angle);
    unsigned long elapsed = now - _rotation_start_time[id];
    float degrees_rotated = (actual_rotation_speed * elapsed) / 1000.0;
    
    if (_rotation_direction[id] == 1) {
        _virtual_angle[id] = _virtual_angle[id] + degrees_rotated;
        if (_virtual_angle[id] >= 180) _virtual_angle[id] -= 180;
    } else if (_rotation_direction[id] == -1) {
        _virtual_angle[id] = _virtual_angle[id] - degrees_rotated;
        if (_virtual_angle[id] < 0) _virtual_angle[id] += 180;
    }
    
    servos[id].current_angle = _virtual_angle[id];
    if (elapsed > 1000) {
        _rotation_start_time[id] = now;
    }
}
