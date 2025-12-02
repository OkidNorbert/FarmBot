#include "servo_manager.h"

ServoManager::ServoManager() {
    _emergency_stop = false;
    _last_update = 0;
    _current_speed = DEFAULT_SPEED; // Initialize with default speed
    _base_rotation_start_time = 0;
    _base_rotation_direction = 0;
    _base_virtual_angle = HOME_ANGLE; // Start at home position
}

void ServoManager::begin() {
    // Initialize Servos with Config
    // ID mapping: 0=Base, 1=Shoulder, 2=Forearm, 3=Elbow, 4=Pitch, 5=Claw
    
    attachServo(0, PIN_SERVO_BASE,     PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_BASE_MIN,     LIMIT_BASE_MAX, BASE_CONTINUOUS_ROTATION);
    attachServo(1, PIN_SERVO_SHOULDER, PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX, false);
    attachServo(2, PIN_SERVO_FOREARM,  PULSE_MIN_MG99X, PULSE_MAX_MG99X, LIMIT_FOREARM_MIN,  LIMIT_FOREARM_MAX, false);
    attachServo(3, PIN_SERVO_ELBOW,    PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_ELBOW_MIN,    LIMIT_ELBOW_MAX, false);
    attachServo(4, PIN_SERVO_PITCH,    PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_PITCH_MIN,    LIMIT_PITCH_MAX, false);
    attachServo(5, PIN_SERVO_CLAW,     PULSE_MIN_SG90,  PULSE_MAX_SG90,  LIMIT_CLAW_MIN,     LIMIT_CLAW_MAX, false);
    
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
        _base_virtual_angle = HOME_ANGLE;
        _base_rotation_direction = 0;
        _base_rotation_start_time = millis();
    } else {
        servos[id].servo.write(initial_angle);
    }
}

void ServoManager::update() {
    if (_emergency_stop) return;
    
    unsigned long now = millis();
    // Use dynamic speed if set, otherwise use default
    int speed = (_current_speed > 0) ? _current_speed : DEFAULT_SPEED;
    if (now - _last_update < (1000 / speed)) return; // Speed control
    
    bool moving = false;
    for (int i = 0; i < 6; i++) {
        if (servos[i].is_continuous) {
            // Handle continuous rotation servo (base)
            updateContinuousRotation(i);
            if (servos[i].current_angle != servos[i].target_angle) {
                moving = true;
            }
        } else {
            // Standard position-based servo
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
    }
    
    if (moving) _last_update = now;
}

bool ServoManager::setTarget(uint8_t servo_id, int angle) {
    if (servo_id >= 6) return false;
    if (_emergency_stop) return false;
    
    servos[servo_id].target_angle = constrainAngle(servo_id, angle);
    
    // For continuous rotation servo, if target changed significantly, reset rotation tracking
    if (servo_id == 0 && servos[servo_id].is_continuous) {
        int diff = abs(servos[servo_id].target_angle - servos[servo_id].current_angle);
        if (diff > 10) { // Significant change
            _base_rotation_start_time = millis(); // Reset timer for new rotation
        }
    }
    
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
    // Home all servos to 90° except claw
    for (int i = 0; i < 5; i++) {  // Servos 0-4 (Base through Pitch)
        setTarget(i, HOME_ANGLE);
    }
    // Claw stays closed (CLAW_CLOSED_POSITION) - it was set to closed in begin() for safety
    // Claw: CLAW_CLOSED_POSITION (typically 0°) = Closed, 90° = Open
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

void ServoManager::setSpeed(int speed_deg_per_sec) {
    // Constrain speed to absolute hardware limits (1-180 degrees per second)
    // Mode-based limits are applied in main_firmware.ino before calling this
    _current_speed = constrain(speed_deg_per_sec, MIN_SPEED, MAX_SPEED);
}

int ServoManager::getSpeed() {
    return _current_speed;
}

void ServoManager::updateContinuousRotation(int id) {
    // Only handle base servo (id=0) as continuous rotation
    if (id != 0 || !servos[id].is_continuous) return;
    
    unsigned long now = millis();
    int target = servos[id].target_angle;
    int current = servos[id].current_angle;
    
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
    if (new_direction != _base_rotation_direction) {
        _base_rotation_direction = new_direction;
        _base_rotation_start_time = now;
    }
    
    // Calculate servo command for continuous rotation based on current speed
    // For continuous rotation: 0° = full CCW, 90° = stop, 180° = full CW
    // Scale servo command angle based on current speed setting
    // Speed range: MIN_SPEED to MAX_SPEED (typically 1-180 deg/s)
    // Servo angle range: 0-90° (CCW) and 90-180° (CW)
    // Map speed to servo angle: faster speed = further from 90° (stop)
    
    int servo_angle;
    int speed = (_current_speed > 0) ? _current_speed : DEFAULT_SPEED;
    
    // Normalize speed to 0-100% range (MIN_SPEED to MAX_SPEED)
    // Then map to servo command range: 60-90° (CCW) and 90-120° (CW)
    // This gives a safe operating range that avoids full speed extremes
    int speed_percent = map(constrain(speed, MIN_SPEED, MAX_SPEED), MIN_SPEED, MAX_SPEED, 0, 100);
    
    if (_base_rotation_direction == 1) {
        // Rotate clockwise (toward higher angles)
        // Map speed: 0% = 90° (stop), 100% = 120° (max CW)
        servo_angle = map(speed_percent, 0, 100, 90, 120);
    } else if (_base_rotation_direction == -1) {
        // Rotate counter-clockwise (toward lower angles)
        // Map speed: 0% = 90° (stop), 100% = 60° (max CCW)
        servo_angle = map(speed_percent, 0, 100, 90, 60);
    } else {
        servo_angle = 90; // Stop
    }
    
    // Calculate speed factor for coasting compensation
    float speed_factor;
    if (_base_rotation_direction == 1) {
        // CW: 90° = 0, 120° = 1.0 (full speed)
        speed_factor = (float)(servo_angle - 90) / 30.0; // 30° range (90-120)
    } else if (_base_rotation_direction == -1) {
        // CCW: 90° = 0, 60° = 1.0 (full speed)
        speed_factor = (float)(90 - servo_angle) / 30.0; // 30° range (60-90)
    } else {
        speed_factor = 0.0;
    }
    
    // Calculate coasting distance based on current rotation speed
    // Coasting occurs because servo doesn't stop instantly
    float actual_rotation_speed = BASE_ROTATION_SPEED * speed_factor;
    float coasting_distance = (actual_rotation_speed * BASE_COASTING_TIME_MS) / 1000.0; // Convert to degrees
    
    // Stop tolerance includes coasting compensation
    float stop_threshold = BASE_STOP_TOLERANCE + coasting_distance;
    
    // If we're at target (accounting for coasting), stop rotation early
    if (abs(diff) < stop_threshold) {
        if (_base_rotation_direction != 0) {
            servos[id].servo.write(90); // Stop (90° = stop for continuous rotation)
            
            // Store direction before resetting for coasting calculation
            int coast_direction = _base_rotation_direction;
            _base_rotation_direction = 0;
            
            // Account for coasting in final position
            if (abs(diff) > BASE_STOP_TOLERANCE) {
                // We stopped early to account for coasting
                // Update virtual position to account for remaining coast
                if (coast_direction == 1) {
                    _base_virtual_angle = target - coasting_distance;
                } else if (coast_direction == -1) {
                    _base_virtual_angle = target + coasting_distance;
                }
            } else {
                _base_virtual_angle = target;
            }
            
            servos[id].current_angle = (int)_base_virtual_angle;
        }
        return;
    }
    
    servos[id].servo.write(servo_angle);
    
    // Update virtual position based on time and ACTUAL rotation speed
    // Use current speed setting instead of hardcoded BASE_ROTATION_SPEED
    // BASE_ROTATION_SPEED is now used as a calibration factor to convert
    // servo command angle to actual rotation speed
    unsigned long elapsed = now - _base_rotation_start_time;
    
    // Calculate degrees rotated using calibrated base speed and speed factor
    // Reuse actual_rotation_speed calculated earlier for coasting compensation
    float degrees_rotated = (actual_rotation_speed * elapsed) / 1000.0; // Convert ms to seconds
    
    if (_base_rotation_direction == 1) {
        _base_virtual_angle = _base_virtual_angle + degrees_rotated;
        if (_base_virtual_angle >= 180) _base_virtual_angle = _base_virtual_angle - 180; // Wrap around
    } else if (_base_rotation_direction == -1) {
        _base_virtual_angle = _base_virtual_angle - degrees_rotated;
        if (_base_virtual_angle < 0) _base_virtual_angle = _base_virtual_angle + 180; // Wrap around
    }
    
    // Update current angle to virtual position
    servos[id].current_angle = (int)_base_virtual_angle;
    
    // Reset timer periodically to prevent overflow
    if (elapsed > 1000) {
        _base_rotation_start_time = now;
    }
}
