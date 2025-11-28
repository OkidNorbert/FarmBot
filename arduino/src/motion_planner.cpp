#include "motion_planner.h"

MotionPlanner::MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr) {
    _servoMgr = servoMgr;
    _tofMgr = tofMgr;
    _currentState = PICK_IDLE;
    _approachOffsetMm = 50;  // Default 50mm approach offset
    _graspDistanceMm = 30;   // Default 30mm grasp distance
    _liftHeightDeg = 20;     // Default 20° lift
    
    // Default bin poses (will be calibrated)
    _binRipe.base = 150;
    _binRipe.shoulder = 60;
    _binRipe.forearm = 110;
    _binRipe.elbow = 90;
    _binRipe.pitch = 80;
    _binRipe.claw = 90;
    
    _binUnripe.base = 30;
    _binUnripe.shoulder = 60;
    _binUnripe.forearm = 110;
    _binUnripe.elbow = 90;
    _binUnripe.pitch = 80;
    _binUnripe.claw = 90;
}

bool MotionPlanner::startPick(int pixel_x, int pixel_y, float confidence, String class_type) {
    if (_currentState != PICK_IDLE) {
        _lastError = "Already picking";
        return false;
    }
    
    if (!_tofMgr->isRangeValid(_tofMgr->getDistance())) {
        _lastError = "ToF sensor not ready";
        return false;
    }
    
    _targetPixelX = pixel_x;
    _targetPixelY = pixel_y;
    _targetClass = class_type;
    _targetConfidence = confidence;
    _pickId = String(millis()); // Simple ID generation
    
    _currentState = PICK_CALCULATE_POSE;
    _stateStartTime = millis();
    _lastToFRead = 0;
    
    return true;
}

void MotionPlanner::update() {
    if (_currentState == PICK_IDLE || _currentState == PICK_COMPLETE || _currentState == PICK_ABORTED) {
        return;
    }
    
    // Check for emergency stop
    if (!_servoMgr->isMoving() && _currentState != PICK_GRASP && _currentState != PICK_RELEASE) {
        // Check if we're stuck
        if (millis() - _stateStartTime > 10000) { // 10 second timeout
            _lastError = "State timeout";
            _currentState = PICK_ABORTED;
            return;
        }
    }
    
    switch (_currentState) {
        case PICK_CALCULATE_POSE:
            calculateTargetPose();
            _currentState = PICK_MOVE_TO_APPROACH;
            _stateStartTime = millis();
            break;
            
        case PICK_MOVE_TO_APPROACH: {
            // Calculate approach pose (offset from target)
            int target_base = pixelToBaseAngle(_targetPixelX);
            int approach_base = target_base;
            int approach_shoulder = 60; // Lower shoulder for approach
            int approach_forearm = 100;
            int approach_elbow = 90;
            int approach_pitch = 100; // Pitch down for approach
            int approach_claw = 90; // Open
            
            if (moveToPose(approach_base, approach_shoulder, approach_forearm, 
                          approach_elbow, approach_pitch, approach_claw)) {
                if (waitForMotionComplete()) {
                    _currentState = PICK_APPROACH_TOF;
                    _stateStartTime = millis();
                }
            }
            break;
        }
        
        case PICK_APPROACH_TOF: {
            // Use ToF to fine-tune approach
            unsigned long now = millis();
            if (now - _lastToFRead > TOF_READ_INTERVAL) {
                int distance = _tofMgr->getDistance();
                _lastToFRead = now;
                
                if (!_tofMgr->isRangeValid(distance)) {
                    _lastError = "ToF out of range";
                    _currentState = PICK_ABORTED;
                    break;
                }
                
                // Check if we're at grasp distance
                if (distance <= _graspDistanceMm) {
                    _currentState = PICK_GRASP;
                    _stateStartTime = millis();
                } else if (distance > 200) { // Too far
                    _lastError = "Target too far";
                    _currentState = PICK_ABORTED;
                } else {
                    // Fine-tune approach: move slightly closer
                    // Adjust pitch down slightly
                    int current_pitch = _servoMgr->getAngle(4); // Pitch is index 4
                    if (current_pitch > 70) {
                        _servoMgr->setTarget(4, current_pitch - 2);
                    }
                }
            }
            
            // Timeout check
            if (millis() - _stateStartTime > 5000) {
                _lastError = "Approach timeout";
                _currentState = PICK_ABORTED;
            }
            break;
        }
        
        case PICK_GRASP: {
            // Close claw
            _servoMgr->setTarget(5, 0); // Claw closed (index 5)
            
            // Wait for claw to close
            delay(500); // Give time to grasp
            
            _currentState = PICK_LIFT;
            _stateStartTime = millis();
            break;
        }
        
        case PICK_LIFT: {
            // Lift by raising shoulder
            int current_shoulder = _servoMgr->getAngle(1);
            int lift_shoulder = constrain(current_shoulder + _liftHeightDeg, 15, 165);
            
            _servoMgr->setTarget(1, lift_shoulder);
            
            if (waitForMotionComplete()) {
                _currentState = PICK_MOVE_TO_BIN;
                _stateStartTime = millis();
            }
            break;
        }
        
        case PICK_MOVE_TO_BIN: {
            // Select bin based on class
            BinPose* binPose = (_targetClass == "ripe" || _targetClass == "tomato" || _targetClass == "ready") ? &_binRipe : &_binUnripe;
            
            if (moveToPose(binPose->base, binPose->shoulder, binPose->forearm,
                          binPose->elbow, binPose->pitch, binPose->claw)) {
                if (waitForMotionComplete()) {
                    _currentState = PICK_RELEASE;
                    _stateStartTime = millis();
                }
            }
            break;
        }
        
        case PICK_RELEASE: {
            // Open claw
            _servoMgr->setTarget(5, 90); // Claw open
            
            delay(300); // Wait for release
            
            _currentState = PICK_RETURN_HOME;
            _stateStartTime = millis();
            break;
        }
        
        case PICK_RETURN_HOME: {
            _servoMgr->home();
            
            if (waitForMotionComplete()) {
                _currentState = PICK_COMPLETE;
            }
            break;
        }
        
        default:
            break;
    }
}

void MotionPlanner::abort() {
    if (_currentState != PICK_IDLE && _currentState != PICK_COMPLETE) {
        _currentState = PICK_ABORTED;
        _servoMgr->emergencyStop();
    }
}

PickState MotionPlanner::getState() {
    return _currentState;
}

bool MotionPlanner::isPicking() {
    return _currentState != PICK_IDLE && _currentState != PICK_COMPLETE && _currentState != PICK_ABORTED;
}

String MotionPlanner::getLastError() {
    return _lastError;
}

void MotionPlanner::setApproachOffset(int mm) {
    _approachOffsetMm = mm;
}

void MotionPlanner::setGraspDistance(int mm) {
    _graspDistanceMm = mm;
}

void MotionPlanner::setLiftHeight(int degrees) {
    _liftHeightDeg = degrees;
}

void MotionPlanner::setBinPose(String bin_type, BinPose pose) {
    if (bin_type == "ripe") {
        _binRipe = pose;
    } else {
        _binUnripe = pose;
    }
}

void MotionPlanner::calculateTargetPose() {
    // This is a simplified calculation
    // In production, this would use inverse kinematics or a lookup table
    // For now, we use pixel X to determine base angle
    int base_angle = pixelToBaseAngle(_targetPixelX);
    
    // Store for later use
    // (In full implementation, would calculate all joint angles)
}

bool MotionPlanner::moveToPose(int base, int shoulder, int forearm, int elbow, int pitch, int claw) {
    return _servoMgr->setTargets(base, shoulder, forearm, elbow, pitch, claw);
}

bool MotionPlanner::waitForMotionComplete(unsigned long timeout_ms) {
    unsigned long start = millis();
    while (_servoMgr->isMoving()) {
        if (millis() - start > timeout_ms) {
            return false; // Timeout
        }
        delay(10);
    }
    return true;
}

int MotionPlanner::pixelToBaseAngle(int pixel_x) {
    // Simple linear mapping: 0-640 pixels -> 0-180 degrees
    // This should be replaced with calibration lookup table
    int angle = map(pixel_x, 0, 640, 0, 180);
    return constrain(angle, LIMIT_BASE_MIN, LIMIT_BASE_MAX);
}

int MotionPlanner::calculateApproachPose(int target_base, int target_shoulder) {
    // Calculate a safe approach pose offset from target
    // For now, just return a slightly different shoulder angle
    return constrain(target_shoulder - 20, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
}

