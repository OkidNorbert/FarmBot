#include "motion_planner.h"

MotionPlanner::MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr) {
    _servoMgr = servoMgr;
    _tofMgr = tofMgr;
    _currentState = PICK_IDLE;
    _approachOffsetMm = 50;  // Default 50mm approach offset
    _graspDistanceMm = 30;   // Default 30mm grasp distance
    _liftHeightDeg = 20;     // Default 20° lift
    
    // Default bin poses (will be calibrated)
    // Right bin (ready/ripe tomatoes) - use -1 for fixed waist/shoulder
    _binRipe.waist = -1;      // Use current waist angle (fixed)
    _binRipe.shoulder = -1;   // Use current shoulder angle (fixed)
    _binRipe.elbow = 90;
    _binRipe.wrist_roll = 110;
    _binRipe.wrist_pitch = 80;
    _binRipe.claw = 90;
    
    // Left bin (other categories) - use -1 for fixed waist/shoulder
    _binUnripe.waist = -1;      // Use current waist angle (fixed)
    _binUnripe.shoulder = -1;   // Use current shoulder angle (fixed)
    _binUnripe.elbow = 90;
    _binUnripe.wrist_roll = 110;
    _binUnripe.wrist_pitch = 80;
    _binUnripe.claw = 90;
}

bool MotionPlanner::startPick(int x, int y, int z, float confidence, String class_type) {
    if (_currentState != PICK_IDLE) {
        _lastError = "Already picking";
        return false;
    }
    
    // Removed strict ToF check here; handle gracefully in state machine
    
    _targetX = x;
    _targetY = y;
    _targetZ = z;
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
            int approach_waist = -1;      
            int approach_shoulder = -1;   
            int approach_elbow = 10;      
            int approach_wrist_roll = 90;
            int approach_wrist_pitch = 100; 
            int approach_claw = 90; 
            
            if (moveToPose(approach_waist, approach_shoulder, approach_elbow, 
                          approach_wrist_roll, approach_wrist_pitch, approach_claw)) {
                _currentState = PICK_WAIT_FOR_APPROACH;
                _stateStartTime = millis();
            }
            break;
        }

        case PICK_WAIT_FOR_APPROACH:
            if (!_servoMgr->isMoving()) {
                _currentState = PICK_APPROACH_TOF;
                _stateStartTime = millis();
            }
            break;
        
        case PICK_APPROACH_TOF: {
            unsigned long now = millis();
            if (now - _lastToFRead > TOF_READ_INTERVAL) {
                int distance = _tofMgr->getDistance();
                _lastToFRead = now;
                
                if (!_tofMgr->isRangeValid(distance) || distance <= _graspDistanceMm || distance > 300) {
                    _currentState = PICK_GRASP;
                    _stateStartTime = millis();
                } else {
                    int current_pitch = _servoMgr->getAngle(4);
                    _servoMgr->setTarget(4, current_pitch + 2); 
                }
            }
            
            if (millis() - _stateStartTime > 5000) {
                _currentState = PICK_GRASP;
                _stateStartTime = millis();
            }
            break;
        }
        
        case PICK_GRASP:
            _servoMgr->setTarget(5, CLAW_CLOSED_POSITION);
            _currentState = PICK_WAIT_GRASP;
            _stateStartTime = millis();
            break;

        case PICK_WAIT_GRASP:
            if (millis() - _stateStartTime > 500) {
                _currentState = PICK_LIFT;
                _stateStartTime = millis();
            }
            break;
        
        case PICK_LIFT: {
            int current_elbow = _servoMgr->getAngle(2);
            int lift_elbow = constrain(current_elbow - _liftHeightDeg, LIMIT_ELBOW_MIN, LIMIT_ELBOW_MAX);
            _servoMgr->setTarget(2, lift_elbow);
            _currentState = PICK_WAIT_FOR_LIFT;
            _stateStartTime = millis();
            break;
        }

        case PICK_WAIT_FOR_LIFT:
            if (!_servoMgr->isMoving()) {
                _currentState = PICK_MOVE_TO_BIN;
                _stateStartTime = millis();
            }
            break;
        
        case PICK_MOVE_TO_BIN: {
            BinPose* binPose = (_targetClass == "ripe" || _targetClass == "tomato" || _targetClass == "ready") ? &_binRipe : &_binUnripe;
            if (moveToPose(binPose->waist, binPose->shoulder, binPose->elbow,
                          binPose->wrist_roll, binPose->wrist_pitch, binPose->claw)) {
                _currentState = PICK_WAIT_FOR_BIN;
                _stateStartTime = millis();
            }
            break;
        }

        case PICK_WAIT_FOR_BIN:
            if (!_servoMgr->isMoving()) {
                _currentState = PICK_RELEASE;
                _stateStartTime = millis();
            }
            break;
        
        case PICK_RELEASE:
            _servoMgr->setTarget(5, 90);
            _currentState = PICK_WAIT_RELEASE;
            _stateStartTime = millis();
            break;

        case PICK_WAIT_RELEASE:
            if (millis() - _stateStartTime > 300) {
                _currentState = PICK_RETURN_HOME;
                _stateStartTime = millis();
            }
            break;
        
        case PICK_RETURN_HOME:
            _servoMgr->home();
            _currentState = PICK_WAIT_FOR_HOME;
            _stateStartTime = millis();
            break;

        case PICK_WAIT_FOR_HOME:
            if (!_servoMgr->isMoving()) {
                _currentState = PICK_COMPLETE;
            }
            break;
        
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
    // Calculate waist angle from cartesian coordinates
    int waist_angle = cartesianToBaseAngle(_targetX, _targetY);
    
    // Store for later use
    // (In full implementation, would calculate all joint angles including Z)
}

bool MotionPlanner::moveToPose(int waist, int shoulder, int elbow, int wrist_roll, int wrist_pitch, int claw) {
    // Handle -1 values (keep current angle) before calling setTargets
    // This allows fixed waist/shoulder to remain unchanged
    int waist_angle = (waist == -1) ? _servoMgr->getAngle(0) : waist;
    int shoulder_angle = (shoulder == -1) ? _servoMgr->getAngle(1) : shoulder;
    int elbow_angle = (elbow == -1) ? _servoMgr->getAngle(2) : elbow;
    int wrist_roll_angle = (wrist_roll == -1) ? _servoMgr->getAngle(3) : wrist_roll;
    int wrist_pitch_angle = (wrist_pitch == -1) ? _servoMgr->getAngle(4) : wrist_pitch;
    int claw_angle = (claw == -1) ? _servoMgr->getAngle(5) : claw;
    
    return _servoMgr->setTargets(waist_angle, shoulder_angle, elbow_angle, wrist_roll_angle, wrist_pitch_angle, claw_angle);
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

int MotionPlanner::cartesianToBaseAngle(int x, int y) {
    // Calculate angle using cartesian coordinates (atan2)
    // Forward (Y) is 90 degrees, Right (X) is 180 degrees, Left (-X) is 0 degrees
    float angle_rad = atan2((float)x, (float)y);
    int angle_deg = (int)(angle_rad * 180.0 / PI) + 90;
    return constrain(angle_deg, LIMIT_BASE_MIN, LIMIT_BASE_MAX);
}

int MotionPlanner::calculateApproachPose(int target_waist, int target_shoulder) {
    // Calculate a safe approach pose offset from target
    // For now, just return a slightly different shoulder angle
    return constrain(target_shoulder - 20, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
}

