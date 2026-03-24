#include "motion_planner.h"

MotionPlanner::MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr) {
    _servoMgr = servoMgr;
    _tofMgr = tofMgr;
    _currentState = PICK_IDLE;
    _approachOffsetMm = 50;
    _graspDistanceMm = 30;
    _liftHeightDeg = 20;

    // Defined from user feedback
    // Adjusted to fit within 0-180 degree limits
    _homePose          = {90, 90, 90, 90, 90, CLAW_CLOSED_POSITION};
    
    // Front Workspace (Center around 45 deg)
    _frontApproachPose = {45, 35, 90, 90, 30, 0};
    _frontPickPose     = {45, 22, 101, 90, 30, 0};
    _frontLiftPose     = {45, 45, 85, 90, 30, 0};

    // Transition (Neutral)
    _transitPose       = {90, 90, 80, 90, 110, 0};

    // Back Workspace (Center around 135 deg)
    _backApproachPose  = {135, 145, 95, 90, 155, 0};
    _backPlacePose     = {135, 158, 90, 90, 160, 0};
    _backRetreatPose   = {135, 145, 85, 90, 150, 0};

    // Bin mirroring (legacy support)
    _binRipe = _backPlacePose;
    _binUnripe = _backPlacePose;
    _binUnripe.waist = 170; // within 180 limit

    _calculatedWaistAngle = 90;
    _targetShoulderAngle = 90;
    _targetElbowAngle = 90;
    _targetClawAngle = CLAW_CLOSED_POSITION;
    _openClawAngle = LIMIT_CLAW_MIN;
    _pickId = "";
}

void MotionPlanner::updateClawTargets(int width_mm) {
    // FINAL USER PREFERENCE (2026-03-24):
    // Use exactamente 88 degrees for the firm grip and 30 for open.
    _targetClawAngle = 88; // User-selected "firm grip" angle
    _openClawAngle = 30;   // Measured "full open" angle

    Serial.print("[MP] Final Grip: In=");
    Serial.print(width_mm);
    Serial.print("mm -> Grip:");
    Serial.print(_targetClawAngle);
    Serial.print(" Open:");
    Serial.println(_openClawAngle);
}

// Legacy functions removed - use updateClawTargets() instead

bool MotionPlanner::startPick(int x, int y, int z, float confidence, String class_type, bool isSimulation, int object_height_mm) {
    // Allow starting if idle, complete, or aborted
    if (_currentState != PICK_IDLE && _currentState != PICK_COMPLETE && _currentState != PICK_ABORTED) {
        _lastError = "Already picking";
        return false;
    }
    
    _targetX = x;
    _targetY = y;
    _targetZ = z;
    _targetClass = class_type;
    _targetConfidence = confidence;
    _isSimulation = isSimulation;
    _pickId = String(millis());
    _objectHeightMm = constrain(object_height_mm, 5, 200); // Store object height
    
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
            // Calculate both grip and open angles once
            updateClawTargets(_targetZ);
            
            // Adjust pick and place depths based on object height.
            {
                int heightDelta = (_objectHeightMm - 50) / 5;
                heightDelta = constrain(heightDelta, -10, 15);
                
                _frontPickPose.shoulder      = constrain(22 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _frontApproachPose.shoulder  = constrain(35 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _frontLiftPose.shoulder      = constrain(45 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                
                _backPlacePose.shoulder      = constrain(158 - heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _backApproachPose.shoulder   = constrain(145 - heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
            }
            
            if (_isSimulation) {
                _currentState = PICK_START_FB;
            } else {
                // manual mode or auto mode with custom coords
                calculateTargetPose();
                _currentState = PICK_START_FB;
            }
            _stateStartTime = millis();
            break;

        case PICK_START_FB:
            // 1. Initial configuration: Open claw based on input width
            _servoMgr->setTarget(5, _openClawAngle);
            _currentState = PICK_MOVE_APPROACH;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_APPROACH:
            // Ensure claw has started opening before moving shoulder/elbow
            if (moveToPose(_frontApproachPose, _openClawAngle)) {
                _currentState = PICK_MOVE_DOWN;
                _stateStartTime = millis();
            }
            break;

        case PICK_MOVE_DOWN:
            if (moveToPose(_frontPickPose, _openClawAngle)) {
                _currentState = PICK_CLOSE_CLAW;
                _stateStartTime = millis();
            }
            break;

        case PICK_CLOSE_CLAW:
            // Ensure arm has reached the object before closing
            if (_servoMgr->isMoving()) return; 
            
            Serial.print("[MP] Trigerring GRIP -> ");
            Serial.println(_targetClawAngle);
            _servoMgr->setTarget(5, _targetClawAngle);
            _stateStartTime = millis();
            _currentState = PICK_WAIT_CLOSE_CLAW;
            break;

        case PICK_WAIT_CLOSE_CLAW:
            if (millis() - _stateStartTime > 1000) { // Slightly longer wait for firm grip
                _currentState = PICK_LIFT;
                _stateStartTime = millis();
            }
            break;

        case PICK_LIFT:
            if (moveToPose(_frontLiftPose, _targetClawAngle)) {
                _currentState = PICK_MOVE_TRANSIT;
                _stateStartTime = millis();
            }
            break;

        case PICK_MOVE_TRANSIT:
            if (_servoMgr->isMoving()) return;
            // Maintain pick pitch/claw while moving to back
            _servoMgr->setTargets(_transitPose.waist, _transitPose.shoulder, _transitPose.elbow, 
                                 _transitPose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
                                 
            _currentState = PICK_MOVE_BACK_APPROACH;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_BACK_APPROACH:
            if (_servoMgr->isMoving()) return;
            _servoMgr->setTargets(_backApproachPose.waist, _backApproachPose.shoulder, _backApproachPose.elbow, 
                                 _backApproachPose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
            _currentState = PICK_MOVE_PLACE_DOWN;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_PLACE_DOWN:
            if (_servoMgr->isMoving()) return;
            _servoMgr->setTargets(_backPlacePose.waist, _backPlacePose.shoulder, _backPlacePose.elbow, 
                                 _backPlacePose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
                                 
            _currentState = PICK_ADJUST_PITCH;
            _stateStartTime = millis();
            break;

        case PICK_ADJUST_PITCH:
            if (_servoMgr->isMoving()) return;
            if (millis() - _stateStartTime < 600) return; // Wait for arm to settle
            
            _servoMgr->setTarget(4, _backPlacePose.wrist_pitch);
            _currentState = PICK_RELEASE;
            _stateStartTime = millis();
            break;

        case PICK_RELEASE:
            if (_servoMgr->isMoving()) return;
            
            Serial.print("[MP] Trigerring RELEASE -> ");
            Serial.println(_openClawAngle);
            _servoMgr->setTarget(5, _openClawAngle);
            _stateStartTime = millis();
            _currentState = PICK_WAIT_RELEASE;
            break;

        case PICK_WAIT_RELEASE:
            if (millis() - _stateStartTime > 800) { // Wait for claw to fully open
                _currentState = PICK_RETREAT;
                _stateStartTime = millis();
            }
            break;

        case PICK_RETREAT:
            if (moveToPose(_backRetreatPose, _openClawAngle)) {
                _currentState = PICK_GO_HOME;
                _stateStartTime = millis();
            }
            break;

        case PICK_GO_HOME:
            if (moveToPose(_homePose, CLAW_CLOSED_POSITION)) {
                _currentState = PICK_COMPLETE;
            }
            break;

        default:
            break;
    }

    // Logging for state changes (helper for the switch above)
    static PickState lastLoggedState = PICK_IDLE;
    if (_currentState != lastLoggedState) {
        Serial.print("[MP] State: ");
        Serial.println(_stateToName(_currentState));
        lastLoggedState = _currentState;
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
    // Report true for any active state including final complete/aborted
    // so the main loop can process the result before we reset to IDLE.
    return _currentState != PICK_IDLE;
}

void MotionPlanner::reset() {
    _currentState = PICK_IDLE;
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
    // Calculate waist angle from cartesian coordinates (maps pixel X to base rotation)
    _calculatedWaistAngle = cartesianToBaseAngle(_targetX, _targetY);
    
    // Calculate reach distance (hypotenuse of X and Y)
    float reach = sqrt(sq((float)_targetX) + sq((float)_targetY));
    
    // Simple 2rd order Inverse Kinematics for Shoulder/Elbow
    // Segment lengths from STL: Brazo=150.0mm, Antebrazo=150.0mm 
    // Adjusted to total 300mm (30cm) to match user's physical horizontal stretch
    float L1 = 150.0;
    float L2 = 150.0;
    
    // Calculate Elbow angle using Law of Cosines
    float cos_elbow = (sq(reach) - sq(L1) - sq(L2)) / (2.0 * L1 * L2);
    cos_elbow = constrain(cos_elbow, -1.0, 1.0);
    float rad_elbow = acos(cos_elbow);
    int elbow_deg = (int)(rad_elbow * 180.0 / PI);
    
    // Calculate Shoulder angle
    float rad_shoulder = atan2((float)_targetY, (float)_targetX) - 
                        atan2(L2 * sin(rad_elbow), L1 + L2 * cos_elbow);
    int shoulder_deg = (int)(rad_shoulder * 180.0 / PI);

    // Map to servo space
    // Front simulation override: 30cm target
    if (reach > 290 && reach < 310 && _targetX == 0) {
        _calculatedWaistAngle = 90;
        _targetShoulderAngle = 27;
        _targetElbowAngle = 101;
    } else {
        _targetShoulderAngle = constrain(96 - shoulder_deg, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
        _targetElbowAngle = constrain(180 - elbow_deg, LIMIT_ELBOW_MIN, LIMIT_ELBOW_MAX);
    }

    Serial.print("Target Calculated: Reach=");
    Serial.print(reach);
    Serial.print("mm -> Waist:");
    Serial.print(_calculatedWaistAngle);
    Serial.print(" Sh:");
    Serial.print(_targetShoulderAngle);
    Serial.print(" El:");
    Serial.print(_targetElbowAngle);
    Serial.print(" Claw(unchanged):");
    Serial.println(_targetClawAngle);
}

bool MotionPlanner::moveToPose(const BinPose& pose, int clawAngle) {
    if (_servoMgr->isMoving()) return false;
    
    _servoMgr->setTargets(pose.waist, pose.shoulder, pose.elbow, 
                        pose.wrist_roll, pose.wrist_pitch, clawAngle);
    return true;
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

String MotionPlanner::_stateToName(PickState state) {
    switch (state) {
        case PICK_IDLE: return "IDLE";
        case PICK_CALCULATE_POSE: return "CALC_POSE";
        case PICK_START_FB: return "START_FB";
        case PICK_MOVE_APPROACH: return "MOVE_APPROACH";
        case PICK_MOVE_DOWN: return "MOVE_DOWN";
        case PICK_CLOSE_CLAW: return "CLOSE_CLAW";
        case PICK_WAIT_CLOSE_CLAW: return "WAIT_CLOSE";
        case PICK_LIFT: return "LIFT";
        case PICK_MOVE_TRANSIT: return "MOVE_TRANSIT";
        case PICK_MOVE_BACK_APPROACH: return "MOVE_BACK_APP";
        case PICK_MOVE_PLACE_DOWN: return "MOVE_PLACE_DOWN";
        case PICK_ADJUST_PITCH: return "ADJUST_PITCH";
        case PICK_RELEASE: return "RELEASE";
        case PICK_WAIT_RELEASE: return "WAIT_RELEASE";
        case PICK_RETREAT: return "RETREAT";
        case PICK_GO_HOME: return "GO_HOME";
        case PICK_COMPLETE: return "COMPLETE";
        case PICK_ABORTED: return "ABORTED";
        case PICK_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

