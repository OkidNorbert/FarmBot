#include "motion_planner.h"

MotionPlanner::MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr) {
    _servoMgr = servoMgr;
    _tofMgr = tofMgr;
    _currentState = PICK_IDLE;
    _approachOffsetMm = 50;
    _graspDistanceMm = 30;
    _liftHeightDeg = 20;

    // 0. Home Pose (Stable default)
    _homePose = {90, 90, 90, 90, 90, CLAW_CLOSED_POSITION};

    // 1. Front Approach (Above object)
    _frontApproachPose = {90, 45, 101, 98, 45, 30};

    // 2. Front Pick (Calibrated Touch Point)
    _frontPickPose = {90, 22, 101, 98, 30, 30};

    // 3. Front Lift (Safe Height above front)
    _frontLiftPose = {90, 60, 101, 98, 90, 30};

    // 4. Transit Pose (Centered for rotation)
    _transitPose = {90, 90, 90, 98, 90, 30};

    // 5. Back Approach (Above Bin)
    _backApproachPose = {90, 130, 90, 98, 120, 30};

    // 6. Back Place (Calibrated Drop Point)
    _backPlacePose = {90, 165, 90, 98, 160, 30};

    // 7. Back Retreat (Safe Height above back)
    _backRetreatPose = {90, 130, 90, 98, 120, 30};

    // Mirroring for Unripe Bin
    _binRipe = _backPlacePose;
    _binUnripe = _backPlacePose;
    _binUnripe.waist = 140;

    _calculatedWaistAngle = 90;
    _targetShoulderAngle = 90;
    _targetElbowAngle = 90;
    _targetClawAngle = CLAW_CLOSED_POSITION;
    _pickId = "";
}

int MotionPlanner::widthToGripAngle(int width_mm) {
    // 0mm -> 115deg (Closed), 80mm -> 30deg (Open)
    width_mm = constrain(width_mm, 0, 80);
    return map(width_mm, 0, 80, LIMIT_CLAW_MAX, LIMIT_CLAW_MIN);
}

int MotionPlanner::widthToOpenAngle(int width_mm) {
    int grip = widthToGripAngle(width_mm);
    return constrain(grip - 25, LIMIT_CLAW_MIN, LIMIT_CLAW_MAX); // Lower is more open
}

bool MotionPlanner::startPick(int x, int y, int z, float confidence, String class_type) {
    // Allow starting if idle, complete, or aborted
    if (_currentState != PICK_IDLE && _currentState != PICK_COMPLETE && _currentState != PICK_ABORTED) {
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
            _targetClawAngle = widthToGripAngle(_targetZ);
            _currentState = PICK_START_FB;
            _stateStartTime = millis();
            break;

        case PICK_START_FB:
            // 1. Initial configuration: Open claw and move to front approach
            _servoMgr->setTarget(5, widthToOpenAngle(_targetZ));
            if (moveToPose(_frontApproachPose.waist, _frontApproachPose.shoulder, _frontApproachPose.elbow,
                          _frontApproachPose.wrist_roll, _frontApproachPose.wrist_pitch, -1)) {
                _currentState = PICK_MOVE_APPROACH;
                _stateStartTime = millis();
            }
            break;

        case PICK_MOVE_APPROACH:
            if (millis() - _stateStartTime > 200 && !_servoMgr->isMoving()) {
                _currentState = PICK_MOVE_DOWN;
                _stateStartTime = millis();
            }
            break;

        case PICK_MOVE_DOWN:
            if (moveToPose(_frontPickPose.waist, _frontPickPose.shoulder, _frontPickPose.elbow,
                          _frontPickPose.wrist_roll, _frontPickPose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_CLOSE_CLAW;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_CLOSE_CLAW:
            _servoMgr->setTarget(5, _targetClawAngle);
            _currentState = PICK_WAIT_CLOSE_CLAW;
            _stateStartTime = millis();
            break;

        case PICK_WAIT_CLOSE_CLAW:
            if (millis() - _stateStartTime > 600) { // Time to squeeze
                _currentState = PICK_LIFT;
                _stateStartTime = millis();
            }
            break;

        case PICK_LIFT:
            if (moveToPose(_frontLiftPose.waist, _frontLiftPose.shoulder, _frontLiftPose.elbow,
                          _frontLiftPose.wrist_roll, _frontLiftPose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_MOVE_TRANSIT;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_MOVE_TRANSIT:
            if (moveToPose(_transitPose.waist, _transitPose.shoulder, _transitPose.elbow,
                          _transitPose.wrist_roll, _transitPose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_MOVE_BACK_APPROACH;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_MOVE_BACK_APPROACH:
            if (moveToPose(_backApproachPose.waist, _backApproachPose.shoulder, _backApproachPose.elbow,
                          _backApproachPose.wrist_roll, _backApproachPose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_MOVE_PLACE_DOWN;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_MOVE_PLACE_DOWN:
            if (moveToPose(_backPlacePose.waist, _backPlacePose.shoulder, _backPlacePose.elbow,
                          _backPlacePose.wrist_roll, _backPlacePose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_RELEASE;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_RELEASE:
            _servoMgr->setTarget(5, widthToOpenAngle(_targetZ));
            _currentState = PICK_WAIT_RELEASE;
            _stateStartTime = millis();
            break;

        case PICK_WAIT_RELEASE:
            if (millis() - _stateStartTime > 500) {
                _currentState = PICK_RETREAT;
                _stateStartTime = millis();
            }
            break;

        case PICK_RETREAT:
            if (moveToPose(_backRetreatPose.waist, _backRetreatPose.shoulder, _backRetreatPose.elbow,
                          _backRetreatPose.wrist_roll, _backRetreatPose.wrist_pitch, -1)) {
                if (!_servoMgr->isMoving() && millis() - _stateStartTime > 200) {
                    _currentState = PICK_GO_HOME;
                    _stateStartTime = millis();
                }
            }
            break;

        case PICK_GO_HOME:
            _servoMgr->home();
            _currentState = PICK_GO_HOME + 1; // Generic wait
            _stateStartTime = millis();
            break;

        case (PICK_GO_HOME + 1):
            if (!_servoMgr->isMoving() && millis() - _stateStartTime > 500) {
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

    // Calculate dynamic claw angle based on target width (_targetZ)
    // Map target size (typically 0-100mm) to claw range (LIMIT_CLAW_MIN to LIMIT_CLAW_MAX)
    // Larger objects = Smaller Angle (More Open), Smaller objects = Larger Angle (More Closed)
    // We assume 3.5cm (35mm) is the standard target.
    // Full Open (30 deg) -> ~80mm width, Full Closed (115 deg) -> 0mm width
    int objectWidth = _targetZ; 
    if (objectWidth < 5) objectWidth = 35; // Fallback to 3.5cm if invalid
    
    // Simple linear mapping: 0mm -> 115deg, 80mm -> 30deg
    float claw_mapped = map(objectWidth, 0, 80, LIMIT_CLAW_MAX, LIMIT_CLAW_MIN);
    _targetClawAngle = constrain((int)claw_mapped, LIMIT_CLAW_MIN, LIMIT_CLAW_MAX);

    Serial.print("Target Calculated: Reach=");
    Serial.print(reach);
    Serial.print("mm -> Waist:");
    Serial.print(_calculatedWaistAngle);
    Serial.print(" Sh:");
    Serial.print(_targetShoulderAngle);
    Serial.print(" El:");
    Serial.print(_targetElbowAngle);
    Serial.print(" Claw:");
    Serial.println(_targetClawAngle);
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

