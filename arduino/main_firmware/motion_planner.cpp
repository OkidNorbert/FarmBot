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
    _pickId = "";
}

int MotionPlanner::widthToGripAngle(int width_mm) {
    // Reverted to hardware limit 110 as per user configuration
    // We target a width 6mm smaller than actual to "push" for a tighter grip
    // using the 3mm clearance on each side of the gripper to absorb excessive grip.
    int target_width = width_mm - 6; 
    
    // Mapping: 10mm gap is at 110 deg (max), 70mm gap is at 35 deg (open)
    int angle = map(target_width, 10, 70, 110, 35);
    
    return constrain(angle, LIMIT_CLAW_MIN, LIMIT_CLAW_MAX);
}

int MotionPlanner::widthToOpenAngle(int width_mm) {
    int grip = widthToGripAngle(width_mm);
    // Ensure we open at least 25 degrees or to the hard limit
    int open = grip - 30;
    if (open < LIMIT_CLAW_MIN) open = LIMIT_CLAW_MIN;
    return open;
}

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
            _targetClawAngle = widthToGripAngle(_targetZ);
            
            // Adjust pick and place depths based on object height.
            // Reference height is 50mm. Each 10mm taller = 2° higher shoulder.
            // Taller object: shoulder goes UP (higher angle = arm draws back = gripper higher).
            {
                int heightDelta = (_objectHeightMm - 50) / 5; // degrees per 5mm height
                heightDelta = constrain(heightDelta, -10, 15);  // cap at safe range
                
                // Apply to pick pose: taller -> higher pick shoulder
                _frontPickPose.shoulder      = constrain(22 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _frontApproachPose.shoulder  = constrain(35 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _frontLiftPose.shoulder      = constrain(45 + heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                
                // Apply to place pose: taller -> higher place shoulder (less lean forward)
                _backPlacePose.shoulder      = constrain(158 - heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
                _backApproachPose.shoulder   = constrain(145 - heightDelta, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX);
            }
            
            if (_isSimulation) {
                _currentState = PICK_START_FB;
            } else {
                calculateTargetPose();
                _currentState = PICK_START_FB;
            }
            _stateStartTime = millis();
            break;

        case PICK_START_FB:
            // 1. Initial configuration: Open claw based on input width
            _servoMgr->setTarget(5, widthToOpenAngle(_targetZ));
            _currentState = PICK_MOVE_APPROACH;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_APPROACH:
            if (moveToPose(_frontApproachPose, widthToOpenAngle(_targetZ))) {
                _currentState = PICK_MOVE_DOWN;
                _stateStartTime = millis();
            }
            break;

        case PICK_MOVE_DOWN:
            if (moveToPose(_frontPickPose, widthToOpenAngle(_targetZ))) {
                _currentState = PICK_CLOSE_CLAW;
                _stateStartTime = millis();
            }
            break;

        case PICK_CLOSE_CLAW:
            // Ensure arm has reached the object before closing
            if (_servoMgr->isMoving()) return; 
            
            _servoMgr->setTarget(5, _targetClawAngle);
            _stateStartTime = millis();
            _currentState = PICK_WAIT_CLOSE_CLAW;
            break;

        case PICK_WAIT_CLOSE_CLAW:
            if (millis() - _stateStartTime > 800) { // Increased wait for firm grasp
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
            // Move through transit position but KEEP the picking pitch (usually 30°)
            // to ensure the gripper orientation doesn't tilt the object yet.
            if (_servoMgr->isMoving()) return;
            _servoMgr->setTargets(_transitPose.waist, _transitPose.shoulder, _transitPose.elbow, 
                                 _transitPose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
                                 
            _currentState = PICK_MOVE_BACK_APPROACH;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_BACK_APPROACH:
            // Ensure TRANSIT motion is finished
            if (_servoMgr->isMoving()) return;
            
            // Move to back approach position still KEEPING the picking pitch (30°).
            _servoMgr->setTargets(_backApproachPose.waist, _backApproachPose.shoulder, _backApproachPose.elbow, 
                                 _backApproachPose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
            _currentState = PICK_MOVE_PLACE_DOWN;
            _stateStartTime = millis();
            break;

        case PICK_MOVE_PLACE_DOWN:
            // Ensure BACK_APPROACH motion is finished
            if (_servoMgr->isMoving()) return;
            
            // Reach the final floor position, still MAINTAINING the picking pitch (30°).
            _servoMgr->setTargets(_backPlacePose.waist, _backPlacePose.shoulder, _backPlacePose.elbow, 
                                 _backPlacePose.wrist_roll, _frontPickPose.wrist_pitch, _targetClawAngle);
                                 
            _currentState = PICK_ADJUST_PITCH;
            _stateStartTime = millis();
            break;

        case PICK_ADJUST_PITCH:
            // Ensure arm has settled at the final floor position
            if (_servoMgr->isMoving()) return;
            
            // Settlement delay for careful handling
            if (millis() - _stateStartTime < 400) return; 
            
            // Now finally perform the pitch adjustment to the release angle (e.g., 160°)
            _servoMgr->setTarget(4, _backPlacePose.wrist_pitch);
            _currentState = PICK_RELEASE;
            _stateStartTime = millis();
            break;

        case PICK_RELEASE:
            // CRITICAL: Ensure arm has reached the floor/bin before opening
            if (_servoMgr->isMoving()) return;
            
            _servoMgr->setTarget(5, widthToOpenAngle(_targetZ));
            _stateStartTime = millis();
            _currentState = PICK_WAIT_RELEASE;
            break;

        case PICK_WAIT_RELEASE:
            if (millis() - _stateStartTime > 500) {
                _currentState = PICK_RETREAT;
                _stateStartTime = millis();
            }
            break;

        case PICK_RETREAT:
            if (moveToPose(_backRetreatPose, widthToOpenAngle(_targetZ))) {
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
    
    // Simple linear mapping: 0mm -> 110deg, 80mm -> 30deg
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

