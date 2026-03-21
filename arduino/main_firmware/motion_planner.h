#ifndef MOTION_PLANNER_H
#define MOTION_PLANNER_H

#include <Arduino.h>
#include "servo_manager.h"
#include "tof_vl53.h"
#include "config.h"

// Bin positions (servo angles)
struct BinPose {
    int waist;    // ID 0
    int shoulder; // ID 1
    int elbow;    // ID 2
    int wrist_roll; // ID 3
    int wrist_pitch; // ID 4
    int claw;     // ID 5
};

// Pick sequence states
enum PickState {
    PICK_IDLE,
    PICK_CALCULATE_POSE,    // Decides simulator vs AI mode
    PICK_START_FB,          // New Front-to-Back Start
    PICK_OPEN_CLAW,
    PICK_WAIT_OPEN_CLAW,
    PICK_MOVE_APPROACH,     // Front approach
    PICK_MOVE_DOWN,         // Front pick
    PICK_CLOSE_CLAW,
    PICK_GRASP = PICK_CLOSE_CLAW, // Alias for backward compatibility
    PICK_WAIT_CLOSE_CLAW,
    PICK_LIFT,              // Lift from front
    PICK_WAIT_LIFT,
    PICK_MOVE_TRANSIT,      // Move to safe rotation height
    PICK_MOVE_BACK_APPROACH, // Above back bin
    PICK_MOVE_PLACE_DOWN,   // At back floor
    PICK_RELEASE,           // Open to release
    PICK_WAIT_RELEASE,
    PICK_RETREAT,           // Up away from back
    PICK_GO_HOME,
    PICK_COMPLETE,
    PICK_ABORTED,
    PICK_ERROR
};

class MotionPlanner {
public:
    MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr);
    
    // Pick sequence control
    bool startPick(int x, int y, int z, float confidence, String class_type, bool isSimulation = false);
    void update(); // Call in loop() for state machine
    void abort();
    void reset();
    
    // Named movement helpers
    bool moveToPose(const BinPose& pose, int clawAngle);
    
    // Status
    PickState getState();
    bool isPicking();
    String getLastError();
    
    // Configuration
    void setApproachOffset(int mm); // Distance to stop before target (default 50mm)
    void setGraspDistance(int mm);  // ToF distance to trigger grasp (default 30mm)
    void setLiftHeight(int degrees); // How much to lift after grasp (default 20°)
    
    // Bin configuration
    void setBinPose(String bin_type, BinPose pose); // "ripe" or "unripe"
    
private:
    ServoManager* _servoMgr;
    ToFManager* _tofMgr;
    
    PickState _currentState;
    String _lastError;
    
    // Current pick parameters
    int _targetX;
    int _targetY;
    int _targetZ;
    String _targetClass;
    float _targetConfidence;
    String _pickId;
    bool _isSimulation;
    int _calculatedWaistAngle;
    int _targetShoulderAngle;
    int _targetElbowAngle;
    int _targetClawAngle;
    
    // Approach parameters
    int _approachOffsetMm;
    int _graspDistanceMm;
    int _liftHeightDeg;
    
    // Bin poses
    BinPose _binRipe;
    BinPose _binUnripe;
    BinPose _homePose;
    BinPose _frontApproachPose;
    BinPose _frontPickPose;
    BinPose _frontLiftPose;
    BinPose _transitPose;
    BinPose _backApproachPose;
    BinPose _backPlacePose;
    BinPose _backRetreatPose;
    
    // State machine helpers
    void calculateTargetPose();
    int widthToGripAngle(int width_mm);
    int widthToOpenAngle(int width_mm);
    bool moveToPose(int waist, int shoulder, int elbow, int wrist_roll, int wrist_pitch, int claw);
    bool waitForMotionComplete(unsigned long timeout_ms = 5000);
    int cartesianToBaseAngle(int x, int y);
    int calculateApproachPose(int target_waist, int target_shoulder);
    
    // Timing
    unsigned long _stateStartTime;
    unsigned long _lastToFRead;
    static const unsigned long TOF_READ_INTERVAL = 100; // ms
};

#endif // MOTION_PLANNER_H

