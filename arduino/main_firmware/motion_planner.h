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
    PICK_CALCULATE_POSE,
    PICK_MOVE_TO_APPROACH,
    PICK_WAIT_FOR_APPROACH,
    PICK_APPROACH_TOF,
    PICK_GRASP,
    PICK_WAIT_GRASP,
    PICK_LIFT,
    PICK_WAIT_FOR_LIFT,
    PICK_MOVE_TO_BIN,
    PICK_WAIT_FOR_BIN,
    PICK_RELEASE,
    PICK_WAIT_RELEASE,
    PICK_RETURN_HOME,
    PICK_WAIT_FOR_HOME,
    PICK_COMPLETE,
    PICK_ABORTED
};

class MotionPlanner {
public:
    MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr);
    
    // Pick sequence control
    bool startPick(int x, int y, int z, float confidence, String class_type);
    void update(); // Call in loop() for state machine
    void abort();
    
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
    
    // Approach parameters
    int _approachOffsetMm;
    int _graspDistanceMm;
    int _liftHeightDeg;
    
    // Bin poses
    BinPose _binRipe;
    BinPose _binUnripe;
    
    // State machine helpers
    void calculateTargetPose();
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

