#ifndef MOTION_PLANNER_H
#define MOTION_PLANNER_H

#include <Arduino.h>
#include "servo_manager.h"
#include "tof_vl53.h"
#include "config.h"

// Bin positions (servo angles)
struct BinPose {
    int base;
    int shoulder;
    int forearm;
    int elbow;
    int pitch;
    int claw;
};

// Pick sequence states
enum PickState {
    PICK_IDLE,
    PICK_CALCULATE_POSE,
    PICK_MOVE_TO_APPROACH,
    PICK_APPROACH_TOF,
    PICK_GRASP,
    PICK_LIFT,
    PICK_MOVE_TO_BIN,
    PICK_RELEASE,
    PICK_RETURN_HOME,
    PICK_COMPLETE,
    PICK_ABORTED
};

class MotionPlanner {
public:
    MotionPlanner(ServoManager* servoMgr, ToFManager* tofMgr);
    
    // Pick sequence control
    bool startPick(int pixel_x, int pixel_y, float confidence, String class_type);
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
    int _targetPixelX;
    int _targetPixelY;
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
    bool moveToPose(int base, int shoulder, int forearm, int elbow, int pitch, int claw);
    bool waitForMotionComplete(unsigned long timeout_ms = 5000);
    int pixelToBaseAngle(int pixel_x); // Simple mapping (needs calibration)
    int calculateApproachPose(int target_base, int target_shoulder);
    
    // Timing
    unsigned long _stateStartTime;
    unsigned long _lastToFRead;
    static const unsigned long TOF_READ_INTERVAL = 100; // ms
};

#endif // MOTION_PLANNER_H

