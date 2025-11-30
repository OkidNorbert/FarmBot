#include <Arduino.h>
#include <ArduinoJson.h>
#include "config.h"
#include "servo_manager.h"
#include "tof_vl53.h"
#include "comm_client.h"
#include "motion_planner.h"
#include "calibration.h"

// Managers
ServoManager servoManager;
ToFManager tofManager;
CommClient commClient;
MotionPlanner motionPlanner(&servoManager, &tofManager);
CalibrationManager calibrationManager;

// State
enum SystemState {
    IDLE,
    MOVING,
    PICKING,
    CALIBRATING,
    ERROR
};

SystemState currentState = IDLE;
unsigned long lastTelemetry = 0;
String currentPickId = "";
unsigned long pickStartTime = 0;
String currentMode = "MANUAL";  // Track current mode: "MANUAL" or "AUTO"

// Forward Declarations
void handleWebSocketMessage(String payload);
void processTextCommand(String command);  // Handle text-based commands (backward compatibility)
void executePick(String id, int x, int y, String type, float confidence);
void sendStatus();
void checkEmergencyStop();

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("========================================");
    Serial.println("FarmBot Tomato Picker - UNO R4 WiFi");
    Serial.println("========================================");
    
    // Initialize Emergency Stop Pin
    pinMode(PIN_EMERGENCY_STOP, INPUT_PULLUP);
    
    // Load Calibration
    if (calibrationManager.load()) {
        Serial.println("Calibration loaded from EEPROM");
    } else {
        Serial.println("Using default calibration");
        calibrationManager.save(); // Save defaults
    }
    
    // Initialize Components
    Serial.println("Initializing servos...");
    servoManager.begin();
    
    // Initialize to manual mode with default speed
    currentMode = "MANUAL";
    servoManager.setSpeed(DEFAULT_SPEED);
    Serial.print("Initialized in MANUAL mode - Speed: ");
    Serial.print(DEFAULT_SPEED);
    Serial.print(" deg/s (max: ");
    Serial.print(MANUAL_MODE_MAX);
    Serial.println(" deg/s)");
    
    Serial.println("Initializing ToF sensor...");
    if (!tofManager.begin()) {
        Serial.println("WARNING: ToF Sensor Failed!");
    } else {
        Serial.println("ToF sensor ready");
    }
    
    // Configure Motion Planner with calibration data
    int16_t bin_ripe[6], bin_unripe[6];
    calibrationManager.getBinPose("ripe", bin_ripe);
    calibrationManager.getBinPose("unripe", bin_unripe);
    
    BinPose ripePose;
    ripePose.base = bin_ripe[0];
    ripePose.shoulder = bin_ripe[1];
    ripePose.forearm = bin_ripe[2];
    ripePose.elbow = bin_ripe[3];
    ripePose.pitch = bin_ripe[4];
    ripePose.claw = bin_ripe[5];
    
    BinPose unripePose;
    unripePose.base = bin_unripe[0];
    unripePose.shoulder = bin_unripe[1];
    unripePose.forearm = bin_unripe[2];
    unripePose.elbow = bin_unripe[3];
    unripePose.pitch = bin_unripe[4];
    unripePose.claw = bin_unripe[5];
    
    motionPlanner.setBinPose("ripe", ripePose);
    motionPlanner.setBinPose("unripe", unripePose);
    
    Serial.println("Connecting to communication...");
    commClient.begin();
    commClient.onMessage(handleWebSocketMessage);
    
    Serial.println("========================================");
    Serial.print("Connection Type: ");
    Serial.println(commClient.getConnectionType());
    Serial.println("System Ready!");
    Serial.println("Waiting for commands...");
    Serial.println("========================================");
}

void loop() {
    // Check Emergency Stop
    checkEmergencyStop();
    
    // Update Managers
    servoManager.update();
    commClient.update();
    motionPlanner.update(); // Update pick sequence state machine
    
    // Handle Pick Sequence State Machine
    if (motionPlanner.isPicking()) {
        currentState = PICKING;
        
        PickState pickState = motionPlanner.getState();
        
        if (pickState == PICK_COMPLETE) {
            // Pick completed successfully
            unsigned long duration = millis() - pickStartTime;
            commClient.sendPickResult(currentPickId.c_str(), "SUCCESS", "ripe", duration);
            currentState = IDLE;
            currentPickId = "";
        } else if (pickState == PICK_ABORTED) {
            // Pick aborted
            unsigned long duration = millis() - pickStartTime;
            commClient.sendPickResult(currentPickId.c_str(), "ABORTED", "none", duration);
            currentState = IDLE;
            currentPickId = "";
        }
    } else if (currentState == PICKING) {
        // Pick finished but state wasn't updated
        currentState = IDLE;
    }
    
    // Update system state based on servo movement
    if (servoManager.isMoving() && currentState == IDLE) {
        currentState = MOVING;
    } else if (!servoManager.isMoving() && currentState == MOVING) {
        currentState = IDLE;
    }
    
    // Telemetry Loop (every 2 seconds)
    if (millis() - lastTelemetry > 2000) {
        sendStatus();
        lastTelemetry = millis();
    }
    
    // Small delay to prevent watchdog issues
    delay(10);
}

void handleWebSocketMessage(String payload) {
    Serial.print("Received: ");
    Serial.println(payload);
    
    // Check if payload looks like a text command (starts with uppercase word)
    // Text commands: MOVE, PICK, HOME, STOP, ANGLE, STATUS, etc.
    payload.trim();
    bool isTextCommand = false;
    if (payload.length() > 0) {
        char firstChar = payload.charAt(0);
        // Text commands start with uppercase letters
        if (firstChar >= 'A' && firstChar <= 'Z') {
            // Check if it's a known text command
            if (payload.startsWith("MOVE") || payload.startsWith("PICK") || 
                payload.startsWith("HOME") || payload.startsWith("STOP") ||
                payload.startsWith("ANGLE") || payload.startsWith("STATUS") ||
                payload.startsWith("GRIP") || payload.startsWith("DISTANCE") ||
                payload.startsWith("SPEED")) {
                isTextCommand = true;
            }
        }
    }
    
    if (isTextCommand) {
        // Directly process as text command (skip JSON parsing)
        processTextCommand(payload);
        return;
    }
    
    // Try to parse as JSON
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) {
        // JSON parsing failed - try parsing as text command (backward compatibility)
        Serial.print("JSON Parse Failed, trying text command: ");
        Serial.println(error.c_str());
        processTextCommand(payload);
        return;
    }
    
    // Check if Socket.IO format (array) or direct JSON (object)
    JsonObject data;
    if (doc.is<JsonArray>()) {
        // Socket.IO format: ["event", {...}]
        JsonArray arr = doc.as<JsonArray>();
        String event = arr[0].as<String>();
        data = arr[1].as<JsonObject>();
        
        if (event != "command") {
            Serial.print("Unknown event: ");
            Serial.println(event);
            return;
        }
    } else if (doc.is<JsonObject>()) {
        // Direct JSON format (from BLE)
        data = doc.as<JsonObject>();
    } else {
        Serial.println("Invalid JSON format");
        return;
    }
    
    // Process command
    if (!data.containsKey("cmd")) {
        Serial.println("No 'cmd' field in message");
        return;
    }
    
    String cmd = data["cmd"].as<String>();
    
    if (cmd == "pick") {
            // Automatic mode - ensure speed is set to auto mode speed
            if (currentMode != "AUTO" && currentMode != "AUTOMATIC") {
                currentMode = "AUTO";
                servoManager.setSpeed(AUTO_MODE_SPEED);
                Serial.println("Switched to AUTO mode for pick operation");
            }
            
            String id = data["id"] | "unknown";
            int x = data["x"] | 320;
            int y = data["y"] | 240;
            String type = data["class"] | "ripe";
            float confidence = data["confidence"] | 0.0;
            executePick(id, x, y, type, confidence);
        }
        else if (cmd == "move_joints") {
            servoManager.setTargets(
                data["base"] | 90,
                data["shoulder"] | 90,
                data["forearm"] | 90,
                data["elbow"] | 90,
                data["pitch"] | 90,
                data["claw"] | 0
            );
        }
        else if (cmd == "move") {
            // Individual servo movement with optional speed
            String servo = data["servo"].as<String>();
            int angle = data["angle"] | 90;
            int speed = data["speed"] | 0;  // 0 means use current speed
            
            // Map servo names to IDs: base=0, shoulder=1, forearm=2, elbow=3, pitch=4, claw=5
            if (servo == "base") {
                servoManager.setTarget(0, angle);
            } else if (servo == "shoulder" || servo == "arm") {
                servoManager.setTarget(1, angle);
            } else if (servo == "forearm") {
                servoManager.setTarget(2, angle);
            } else if (servo == "elbow" || servo == "wrist_yaw") {
                servoManager.setTarget(3, angle);
            } else if (servo == "pitch" || servo == "wrist_pitch") {
                servoManager.setTarget(4, angle);
            } else if (servo == "claw") {
                servoManager.setTarget(5, angle);
            } else {
                Serial.print("Unknown servo name: ");
                Serial.println(servo);
                return;
            }
            
            // Set speed if provided, with mode-based limits
            if (speed > 0) {
                int maxAllowedSpeed = (currentMode == "AUTO" || currentMode == "AUTOMATIC") 
                                      ? AUTO_MODE_SPEED 
                                      : MANUAL_MODE_MAX;
                int constrainedSpeed = constrain(speed, MIN_SPEED, maxAllowedSpeed);
                servoManager.setSpeed(constrainedSpeed);
                
                if (speed != constrainedSpeed) {
                    Serial.print("Speed constrained to ");
                    Serial.print(constrainedSpeed);
                    Serial.print(" deg/s (");
                    Serial.print(currentMode);
                    Serial.println(" mode limit)");
                }
            }
            
            Serial.print("Servo ");
            Serial.print(servo);
            Serial.print(" set to ");
            Serial.print(angle);
            Serial.println("°");
        }
        else if (cmd == "set_speed") {
            int speed = data["speed"] | DEFAULT_SPEED;
            
            // Apply mode-based speed limits
            int maxAllowedSpeed = (currentMode == "AUTO" || currentMode == "AUTOMATIC") 
                                  ? AUTO_MODE_SPEED 
                                  : MANUAL_MODE_MAX;
            int constrainedSpeed = constrain(speed, MIN_SPEED, maxAllowedSpeed);
            servoManager.setSpeed(constrainedSpeed);
            
            if (speed != constrainedSpeed) {
                Serial.print("Speed constrained from ");
                Serial.print(speed);
                Serial.print(" to ");
                Serial.print(constrainedSpeed);
                Serial.print(" deg/s (");
                Serial.print(currentMode);
                Serial.println(" mode limit)");
            } else {
                Serial.print("Speed set to: ");
                Serial.print(constrainedSpeed);
                Serial.println(" deg/s");
            }
        }
        else if (cmd == "home") {
            servoManager.home();
        }
        else if (cmd == "stop") {
            motionPlanner.abort();
            servoManager.emergencyStop();
            currentState = ERROR;
        }
        else if (cmd == "set_mode") {
            String mode = data["mode"] | "AUTO";
            mode.toUpperCase();
            currentMode = mode;
            
            // Set appropriate speed based on mode
            if (mode == "AUTO" || mode == "AUTOMATIC") {
                servoManager.setSpeed(AUTO_MODE_SPEED);
                Serial.print("Mode set to: AUTO - Speed set to ");
                Serial.print(AUTO_MODE_SPEED);
                Serial.println(" deg/s (optimized for AI/camera/sensor coordination)");
            } else {
                // Manual mode - allow up to max speed, keep current speed if already set
                int currentSpeed = servoManager.getSpeed();
                if (currentSpeed > MANUAL_MODE_MAX) {
                    servoManager.setSpeed(MANUAL_MODE_MAX);
                    Serial.print("Mode set to: MANUAL - Speed limited to ");
                    Serial.print(MANUAL_MODE_MAX);
                    Serial.println(" deg/s");
                } else {
                    Serial.print("Mode set to: MANUAL - Current speed: ");
                    Serial.print(currentSpeed);
                    Serial.println(" deg/s (max: 120 deg/s)");
                }
            }
        }
        else if (cmd == "calibrate") {
            // Enter calibration mode
            currentState = CALIBRATING;
            Serial.println("Entering calibration mode");
        }
    else {
        Serial.print("Unknown command: ");
        Serial.println(cmd);
    }
}

void processTextCommand(String command) {
    // Handle text-based commands for backward compatibility with old web interface
    command.trim();
    
    if (command.startsWith("MOVE")) {
        // MOVE X Y Z or MOVE X Y CLASS - Move to world coordinates
        int firstSpace = command.indexOf(' ');
        int secondSpace = command.indexOf(' ', firstSpace + 1);
        int thirdSpace = command.indexOf(' ', secondSpace + 1);
        
        if (firstSpace == -1 || secondSpace == -1) {
            Serial.println("Invalid MOVE command format - need at least X Y");
            return;
        }
        
        float x = command.substring(firstSpace + 1, secondSpace).toFloat();
        float y = command.substring(secondSpace + 1, (thirdSpace == -1 ? command.length() : thirdSpace)).toFloat();
        
        // Optional third parameter (Z or CLASS)
        int class_id = 0;  // Default to unripe
        if (thirdSpace != -1) {
            class_id = command.substring(thirdSpace + 1).toInt();
        }
        
        Serial.print("MOVE command: X=");
        Serial.print(x);
        Serial.print(", Y=");
        Serial.print(y);
        Serial.print(", Class=");
        Serial.println(class_id);
        
        // Convert to pick command (using motion planner)
        String type = (class_id == 1) ? "ripe" : "unripe";
        executePick("move_" + String(millis()), (int)x, (int)y, type, 1.0);
    }
    else if (command.startsWith("PICK")) {
        // PICK X Y Z CLASS - Pick from coordinates
        int firstSpace = command.indexOf(' ');
        int secondSpace = command.indexOf(' ', firstSpace + 1);
        int thirdSpace = command.indexOf(' ', secondSpace + 1);
        int fourthSpace = command.indexOf(' ', thirdSpace + 1);
        
        if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1 || fourthSpace == -1) {
            Serial.println("Invalid PICK command format");
            return;
        }
        
        float x = command.substring(firstSpace + 1, secondSpace).toFloat();
        float y = command.substring(secondSpace + 1, thirdSpace).toFloat();
        float z = command.substring(thirdSpace + 1, fourthSpace).toFloat();
        int class_id = command.substring(fourthSpace + 1).toInt();
        
        Serial.print("PICK command: X=");
        Serial.print(x);
        Serial.print(", Y=");
        Serial.print(y);
        Serial.print(", Z=");
        Serial.print(z);
        Serial.print(", Class=");
        Serial.println(class_id);
        
        // Convert to pick command
        String type = (class_id == 1) ? "ripe" : "unripe";
        executePick("pick_" + String(millis()), (int)x, (int)y, type, 1.0);
    }
    else if (command.startsWith("HOME")) {
        Serial.println("HOME command received");
        servoManager.home();
    }
    else if (command.startsWith("STOP")) {
        Serial.println("STOP command received");
        motionPlanner.abort();
        servoManager.emergencyStop();
        currentState = ERROR;
    }
    else if (command.startsWith("SPEED")) {
        // SPEED <value> - Set movement speed in degrees per second
        int firstSpace = command.indexOf(' ');
        if (firstSpace == -1) {
            Serial.println("Invalid SPEED command format - need value");
            return;
        }
        
        int speed = command.substring(firstSpace + 1).toInt();
        
        // Apply mode-based speed limits
        int maxAllowedSpeed = (currentMode == "AUTO" || currentMode == "AUTOMATIC") 
                              ? AUTO_MODE_SPEED 
                              : MANUAL_MODE_MAX;
        int constrainedSpeed = constrain(speed, MIN_SPEED, maxAllowedSpeed);
        servoManager.setSpeed(constrainedSpeed);
        
        if (speed != constrainedSpeed) {
            Serial.print("Speed constrained from ");
            Serial.print(speed);
            Serial.print(" to ");
            Serial.print(constrainedSpeed);
            Serial.print(" deg/s (");
            Serial.print(currentMode);
            Serial.println(" mode limit)");
        } else {
            Serial.print("Speed set to: ");
            Serial.print(constrainedSpeed);
            Serial.println(" deg/s");
        }
    }
    else if (command.startsWith("ANGLE")) {
        // ANGLE base shoulder forearm elbow pitch claw
        // Format: ANGLE A1 A2 A3 A4 A5 A6
        // Order: Base, Shoulder, Forearm, Elbow, Pitch, Claw
        // Use -1 to keep current angle for any servo
        int firstSpace = command.indexOf(' ');
        if (firstSpace == -1) {
            Serial.println("Invalid ANGLE command format");
            return;
        }
        
        command = command.substring(firstSpace + 1);
        int angles[6];
        int angleIndex = 0;
        
        while (command.length() > 0 && angleIndex < 6) {
            int spaceIdx = command.indexOf(' ');
            String token;
            if (spaceIdx == -1) {
                token = command;
                command = "";
            } else {
                token = command.substring(0, spaceIdx);
                command = command.substring(spaceIdx + 1);
            }
            
            if (token.length() > 0) {
                int val = token.toInt();
                if (val >= 0) {
                    angles[angleIndex] = val;
                } else {
                    // Use current angle if -1
                    angles[angleIndex] = servoManager.getAngle(angleIndex);
                }
                angleIndex++;
            }
        }
        
        if (angleIndex == 6) {
            servoManager.setTargets(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5]);
            Serial.println("ANGLE command executed");
    } else {
            Serial.println("Invalid ANGLE command - need 6 angles");
        }
    }
    else if (command.startsWith("STATUS")) {
        sendStatus();
    }
    else if (command.startsWith("DISTANCE")) {
        // DISTANCE - Read distance from ToF sensor
        int distance = tofManager.getDistance();
        if (distance < 0) {
            // Sensor not initialized or out of range
            Serial.println("DISTANCE: OUT_OF_RANGE");
        } else {
            // Check if distance is in valid range
            if (tofManager.isRangeValid(distance)) {
                Serial.print("DISTANCE: ");
                Serial.println(distance);
            } else {
                Serial.println("DISTANCE: OUT_OF_RANGE");
            }
        }
    }
    else {
        Serial.print("Unknown text command: ");
        Serial.println(command);
    }
}

void executePick(String id, int x, int y, String type, float confidence) {
    if (currentState != IDLE && currentState != MOVING) {
        Serial.println("Cannot start pick - system busy");
        commClient.sendPickResult(id.c_str(), "FAILED", "busy", 0);
        return;
    }
    
    // Check confidence threshold (optional)
    if (confidence > 0 && confidence < 0.5) {
        Serial.println("Pick rejected - low confidence");
        commClient.sendPickResult(id.c_str(), "REJECTED", "low_confidence", 0);
        return;
    }
    
    Serial.print("Starting Pick Sequence: ID=");
    Serial.print(id);
    Serial.print(", Pixel=(");
    Serial.print(x);
    Serial.print(",");
    Serial.print(y);
    Serial.print("), Class=");
    Serial.print(type);
    Serial.print(", Confidence=");
    Serial.println(confidence);
    
    currentPickId = id;
    pickStartTime = millis();
    
    // Normalize class type: "ready" and "ripe" both go to right bin
    String normalizedType = type;
    if (type == "ready" || type == "ripe") {
        normalizedType = "ripe";  // Use "ripe" for right bin
    } else {
        normalizedType = "unripe";  // Everything else goes to left bin
    }
    
    // Start motion planner
    if (motionPlanner.startPick(x, y, confidence, normalizedType)) {
        currentState = PICKING;
        Serial.println("Pick sequence started");
    } else {
        String error = motionPlanner.getLastError();
        Serial.print("Pick start failed: ");
        Serial.println(error);
        commClient.sendPickResult(id.c_str(), "FAILED", error.c_str(), 0);
        currentState = IDLE;
    }
}

void checkEmergencyStop() {
    // Check hardware emergency stop pin (active LOW)
    if (digitalRead(PIN_EMERGENCY_STOP) == LOW) {
        if (currentState != ERROR) {
            Serial.println("EMERGENCY STOP ACTIVATED!");
            motionPlanner.abort();
            servoManager.emergencyStop();
            currentState = ERROR;
        }
    }
}

void sendStatus() {
    // Read Battery (Voltage divider on A0 usually)
    float voltage = analogRead(A0) * (5.0 / 1023.0) * 2; // Example divider
    
    const char* statusStr = "IDLE";
    if (currentState == MOVING) statusStr = "MOVING";
    if (currentState == PICKING) statusStr = "PICKING";
    if (currentState == ERROR) statusStr = "ERROR";
    
    commClient.sendTelemetry(voltage, statusStr, "None");
}
