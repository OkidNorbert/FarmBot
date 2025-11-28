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

// Forward Declarations
void handleWebSocketMessage(String payload);
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
    
    // Parse JSON - handle both Socket.IO format and direct JSON
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) {
        Serial.print("JSON Parse Failed: ");
        Serial.println(error.c_str());
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
    String cmd = data["cmd"].as<String>();
    
    if (cmd == "pick") {
            String id = data["id"] | "unknown";
            int x = data["x"] | 320;
            int y = data["y"] | 240;
            String type = data["class"] | "ripe";
            float confidence = data["confidence"] | 0.0;
            executePick(id, x, y, type, confidence);
        }
        else if (cmd == "move_joints") {
            servoManager.setTargets(
                data["base"],
                data["shoulder"],
                data["forearm"],
                data["elbow"],
                data["pitch"],
                data["claw"]
            );
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
            // Mode switching logic (for future use)
            Serial.print("Mode set to: ");
            Serial.println(mode);
        }
        else if (cmd == "calibrate") {
            // Enter calibration mode
            currentState = CALIBRATING;
            Serial.println("Entering calibration mode");
        }
    } else {
        Serial.print("No 'cmd' field in message");
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
    
    // Start motion planner
    if (motionPlanner.startPick(x, y, confidence, type)) {
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
