#ifndef CONFIG_H
#define CONFIG_H

// ==========================================
// Pin Definitions
// ==========================================

// Servos
#define PIN_SERVO_CLAW         2
#define PIN_SERVO_WRIST_PITCH  3   // Previously Pitch
#define PIN_SERVO_WRIST_ROLL   4   // Previously Elbow
#define PIN_SERVO_ELBOW        5   // Previously Forearm
#define PIN_SERVO_SHOULDER     6
#define PIN_SERVO_BASE         7   // Waist

// Sensors
// VL53L0X uses I2C (SDA/SCL) - on R4 WiFi these are default pins

// Safety
#define PIN_EMERGENCY_STOP  8  // Hardware kill switch input (Active LOW)

// ==========================================
// Servo Configuration
// ==========================================

// Pulse Widths (Microseconds)
#define PULSE_MIN_SG90      500
#define PULSE_MAX_SG90      2400
#define PULSE_MIN_MG99X     600
#define PULSE_MAX_MG99X     2400

// Safety Limits (Degrees)
// Claw (SG90) - 10° = Open, 100° = Closed
// NOTE: Some servos may need a slight offset (1-2°) to fully close
// The claw is considered "fully open" at 10° by default; the host software
// clamps any commands below 10° as well.  You can still adjust these values
// during calibration if your hardware behaves differently.
#define LIMIT_CLAW_MIN      10   // do not drive claw below this angle (fully open)
#define LIMIT_CLAW_MAX      115  // do not drive claw above this angle (fully closed)
#define CLAW_CLOSED_POSITION 115  // Actual position for fully closed claw

// Wrist Pitch (SG90) - Previously Pitch
#define LIMIT_WRIST_PITCH_MIN  0
#define LIMIT_WRIST_PITCH_MAX  160

// Wrist Roll (SG90) - Previously Elbow
#define LIMIT_WRIST_ROLL_MIN   15
#define LIMIT_WRIST_ROLL_MAX   165

// Elbow (MG99x) - Previously Forearm
#define LIMIT_ELBOW_MIN        10
#define LIMIT_ELBOW_MAX        180

// Shoulder (SG90) - Updated to match servo type
#define LIMIT_SHOULDER_MIN     15
#define LIMIT_SHOULDER_MAX     165

// Base (MG99x)
// Set to true if using a continuous rotation servo for base
// NOTE: Continuous rotation servos don't have position feedback, so the firmware
//       tracks a "virtual position" based on rotation time and speed.
//       The continuous rotation servo now respects the global speed setting (setSpeed()).
//       BASE_ROTATION_SPEED is used as a calibration factor - it represents the maximum
//       rotation speed when the servo command is at full speed (60° CCW or 120° CW).
//       You may need to calibrate BASE_ROTATION_SPEED to match your servo's actual speed.
#define BASE_CONTINUOUS_ROTATION  true  // Change to false for standard 180° servo
#define LIMIT_BASE_MIN      0
#define LIMIT_BASE_MAX      180  // Virtual limits for continuous rotation servo
#define BASE_ROTATION_SPEED 30   // Max degrees per second at full servo command (calibrate this!)
#define BASE_COASTING_TIME_MS 100 // Estimated coasting time in milliseconds after stop command
#define BASE_STOP_TOLERANCE  2    // Stop tolerance in degrees (virtual position)

// Pitch Servo (Continuous rotation if true, standard 180 if false)
#define PITCH_CONTINUOUS_ROTATION  false
#define PITCH_ROTATION_SPEED       90.0 // Degrees per second calibration
#define PITCH_COASTING_TIME_MS     100  // MS to coast after stop command
#define PITCH_STOP_TOLERANCE       2.0  // Degrees threshold to stop early

// Motion - Speed Configuration
#define DEFAULT_SPEED       20  // Default speed (degrees per second)
#define AUTO_MODE_SPEED     45  // Automatic mode speed (deg/s) - optimized for AI/camera/sensor coordination
#define MANUAL_MODE_MAX     120 // Maximum speed for manual mode (deg/s)
#define MIN_SPEED           1   // Minimum speed (deg/s)
#define MAX_SPEED           180 // Absolute maximum speed (deg/s) - hardware limit
#define HOME_BASE_ANGLE         90
#define HOME_SHOULDER_ANGLE     90
#define HOME_ELBOW_ANGLE        90   // Reset to 90 per user request
#define HOME_WRIST_ROLL_ANGLE   90
#define HOME_WRIST_PITCH_ANGLE  90
#define HOME_CLAW_ANGLE         CLAW_CLOSED_POSITION // Closed on initiation per request

#define HOME_ANGLE          90   // Legacy default for unspecified joints
// Choose communication method: "WIFI", "BLE", or "AUTO" (tries WiFi first, falls back to BLE)
// Set to BLE since WiFi WebSocket client library has compatibility issues
#define COMM_MODE           "BLE"

// Enable communication methods (comment out to disable)
// NOTE: Make sure you select "Arduino UNO R4 WiFi" board in Arduino IDE!
// NOT "Arduino Nano R4" - they are different boards!
// WiFi temporarily disabled - WebSocket client library compatibility issue
// #define USE_WIFI            1
#define USE_BLE             1

// WiFi / WebSocket Configuration
#define WIFI_SSID           "FarmBot_Net"
#define WIFI_PASS           "tomato123"
#define WS_HOST             "192.168.1.100" // Replace with actual PC IP
#define WS_PORT             5000
#define WS_PATH             "/socket.io/?EIO=4&transport=websocket"

// BLE Configuration
#define BLE_DEVICE_NAME     "FarmBot"
#define BLE_SERVICE_UUID    "19B10000-E8F2-537E-4F6C-D104768A1214"
#define BLE_CHAR_UUID       "19B10001-E8F2-537E-4F6C-D104768A1214"

#endif // CONFIG_H

