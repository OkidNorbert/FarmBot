#ifndef CONFIG_H
#define CONFIG_H

// ==========================================
// Pin Definitions
// ==========================================

// Servos
#define PIN_SERVO_CLAW      2
#define PIN_SERVO_PITCH     3
#define PIN_SERVO_ELBOW     4
#define PIN_SERVO_FOREARM   5
#define PIN_SERVO_SHOULDER  6
#define PIN_SERVO_BASE      7

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
// Claw (SG90) - 0° = Open, 90° = Closed (REVERSED)
#define LIMIT_CLAW_MIN      0
#define LIMIT_CLAW_MAX      90

// Pitch (SG90)
#define LIMIT_PITCH_MIN     20
#define LIMIT_PITCH_MAX     160

// Elbow (SG90)
#define LIMIT_ELBOW_MIN     15
#define LIMIT_ELBOW_MAX     165

// Forearm (MG99x)
#define LIMIT_FOREARM_MIN   10
#define LIMIT_FOREARM_MAX   170

// Shoulder (MG99x)
#define LIMIT_SHOULDER_MIN  15
#define LIMIT_SHOULDER_MAX  165

// Base (MG99x)
// Set to true if using a continuous rotation servo for base
// NOTE: Continuous rotation servos don't have position feedback, so the firmware
//       tracks a "virtual position" based on rotation time and speed.
//       You may need to calibrate BASE_ROTATION_SPEED to match your servo's actual speed.
#define BASE_CONTINUOUS_ROTATION  true  // Change to false for standard 180° servo
#define LIMIT_BASE_MIN      0
#define LIMIT_BASE_MAX      180  // Virtual limits for continuous rotation servo
#define BASE_ROTATION_SPEED 30   // Degrees per second for continuous rotation base (calibrate this!)

// Motion - Speed Configuration
#define DEFAULT_SPEED       20  // Default speed (degrees per second)
#define AUTO_MODE_SPEED     45  // Automatic mode speed (deg/s) - optimized for AI/camera/sensor coordination
#define MANUAL_MODE_MAX     120 // Maximum speed for manual mode (deg/s)
#define MIN_SPEED           1   // Minimum speed (deg/s)
#define MAX_SPEED           180 // Absolute maximum speed (deg/s) - hardware limit
#define HOME_ANGLE          90

// ==========================================
// Communication Configuration
// ==========================================
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

