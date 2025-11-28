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
// Claw (SG90)
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
#define LIMIT_BASE_MIN      0
#define LIMIT_BASE_MAX      180

// Motion
#define DEFAULT_SPEED       20  // Degrees per second (approx)
#define HOME_ANGLE          90

// ==========================================
// Communication Configuration
// ==========================================
// Choose communication method: "WIFI", "BLE", or "AUTO" (tries WiFi first, falls back to BLE)
#define COMM_MODE           "AUTO"

// Enable communication methods (comment out to disable)
#define USE_WIFI            1
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
