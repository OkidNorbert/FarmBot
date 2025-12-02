#include "comm_client_ble.h"
#include <ArduinoJson.h>

CommClientBLE::CommClientBLE() {
    _service = nullptr;
    _commandChar = nullptr;
    _telemetryChar = nullptr;
    _connected = false;
    _messageCallback = nullptr;
    _last_status_print = 0;
}

bool CommClientBLE::begin() {
    Serial.println("========================================");
    Serial.println("Initializing BLE...");
    Serial.println("========================================");
    
    if (!BLE.begin()) {
        Serial.println("❌ BLE initialization failed!");
        Serial.println("   Check that you're using Arduino UNO R4 WiFi board");
        return false;
    }
    
    Serial.println("✅ BLE hardware initialized");
    
    // Set local name
    BLE.setLocalName(BLE_DEVICE_NAME);
    Serial.print("✅ Device name set to: ");
    Serial.println(BLE_DEVICE_NAME);
    
    // Create service
    _service = new BLEService(BLE_SERVICE_UUID);
    Serial.print("✅ Service created: ");
    Serial.println(BLE_SERVICE_UUID);
    
    // Create characteristics
    // Command characteristic (write from central, read by central)
    _commandChar = new BLEStringCharacteristic(
        BLE_CHAR_UUID,
        BLERead | BLEWrite | BLENotify,
        512  // Max 512 bytes
    );
    Serial.print("✅ Command characteristic created: ");
    Serial.println(BLE_CHAR_UUID);
    
    // Telemetry characteristic (read by central, notify)
    _telemetryChar = new BLEStringCharacteristic(
        "19B10002-E8F2-537E-4F6C-D104768A1214",
        BLERead | BLENotify,
        512
    );
    Serial.println("✅ Telemetry characteristic created");
    
    // Add characteristics to service
    _service->addCharacteristic(*_commandChar);
    _service->addCharacteristic(*_telemetryChar);
    Serial.println("✅ Characteristics added to service");
    
    // Add service
    BLE.addService(*_service);
    Serial.println("✅ Service added to BLE");
    
    // Start advertising
    BLE.advertise();
    
    Serial.println("========================================");
    Serial.print("📡 BLE device advertising as: ");
    Serial.println(BLE_DEVICE_NAME);
    Serial.print("   Service UUID: ");
    Serial.println(BLE_SERVICE_UUID);
    Serial.print("   Characteristic UUID: ");
    Serial.println(BLE_CHAR_UUID);
    Serial.println("========================================");
    Serial.println("⏳ Waiting for BLE central to connect...");
    Serial.println("   (Use Python client or mobile app to connect)");
    Serial.println("========================================");
    
    // Print status every 10 seconds if not connected
    _last_status_print = 0;
    
    return true;
}

void CommClientBLE::update() {
    // Check for BLE central connection
    BLEDevice central = BLE.central();
    
    if (central) {
        if (!_connected) {
            Serial.println("========================================");
            Serial.println("✅ BLE CONNECTION ESTABLISHED!");
            Serial.print("   Central device: ");
            Serial.println(central.address());
            Serial.print("   RSSI: ");
            Serial.print(central.rssi());
            Serial.println(" dBm");
            Serial.println("========================================");
            _connected = true;
        }
        
        // Check if still connected
        if (central.connected()) {
            // Check if command received
            processReceivedCommand();
        } else {
            // Connection lost
            if (_connected) {
                Serial.println("⚠️  BLE central disconnected");
                _connected = false;
            }
        }
    } else {
        if (_connected) {
            Serial.println("⚠️  BLE central disconnected");
            _connected = false;
        }
        
        // Print status every 10 seconds if not connected
        unsigned long now = millis();
        if (now - _last_status_print > 10000) {
            Serial.println("📡 BLE advertising... Waiting for connection...");
            Serial.print("   Device name: ");
            Serial.println(BLE_DEVICE_NAME);
            _last_status_print = now;
        }
    }
}

bool CommClientBLE::isConnected() {
    BLEDevice central = BLE.central();
    return central && central.connected();
}

bool CommClientBLE::available() {
    return isConnected() && _commandChar->written();
}

void CommClientBLE::onMessage(void (*callback)(String)) {
    _messageCallback = callback;
}

void CommClientBLE::processReceivedCommand() {
    if (_commandChar && _commandChar->written()) {
        String command = _commandChar->value();
        Serial.print("📥 BLE command received: ");
        Serial.println(command);
        
        if (_messageCallback) {
            _messageCallback(command);
        } else {
            Serial.println("⚠️  Warning: No message callback set!");
        }
    }
}

void CommClientBLE::sendTelemetry(float voltage, const char* status, const char* last_action) {
    if (!isConnected()) return;
    
    // Create JSON
    String json = "{\"battery_voltage\":";
    json += String(voltage);
    json += ",\"status\":\"";
    json += status;
    json += "\",\"last_action\":\"";
    json += last_action;
    json += "\"}";
    
    // Send via telemetry characteristic
    if (_telemetryChar) {
        _telemetryChar->writeValue(json);
    }
}

void CommClientBLE::sendPickResult(const char* id, const char* status, const char* result, unsigned long duration_ms) {
    if (!isConnected()) return;
    
    // Create JSON
    String json = "{\"id\":\"";
    json += id;
    json += "\",\"status\":\"";
    json += status;
    json += "\",\"result\":\"";
    json += result;
    json += "\",\"duration_ms\":";
    json += String(duration_ms);
    json += "}";
    
    // Send via telemetry characteristic
    if (_telemetryChar) {
        _telemetryChar->writeValue(json);
    }
}

String CommClientBLE::createJSON(const char* type, const char* data) {
    String json = "{\"type\":\"";
    json += type;
    json += "\",\"data\":";
    json += data;
    json += "}";
    return json;
}

