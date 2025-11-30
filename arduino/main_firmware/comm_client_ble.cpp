#include "comm_client_ble.h"
#include <ArduinoJson.h>

CommClientBLE::CommClientBLE() {
    _service = nullptr;
    _commandChar = nullptr;
    _telemetryChar = nullptr;
    _connected = false;
    _messageCallback = nullptr;
}

bool CommClientBLE::begin() {
    Serial.println("Initializing BLE...");
    
    if (!BLE.begin()) {
        Serial.println("BLE initialization failed!");
        return false;
    }
    
    // Set local name
    BLE.setLocalName(BLE_DEVICE_NAME);
    
    // Create service
    _service = new BLEService(BLE_SERVICE_UUID);
    
    // Create characteristics
    // Command characteristic (write from central, read by central)
    _commandChar = new BLEStringCharacteristic(
        BLE_CHAR_UUID,
        BLERead | BLEWrite | BLENotify,
        512  // Max 512 bytes
    );
    
    // Telemetry characteristic (read by central, notify)
    _telemetryChar = new BLEStringCharacteristic(
        "19B10002-E8F2-537E-4F6C-D104768A1214",
        BLERead | BLENotify,
        512
    );
    
    // Add characteristics to service
    _service->addCharacteristic(*_commandChar);
    _service->addCharacteristic(*_telemetryChar);
    
    // Add service
    BLE.addService(*_service);
    
    // Start advertising
    BLE.advertise();
    
    Serial.print("BLE device advertising as: ");
    Serial.println(BLE_DEVICE_NAME);
    Serial.println("Waiting for BLE central to connect...");
    
    return true;
}

void CommClientBLE::update() {
    // Check for BLE central connection
    BLEDevice central = BLE.central();
    
    if (central) {
        if (!_connected) {
            Serial.print("Connected to BLE central: ");
            Serial.println(central.address());
            _connected = true;
        }
        
        // Check if command received
        processReceivedCommand();
    } else {
        if (_connected) {
            Serial.println("BLE central disconnected");
            _connected = false;
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
        if (_messageCallback) {
            _messageCallback(command);
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

