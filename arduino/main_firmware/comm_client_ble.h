#ifndef COMM_CLIENT_BLE_H
#define COMM_CLIENT_BLE_H

#include <Arduino.h>
#include <ArduinoBLE.h>
#include "config.h"

class CommClientBLE {
public:
    CommClientBLE();
    bool begin();
    void update();
    bool isConnected();
    void sendTelemetry(float voltage, const char* status, const char* last_action);
    void sendPickResult(const char* id, const char* status, const char* result, unsigned long duration_ms = 0);
    
    // Callback for received messages
    void onMessage(void (*callback)(String));
    
    // Check if data available
    bool available();

private:
    BLEService* _service;
    BLEStringCharacteristic* _commandChar;
    BLEStringCharacteristic* _telemetryChar;
    bool _connected;
    void (*_messageCallback)(String);
    
    void processReceivedCommand();
    String createJSON(const char* type, const char* data);
};

#endif // COMM_CLIENT_BLE_H

