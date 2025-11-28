#ifndef COMM_CLIENT_H
#define COMM_CLIENT_H

#include <Arduino.h>
#include "config.h"

// Forward declarations
class CommClientBLE;

#ifdef USE_WIFI
#include <WiFiS3.h>
#include <ArduinoWebsockets.h>
using namespace websockets;
#endif

class CommClient {
public:
    CommClient();
    void begin();
    void update();
    bool isConnected();
    void sendTelemetry(float voltage, const char* status, const char* last_action);
    void sendPickResult(const char* id, const char* status, const char* result, unsigned long duration_ms = 0);
    
    // Callback for received messages
    void onMessage(void (*callback)(String));
    
    // Get connection type
    const char* getConnectionType();

private:
    #ifdef USE_WIFI
    WebsocketsClient* wifiClient;
    bool _wifiConnected;
    unsigned long _last_reconnect_attempt;
    void connectWiFi();
    void connectWS();
    #endif
    
    #ifdef USE_BLE
    CommClientBLE* bleClient;
    bool _bleConnected;
    #endif
    
    bool _connected;
    String _connectionType;
    void (*_messageCallback)(String);
    
    void handleMessage(String payload);
};

#endif // COMM_CLIENT_H
