#ifndef COMM_CLIENT_H
#define COMM_CLIENT_H

#include <Arduino.h>
#include <WiFiS3.h> // For UNO R4 WiFi
#include <ArduinoWebsockets.h>
#include "config.h"

using namespace websockets;

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

private:
    WebsocketsClient client;
    bool _connected;
    unsigned long _last_reconnect_attempt;
    
    void connectWiFi();
    void connectWS();
};

#endif // COMM_CLIENT_H
