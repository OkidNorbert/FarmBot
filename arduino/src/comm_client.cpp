#include "comm_client.h"
#include <ArduinoJson.h>

#ifdef USE_WIFI
#include <WiFiS3.h>
#include <ArduinoWebsockets.h>
using namespace websockets;
#endif

#ifdef USE_BLE
#include "comm_client_ble.h"
#endif

CommClient::CommClient() {
    _connected = false;
    _connectionType = "NONE";
    _messageCallback = nullptr;
    
    #ifdef USE_WIFI
    wifiClient = new WebsocketsClient();
    _wifiConnected = false;
    _last_reconnect_attempt = 0;
    #endif
    
    #ifdef USE_BLE
    bleClient = new CommClientBLE();
    _bleConnected = false;
    #endif
}

void CommClient::begin() {
    String commMode = String(COMM_MODE);
    commMode.toUpperCase();
    
    Serial.println("========================================");
    Serial.println("Initializing Communication...");
    Serial.print("Mode: ");
    Serial.println(commMode);
    Serial.println("========================================");
    
    if (commMode == "AUTO" || commMode == "WIFI") {
        #ifdef USE_WIFI
        Serial.println("Attempting WiFi connection...");
        connectWiFi();
        if (_wifiConnected) {
            connectWS();
            if (_connected) {
                _connectionType = "WIFI";
                Serial.println("✅ Connected via WiFi/WebSocket");
                return;
            }
        }
        #endif
        
        if (commMode == "AUTO" && !_connected) {
            Serial.println("WiFi failed, trying BLE...");
            #ifdef USE_BLE
            if (bleClient->begin()) {
                _bleConnected = true;
                _connected = true;
                _connectionType = "BLE";
                Serial.println("✅ Connected via BLE");
                if (_messageCallback) {
                    bleClient->onMessage(_messageCallback);
                }
                return;
            }
            #endif
        }
    } else if (commMode == "BLE") {
        #ifdef USE_BLE
        Serial.println("Initializing BLE...");
        if (bleClient->begin()) {
            _bleConnected = true;
            _connected = true;
            _connectionType = "BLE";
            Serial.println("✅ Connected via BLE");
            if (_messageCallback) {
                bleClient->onMessage(_messageCallback);
            }
            return;
        }
        #endif
    }
    
    Serial.println("❌ No communication method available");
    _connected = false;
}

void CommClient::update() {
    #ifdef USE_WIFI
    if (_connectionType == "WIFI") {
        if (WiFi.status() != WL_CONNECTED) {
            _wifiConnected = false;
            _connected = false;
            connectWiFi();
            return;
        }
        
        if (!wifiClient->available()) {
            unsigned long now = millis();
            if (now - _last_reconnect_attempt > 5000) {
                connectWS();
                _last_reconnect_attempt = now;
            }
        } else {
            wifiClient->poll();
        }
    }
    #endif
    
    #ifdef USE_BLE
    if (_connectionType == "BLE") {
        bleClient->update();
        _connected = bleClient->isConnected();
    }
    #endif
}

bool CommClient::isConnected() {
    #ifdef USE_WIFI
    if (_connectionType == "WIFI") {
        return wifiClient->available();
    }
    #endif
    
    #ifdef USE_BLE
    if (_connectionType == "BLE") {
        return bleClient->isConnected();
    }
    #endif
    
    return false;
}

void CommClient::onMessage(void (*callback)(String)) {
    _messageCallback = callback;
    
    #ifdef USE_WIFI
    if (_connectionType == "WIFI" && wifiClient) {
        wifiClient->onMessage([this](WebsocketsMessage message) {
            String data = message.data();
            // Socket.IO protocol handling
            if (data.startsWith("42")) {
                String payload = data.substring(2);
                this->handleMessage(payload);
            }
        });
    }
    #endif
    
    #ifdef USE_BLE
    if (_connectionType == "BLE" && bleClient) {
        bleClient->onMessage([this](String message) {
            this->handleMessage(message);
        });
    }
    #endif
}

void CommClient::handleMessage(String payload) {
    if (_messageCallback) {
        _messageCallback(payload);
    }
}

void CommClient::sendTelemetry(float voltage, const char* status, const char* last_action) {
    if (!isConnected()) return;
    
    #ifdef USE_WIFI
    if (_connectionType == "WIFI") {
        String json = "{\"battery_voltage\":";
        json += String(voltage);
        json += ",\"status\":\"";
        json += status;
        json += "\",\"last_action\":\"";
        json += last_action;
        json += "\"}";
        
        String packet = "42[\"telemetry\"," + json + "]";
        wifiClient->send(packet);
        return;
    }
    #endif
    
    #ifdef USE_BLE
    if (_connectionType == "BLE") {
        bleClient->sendTelemetry(voltage, status, last_action);
        return;
    }
    #endif
}

void CommClient::sendPickResult(const char* id, const char* status, const char* result, unsigned long duration_ms) {
    if (!isConnected()) return;
    
    #ifdef USE_WIFI
    if (_connectionType == "WIFI") {
        String json = "{\"id\":\"";
        json += id;
        json += "\",\"status\":\"";
        json += status;
        json += "\",\"result\":\"";
        json += result;
        json += "\",\"duration_ms\":";
        json += String(duration_ms);
        json += "}";
        
        String packet = "42[\"pick_result\"," + json + "]";
        wifiClient->send(packet);
        return;
    }
    #endif
    
    #ifdef USE_BLE
    if (_connectionType == "BLE") {
        bleClient->sendPickResult(id, status, result, duration_ms);
        return;
    }
    #endif
}

const char* CommClient::getConnectionType() {
    return _connectionType.c_str();
}

#ifdef USE_WIFI
void CommClient::connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) {
        _wifiConnected = true;
        return;
    }
    
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi Connected!");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
        _wifiConnected = true;
    } else {
        Serial.println("\nWiFi Connection Failed");
        _wifiConnected = false;
    }
}

void CommClient::connectWS() {
    if (wifiClient->available()) {
        _connected = true;
        return;
    }
    
    Serial.println("Connecting to WebSocket...");
    String url = String("ws://") + WS_HOST + ":" + WS_PORT + "/socket.io/?EIO=4&transport=websocket";
    
    bool connected = wifiClient->connect(url);
    
    if (connected) {
        Serial.println("WebSocket Connected!");
        _connected = true;
        wifiClient->send("40"); // Socket.IO handshake
    } else {
        Serial.println("WebSocket Connection Failed");
        _connected = false;
    }
}
#endif
