#include "comm_client.h"

CommClient::CommClient() {
    _connected = false;
    _last_reconnect_attempt = 0;
}

void CommClient::begin() {
    connectWiFi();
    connectWS();
}

void CommClient::connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    
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
    } else {
        Serial.println("\nWiFi Connection Failed");
    }
}

void CommClient::connectWS() {
    if (client.available()) return;
    
    Serial.println("Connecting to WebSocket...");
    // Construct full URL
    String url = String("ws://") + WS_HOST + ":" + WS_PORT + "/socket.io/?EIO=4&transport=websocket";
    
    bool connected = client.connect(url);
    
    if (connected) {
        Serial.println("WebSocket Connected!");
        _connected = true;
        // Send initial handshake if needed for Socket.IO 4
        client.send("40"); 
    } else {
        Serial.println("WebSocket Connection Failed");
        _connected = false;
    }
}

void CommClient::update() {
    if (WiFi.status() != WL_CONNECTED) {
        connectWiFi();
        return;
    }
    
    if (!client.available()) {
        unsigned long now = millis();
        if (now - _last_reconnect_attempt > 5000) {
            connectWS();
            _last_reconnect_attempt = now;
        }
    } else {
        client.poll();
    }
}

bool CommClient::isConnected() {
    return client.available();
}

void CommClient::onMessage(void (*callback)(String)) {
    client.onMessage([callback](WebsocketsMessage message) {
        String data = message.data();
        // Socket.IO protocol handling
        // 42["event", data]
        if (data.startsWith("42")) {
            String payload = data.substring(2);
            callback(payload);
        }
    });
}

void CommClient::sendTelemetry(float voltage, const char* status, const char* last_action) {
    if (!isConnected()) return;
    
    // Format: 42["telemetry", {...}]
    String json = "{\"battery_voltage\":";
    json += String(voltage);
    json += ",\"status\":\"";
    json += status;
    json += "\",\"last_action\":\"";
    json += last_action;
    json += "\"}";
    
    String packet = "42[\"telemetry\"," + json + "]";
    client.send(packet);
}

void CommClient::sendPickResult(const char* id, const char* status, const char* result, unsigned long duration_ms) {
    if (!isConnected()) return;
    
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
    client.send(packet);
}
