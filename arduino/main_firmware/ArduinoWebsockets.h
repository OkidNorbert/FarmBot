#ifndef ARDUINOWEBSOCKETS_H
#define ARDUINOWEBSOCKETS_H

#include "WebSockets.h"
#include "WebSocketsClient.h"

// Wrapper namespace to match expected API
namespace websockets {

// Message wrapper class
class WebsocketsMessage {
private:
    String _data;
public:
    WebsocketsMessage(String data) : _data(data) {}
    String data() { return _data; }
};

// Client wrapper class to match expected API
class WebsocketsClient {
private:
    ::WebSocketsClient* _client;
    bool _connected;
    void (*_onMessageCallback)(WebsocketsMessage);
    
    static void eventHandler(WStype_t type, uint8_t * payload, size_t length) {
        // This is a simplified handler - you may need to adjust based on your needs
        if (type == WStype_TEXT) {
            String message = String((char*)payload);
            // Call the callback if set
            // Note: This is a limitation - we can't easily access instance from static
        }
    }
    
public:
    WebsocketsClient() {
        _client = new ::WebSocketsClient();
        _connected = false;
        _onMessageCallback = nullptr;
    }
    
    ~WebsocketsClient() {
        delete _client;
    }
    
    bool connect(String url) {
        // Parse URL (simplified - assumes ws://host:port/path format)
        int protocolEnd = url.indexOf("://");
        if (protocolEnd == -1) return false;
        
        String host = url.substring(protocolEnd + 3);
        int portEnd = host.indexOf(':');
        int pathStart = host.indexOf('/');
        
        String hostname;
        uint16_t port = 80;
        String path = "/";
        
        if (portEnd != -1) {
            hostname = host.substring(0, portEnd);
            if (pathStart != -1 && pathStart > portEnd) {
                port = host.substring(portEnd + 1, pathStart).toInt();
                path = host.substring(pathStart);
            } else {
                port = host.substring(portEnd + 1).toInt();
            }
        } else if (pathStart != -1) {
            hostname = host.substring(0, pathStart);
            path = host.substring(pathStart);
        } else {
            hostname = host;
        }
        
        _client->begin(hostname.c_str(), port, path.c_str());
        _connected = true;
        return true;
    }
    
    bool available() {
        return _connected && _client->isConnected();
    }
    
    void poll() {
        _client->loop();
    }
    
    void onMessage(void (*callback)(WebsocketsMessage)) {
        _onMessageCallback = callback;
        // Set up event handler
        _client->onEvent([this](WStype_t type, uint8_t * payload, size_t length) {
            if (type == WStype_TEXT && _onMessageCallback) {
                String message = String((char*)payload);
                WebsocketsMessage msg(message);
                _onMessageCallback(msg);
            }
        });
    }
    
    void send(String data) {
        if (_connected) {
            _client->sendTXT(data);
        }
    }
    
    bool isConnected() {
        return _client->isConnected();
    }
};

} // namespace websockets

#endif // ARDUINOWEBSOCKETS_H

