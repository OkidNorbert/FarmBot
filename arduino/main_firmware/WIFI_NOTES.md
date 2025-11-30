# WiFi WebSocket Client Support for Arduino R4 WiFi

## Current Status

The "Web Server for Arduino Uno R4 WiFi" library from DIYables.io only supports **WebSocket SERVER** mode, not **CLIENT** mode.

## Architecture Requirements

The current system architecture requires:
- **Arduino as CLIENT** → Connects to PC web server via WebSocket
- **PC as SERVER** → Runs Flask web server with WebSocket support

## Library Limitation

The DIYables library provides:
- ✅ WebSocket Server (Arduino hosts server, clients connect to it)
- ❌ WebSocket Client (Arduino connects to external server) - **NOT SUPPORTED**

## Solutions

### Option 1: Use BLE (Current - Recommended)
- ✅ Works perfectly
- ✅ No library compatibility issues
- ✅ Simple and reliable
- ⚠️ Limited range (~10 meters)

### Option 2: Reverse Architecture (Arduino as Server)
Would require:
- Arduino runs WebSocket server
- PC connects to Arduino (instead of Arduino connecting to PC)
- Significant code changes in both Arduino and Python web interface
- More complex network setup

### Option 3: Use HTTP REST API Instead of WebSocket
- Replace WebSocket with HTTP POST/GET requests
- Simpler but less real-time
- Requires code changes

### Option 4: Find/Create WebSocket Client Library
- Look for other libraries that support WebSocket client mode
- Or create a custom WebSocket client implementation

## Recommendation

**Keep using BLE** - it's working well and provides reliable communication without the complexity of WiFi setup.

If WiFi is absolutely required, consider Option 2 (reverse architecture) but it will require significant refactoring.

