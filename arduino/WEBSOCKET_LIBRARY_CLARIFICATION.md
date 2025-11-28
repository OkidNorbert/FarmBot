# WebSocket Library Clarification

## ⚠️ Important: Which WebSocket Library to Use

You're seeing "WebSockets by gii moimon" (Gil Maimon) in the Library Manager. Here's what you need to know:

### Option 1: ArduinoWebsockets by Links2004 (Recommended)

**This is the library the code is designed for:**
- **Name**: `ArduinoWebsockets`
- **Author**: **Links2004** (Markus Sattler)
- **GitHub**: https://github.com/Links2004/arduinoWebSockets

**If you can't find this in Library Manager:**
1. Try searching: `Links2004` (author name)
2. Or install manually from GitHub (see below)

### Option 2: ArduinoWebsockets by Gil Maimon (Alternative)

**If Links2004's library is not available, you can try:**
- **Name**: `ArduinoWebsockets`
- **Author**: **Gil Maimon** (gii moimon)
- **GitHub**: https://github.com/gilmaimon/ArduinoWebsockets

**Note**: This library might work, but the API might be slightly different. You may need to adjust the code.

## How to Check Which One You Have

After installing, check the library folder:
- **Windows**: `Documents/Arduino/libraries/ArduinoWebsockets/`
- **Mac/Linux**: `~/Arduino/libraries/ArduinoWebsockets/`

Look at the `library.properties` file - it should show the author.

## Recommended Solution

### If Links2004's Library is Available:
✅ **Install**: "ArduinoWebsockets" by **Links2004**

### If Only Gil Maimon's is Available:
1. **Try installing it first** - it might work with minor code adjustments
2. **Or install Links2004's manually** (see below)

## Manual Installation (Links2004's Library)

If you can't find Links2004's library in Library Manager:

### Method 1: Download ZIP
1. Go to: https://github.com/Links2004/arduinoWebSockets
2. Click **Code** → **Download ZIP**
3. Extract to: `Arduino/libraries/ArduinoWebsockets/`
4. Restart Arduino IDE

### Method 2: Git Clone
```bash
cd ~/Arduino/libraries/
git clone https://github.com/Links2004/arduinoWebSockets.git ArduinoWebsockets
```

## Testing Compatibility

After installing either library:

1. **Compile the code**: Sketch → Verify/Compile
2. **Check for errors**:
   - ✅ If it compiles → Library works!
   - ❌ If errors → May need code adjustments

## Code Compatibility

The current code uses:
```cpp
#include <ArduinoWebsockets.h>
using namespace websockets;
WebsocketsClient client;
```

**Both libraries should support this**, but if you get errors with Gil Maimon's version, we can adjust the code.

## Quick Decision Guide

**If you see in Library Manager:**
- ✅ "ArduinoWebsockets by Links2004" → Install this one
- ⚠️ "ArduinoWebsockets by Gil Maimon" → Can try this, or install Links2004 manually
- ❌ "WebSockets" (different name) → Not the right one

## What to Do Now

1. **Try installing** "ArduinoWebsockets by gii moimon" (Gil Maimon)
2. **Compile the code** to see if it works
3. **If it doesn't work**, install Links2004's library manually (see above)
4. **Report back** if you need code adjustments

## Alternative: Use BLE Only

If WebSocket libraries are problematic, you can:
1. Set `COMM_MODE` to `"BLE"` in `config.h`
2. Use Bluetooth only (no WiFi/WebSocket needed)
3. This avoids WebSocket library issues entirely

Let me know which library you installed and if the code compiles!

