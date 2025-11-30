# ⚠️ CRITICAL: Board Selection in Arduino IDE

## Correct Board Selection

**You MUST select the correct board in Arduino IDE:**

### ✅ CORRECT:
- **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi**

### ❌ WRONG:
- ~~Arduino Nano R4~~ (Different board, no BLE support in ArduinoBLE library)
- ~~Arduino UNO R4 Boards → Arduino UNO R4~~ (No WiFi/BLE)

## Why This Matters

- **Arduino UNO R4 WiFi** has:
  - ✅ Built-in WiFi (WiFiS3 library)
  - ✅ Built-in Bluetooth Low Energy (ArduinoBLE library supports it)
  - ✅ Renesas RA4M1 microcontroller

- **Arduino Nano R4** has:
  - ❌ Different microcontroller
  - ❌ ArduinoBLE library doesn't support it (causes compilation errors)
  - ❌ Different pin layout

## How to Select Correct Board

1. Open Arduino IDE
2. Go to **Tools → Board → Boards Manager**
3. Search for **"Arduino UNO R4 WiFi"**
4. Install the board package if not already installed
5. Go to **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi**
6. Select the correct **Port** (Tools → Port)

## Verification

After selecting the correct board, compile the code. You should NOT see:
- ❌ `"Unsupported board selected!"` error
- ❌ `SerialHCI was not declared` error

If you see these errors, you have the wrong board selected!

## Current Configuration

- **Board**: Arduino UNO R4 WiFi (should be selected)
- **WiFi**: Enabled
- **BLE**: Enabled  
- **Communication Mode**: AUTO (tries WiFi first, falls back to BLE)

