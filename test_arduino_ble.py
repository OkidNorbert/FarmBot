#!/usr/bin/env python3
"""
Simple BLE Test Script for Arduino UNO R4 WiFi
Tests connection to Arduino via Bluetooth Low Energy
"""

import asyncio
import sys

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    print("❌ Error: Bleak library not installed")
    print("Install it with: pip install bleak")
    sys.exit(1)

# BLE Configuration (matches Arduino firmware)
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
DEVICE_NAME = "FarmBot"

async def scan_for_arduino():
    """Scan for Arduino BLE device"""
    print("🔍 Scanning for Arduino BLE device...")
    print(f"Looking for device named: {DEVICE_NAME}")
    print("Make sure Arduino is powered on and BLE is active!")
    print("-" * 60)
    
    devices = await BleakScanner.discover(timeout=10.0)
    
    print(f"\n📡 Found {len(devices)} BLE device(s):")
    print("-" * 60)
    
    arduino_found = None
    for device in devices:
        name = device.name or "Unknown"
        print(f"  • {name} ({device.address})")
        if DEVICE_NAME.lower() in name.lower():
            arduino_found = device
            print(f"  ✅ MATCH! This is your Arduino!")
    
    print("-" * 60)
    
    if arduino_found:
        print(f"\n✅ Arduino found: {arduino_found.name}")
        print(f"   Address: {arduino_found.address}")
        return arduino_found
    else:
        print(f"\n❌ Arduino '{DEVICE_NAME}' not found!")
        print("\nTroubleshooting:")
        print("  1. Make sure Arduino is powered on")
        print("  2. Check that firmware is uploaded and running")
        print("  3. Verify Arduino is advertising BLE (check Serial Monitor)")
        print("  4. Make sure Bluetooth is enabled on your computer")
        return None

async def test_connection(device):
    """Test connection to Arduino"""
    print(f"\n🔌 Attempting to connect to {device.name}...")
    
    try:
        async with BleakClient(device) as client:
            print("✅ Connected to Arduino!")
            print(f"   Service UUID: {SERVICE_UUID}")
            print(f"   Characteristic UUID: {CHAR_UUID}")
            
            # Try to read services
            services = await client.get_services()
            print(f"\n📋 Available services: {len(services)}")
            for service in services:
                print(f"   Service: {service.uuid}")
                for char in service.characteristics:
                    print(f"     Characteristic: {char.uuid}")
            
            # Test sending a command
            print("\n📤 Testing command send...")
            test_command = '{"cmd":"home"}'
            await client.write_gatt_char(CHAR_UUID, test_command.encode())
            print("✅ Command sent successfully!")
            print("   Command: HOME (return to home position)")
            print("   Check Arduino Serial Monitor for response")
            
            await asyncio.sleep(1)
            print("\n✅ Connection test complete!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Arduino is not connected to another device")
        print("  2. Try resetting the Arduino")
        print("  3. Check Serial Monitor for Arduino status")

async def main():
    """Main test function"""
    print("=" * 60)
    print("🤖 Arduino UNO R4 WiFi - BLE Connection Test")
    print("=" * 60)
    print()
    
    device = await scan_for_arduino()
    
    if device:
        await test_connection(device)
    else:
        print("\n⚠️  Cannot test connection - Arduino not found")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

