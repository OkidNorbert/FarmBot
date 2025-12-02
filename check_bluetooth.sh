#!/bin/bash
# Bluetooth Diagnostic and Setup Script

echo "=========================================="
echo "Bluetooth Diagnostic Script"
echo "=========================================="
echo ""

# Check if bluetoothctl exists
if ! command -v bluetoothctl &> /dev/null; then
    echo "❌ bluetoothctl not found"
    echo "Install with: sudo apt-get install bluez bluez-utils"
    exit 1
fi

echo "✅ bluetoothctl found"
echo ""

# Check Bluetooth service status
echo "Checking Bluetooth service status..."
if systemctl is-active --quiet bluetooth; then
    echo "✅ Bluetooth service is running"
else
    echo "❌ Bluetooth service is NOT running"
    echo ""
    echo "To start it:"
    echo "  sudo systemctl start bluetooth"
    echo "  sudo systemctl enable bluetooth"
    echo ""
fi

# Check if Bluetooth adapter is available
echo ""
echo "Checking for Bluetooth adapters..."
ADAPTER_COUNT=$(bluetoothctl list 2>&1 | grep -c "Controller" || echo "0")

if [ "$ADAPTER_COUNT" -eq "0" ]; then
    echo "❌ No Bluetooth adapters found"
    echo ""
    echo "Possible causes:"
    echo "  1. No Bluetooth hardware installed"
    echo "  2. Bluetooth adapter disabled in BIOS/UEFI"
    echo "  3. Bluetooth driver not loaded"
    echo ""
    echo "To check hardware:"
    echo "  lsusb | grep -i bluetooth"
    echo "  lspci | grep -i bluetooth"
    echo "  dmesg | grep -i bluetooth"
    echo ""
    echo "To load Bluetooth module (if available):"
    echo "  sudo modprobe btusb"
    echo ""
else
    echo "✅ Found $ADAPTER_COUNT Bluetooth adapter(s)"
    echo ""
    echo "Adapter details:"
    bluetoothctl show
    echo ""
    
    # Check if powered on
    POWER_STATE=$(bluetoothctl show | grep -i "Powered:" | awk '{print $2}')
    if [ "$POWER_STATE" = "yes" ]; then
        echo "✅ Bluetooth is powered ON"
    else
        echo "❌ Bluetooth is powered OFF"
        echo ""
        echo "To power on:"
        echo "  bluetoothctl power on"
        echo "  (or: sudo bluetoothctl power on)"
        echo ""
    fi
fi

echo "=========================================="
echo "Diagnostic complete"
echo "=========================================="

