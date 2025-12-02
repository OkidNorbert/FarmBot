# Fix: "No Bluetooth adapters found" Error

## Problem

You're seeing this error:
```
ERROR:HardwareController:BLE scan error: ('No Bluetooth adapters found.', <BleakBluetoothNotAvailableReason.NO_BLUETOOTH: 1>)
```

This means your Linux system doesn't have a Bluetooth adapter available or it's not detected.

## Quick Diagnosis

Run the diagnostic script:
```bash
./check_bluetooth.sh
```

Or check manually:
```bash
# Check if Bluetooth service is running
systemctl status bluetooth

# Check for Bluetooth adapters
bluetoothctl show

# Check for Bluetooth hardware
lsusb | grep -i bluetooth
lspci | grep -i bluetooth
```

## Solutions

### Solution 1: Use Serial Connection Instead (Recommended if no Bluetooth)

If you don't have Bluetooth hardware or don't need it, use Serial connection:

**Option A: Change in code**
```python
# In web_interface.py or wherever HardwareController is initialized
hw_controller = HardwareController(
    connection_type='serial',  # Use serial instead of 'auto' or 'bluetooth'
    ble_device_name="FarmBot"
)
```

**Option B: Connect Arduino via USB**
- Connect Arduino UNO R4 WiFi to your computer via USB cable
- The system will automatically use Serial connection when Bluetooth is not available

### Solution 2: Enable Bluetooth Hardware

#### Check if Bluetooth Hardware Exists

```bash
# Check USB devices
lsusb | grep -i bluetooth

# Check PCI devices  
lspci | grep -i bluetooth

# Check kernel messages
dmesg | grep -i bluetooth
```

#### If Hardware Found but Not Working

1. **Load Bluetooth module**:
   ```bash
   sudo modprobe btusb
   ```

2. **Check if module loaded**:
   ```bash
   lsmod | grep btusb
   ```

3. **Restart Bluetooth service**:
   ```bash
   sudo systemctl restart bluetooth
   ```

4. **Power on Bluetooth**:
   ```bash
   bluetoothctl power on
   ```

#### If No Hardware Found

- **Desktop/Laptop**: May need to enable Bluetooth in BIOS/UEFI settings
- **Virtual Machine**: Bluetooth passthrough may not be configured
- **No Bluetooth**: Consider using USB Bluetooth adapter or Serial connection

### Solution 3: Install USB Bluetooth Adapter

If your system doesn't have built-in Bluetooth:

1. **Purchase USB Bluetooth adapter** (BLE 4.0+ recommended)
2. **Plug it in** to USB port
3. **Check if detected**:
   ```bash
   lsusb | grep -i bluetooth
   ```
4. **Load driver** (usually automatic):
   ```bash
   sudo modprobe btusb
   ```
5. **Restart Bluetooth service**:
   ```bash
   sudo systemctl restart bluetooth
   ```

### Solution 4: Enable Bluetooth in BIOS/UEFI

Some systems have Bluetooth hardware but it's disabled:

1. **Reboot and enter BIOS/UEFI** (usually F2, F10, Del during boot)
2. **Look for Bluetooth/Wireless settings**
3. **Enable Bluetooth**
4. **Save and exit**
5. **Boot into Linux and check**:
   ```bash
   bluetoothctl show
   ```

## Verify Fix

After applying a solution, verify:

```bash
# Check Bluetooth adapter
bluetoothctl show

# Should show something like:
# Controller AA:BB:CC:DD:EE:FF
#     Name: Your-Computer
#     Alias: Your-Computer
#     Powered: yes
#     Discoverable: no
#     ...
```

Then test Python connection:
```bash
python test_arduino_ble.py
```

## Recommended: Use Serial Connection

For development and testing, **Serial connection is more reliable**:

1. **Connect Arduino via USB cable**
2. **Change connection type to 'serial'**:
   ```python
   hw_controller = HardwareController(connection_type='serial')
   ```
3. **No Bluetooth needed!**

Serial connection advantages:
- ✅ More reliable
- ✅ No range limitations
- ✅ No pairing needed
- ✅ Works on any system
- ✅ Faster data transfer

## System-Specific Notes

### Kali Linux (Your System)

Kali Linux should have Bluetooth support. If not working:

```bash
# Install Bluetooth packages
sudo apt-get update
sudo apt-get install bluez bluez-tools

# Start service
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# Check status
systemctl status bluetooth
```

### Virtual Machines

If running in a VM:
- Bluetooth passthrough may not be configured
- Use Serial connection instead (USB passthrough is easier)
- Or use host system's Bluetooth if VM supports it

## Still Having Issues?

1. **Check system logs**:
   ```bash
   journalctl -u bluetooth -n 50
   dmesg | grep -i bluetooth
   ```

2. **Check permissions**:
   ```bash
   # Add user to bluetooth group
   sudo usermod -aG bluetooth $USER
   # Log out and back in
   ```

3. **Try Serial connection** - it's simpler and more reliable for development

4. **Check Arduino Serial Monitor** - make sure Arduino BLE is advertising

