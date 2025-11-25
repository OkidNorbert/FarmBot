# Arduino CLI Quick Guide

Arduino CLI is installed and ready to use!

## Quick Upload Command

From the project root directory:

```bash
# Make sure PATH includes Arduino CLI
export PATH="$HOME/.local/bin:$PATH"

# Upload the test sketch
cd arduino/test_all_components
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:renesas_uno:unor4wifi .
```

## Or Use the Helper Script

```bash
./arduino/upload_sketch.sh
```

This script will:
- Check permissions
- Fix them temporarily if needed
- Compile and upload the sketch

## Fix Permissions Permanently

To avoid permission issues every time, add yourself to the dialout group:

```bash
sudo usermod -a -G dialout $USER
```

Then **log out and log back in** for the change to take effect.

## Useful Arduino CLI Commands

### List connected boards
```bash
arduino-cli board list
```

### Compile a sketch
```bash
arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi arduino/test_all_components
```

### Upload a sketch
```bash
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:renesas_uno:unor4wifi arduino/test_all_components
```

### Monitor serial output
```bash
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200
```

### Install libraries
```bash
arduino-cli lib install "Adafruit_VL53L0X"
arduino-cli lib install Servo
```

### List installed libraries
```bash
arduino-cli lib list
```

## Troubleshooting

### Permission Denied
```bash
# Temporary fix (resets after unplugging)
sudo chmod 666 /dev/ttyACM0

# Permanent fix
sudo usermod -a -G dialout $USER
# Then log out and log back in
```

### Board Not Found
- Check USB connection
- Try: `arduino-cli board list`
- Make sure Arduino is in programming mode

### Library Not Found
- Search: `arduino-cli lib search <name>`
- Install: `arduino-cli lib install <name>`

## Test Your Upload

After uploading, open serial monitor:

```bash
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200
```

Then type commands:
- `TEST ALL` - Full test sequence
- `STATUS` - Component status
- `HELP` - Show all commands

