# Deployment Guide - AI Tomato Sorter

This guide explains how to deploy the AI Tomato Sorter software to your Raspberry Pi 4.

## Prerequisites
- Raspberry Pi 4 with Raspberry Pi OS installed.
- Internet connection on the Pi.
- Arduino R4 connected via USB.
- USB Camera connected.

## Step 1: Transfer Files
Copy the entire project folder to your Raspberry Pi (e.g., to `/home/pi/tomato_sorter`).

```bash
# Example using scp from your computer
scp -r emebeded/ pi@<pi-ip-address>:/home/pi/tomato_sorter
```

## Step 2: Install Dependencies
On the Raspberry Pi, run the setup script:

```bash
cd /home/pi/tomato_sorter
chmod +x setup.sh
./setup.sh
```

This will install system packages and create a Python virtual environment with all required libraries.

## Step 3: Upload Arduino Firmware
1.  Install the Arduino IDE on your computer (or the Pi).
2.  Open `arduino/tomato_arm/tomato_arm.ino`.
3.  Select your Board (Arduino R4) and Port.
4.  Click **Upload**.

## Step 4: Configure Auto-Start (Optional)
To make the system start automatically when the Pi turns on:

1.  Copy the service file:
    ```bash
    sudo cp tomato_sorter.service /etc/systemd/system/
    ```
2.  Edit the file if your path is different (default assumes `/home/pi/tomato_sorter`):
    ```bash
    sudo nano /etc/systemd/system/tomato_sorter.service
    ```
3.  Enable and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable tomato_sorter
    sudo systemctl start tomato_sorter
    ```

## Step 5: Usage
- Access the web interface at `http://<pi-ip-address>:5000`.
- Use the **Dashboard** to monitor the system.
- Use **Control** to manually move the arm.
- Enable **Automatic Mode** for autonomous sorting.

## Troubleshooting
- **Logs**: Check logs with `journalctl -u tomato_sorter -f` or view `pi_controller.log` in the project directory.
- **Camera**: Ensure no other process is using the camera.
- **Arduino**: Ensure the USB cable is connected and the port matches (default `/dev/ttyUSB0` or `/dev/ttyACM0`).
