# 🛠️ AI Tomato Sorter - Setup Guide

Complete step-by-step guide to set up the AI Tomato Sorter system from scratch.

## 📋 Prerequisites

### Hardware Requirements
- **Raspberry Pi 5** (4GB RAM recommended)
- **Arduino Uno/Nano** or **ESP32**
- **3x Servo Motors** (SG90 or similar, 3-5kg torque)
- **Camera**: Pi Camera v2 or USB webcam
- **Power Supply**: 5V/2A for Arduino, 5V/3A for Pi
- **MicroSD Card**: 32GB+ (Class 10)
- **Mechanical Components**: 3D printed arm parts, gripper, sorting bins

### Software Requirements
- **Raspberry Pi OS** (64-bit, latest)
- **Arduino IDE** or **PlatformIO**
- **Python 3.8+**
- **Git**

## 🚀 Installation Steps

### Step 1: Raspberry Pi Setup

#### 1.1 Flash Raspberry Pi OS
```bash
# Download Raspberry Pi Imager
# Flash latest Raspberry Pi OS (64-bit) to microSD card
# Enable SSH and set up WiFi during imaging
```

#### 1.2 Initial Pi Configuration
```bash
# SSH into your Pi
ssh pi@<pi-ip-address>

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv git vim
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y python3-serial python3-flask
sudo apt install -y python3-matplotlib python3-pandas
```

#### 1.3 Enable Camera and GPIO
```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Enable I2C and SPI if needed
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable
# Navigate to: Interface Options → SPI → Enable

# Reboot
sudo reboot
```

### Step 2: Clone and Setup Project

#### 2.1 Clone Repository
```bash
# Clone the project
git clone <repository-url>
cd emebeded

# Create virtual environment
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 2.2 Verify Installation
```bash
# Test OpenCV
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Test serial communication
python3 -c "import serial; print('Serial module OK')"
```

### Step 3: Dataset Preparation

#### 3.1 Collect Images
```bash
# Create dataset directory structure
mkdir -p tomato_dataset/{images/{train,val,test},labels/{train,val,test}}

# Copy your images to appropriate directories
# Use your smartphone or Pi camera to capture images
# Aim for 2000-4000 images total with balanced classes
```

#### 3.2 Annotate Images
```bash
# Install LabelImg for annotation
pip install labelImg

# Start annotation tool
labelImg

# Annotate images with bounding boxes
# Save in YOLO format (.txt files)
# Class IDs: 0=not_ready, 1=ready, 2=spoilt
```

#### 3.3 Prepare Dataset
```bash
# Run data preparation script
python train/data_preparation.py \
    --source_images /path/to/your/images \
    --source_labels /path/to/your/labels \
    --output tomato_dataset \
    --validate \
    --analyze \
    --visualize
```

### Step 4: Model Training

#### 4.1 Train YOLOv8 Model
```bash
# Start training
python train/train_tomato_detector.py \
    --data data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --plot

# Monitor training progress
# Check runs/detect/tomato_sorter/ for results
```

#### 4.2 Evaluate Model
```bash
# Run validation
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/tomato_sorter/weights/best.pt')
results = model.val(data='data.yaml')
print('mAP@0.5:', results.box.map50)
"
```

### Step 5: Model Export

#### 5.1 Export for Raspberry Pi
```bash
# Export to ONNX and TFLite
python export/export_models.py \
    --model runs/detect/tomato_sorter/weights/best.pt \
    --formats onnx tflite \
    --quantize \
    --benchmark
```

#### 5.2 Test Exported Models
```bash
# Test ONNX model
python -c "
import cv2
net = cv2.dnn.readNetFromONNX('exported_models/tomato_sorter.onnx')
print('ONNX model loaded successfully')
"

# Test TFLite model
python -c "
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter('exported_models/tomato_sorter.tflite')
print('TFLite model loaded successfully')
"
```

### Step 6: Arduino Setup

#### 6.1 Hardware Connections
```
Arduino Pin Layout:
- Pin 3:  Servo 1 (Base rotation)
- Pin 5:  Servo 2 (Shoulder joint)
- Pin 6:  Servo 3 (Elbow joint)
- Pin 9:  Servo 4 (Wrist pitch)
- Pin 10: Servo 5 (Gripper)
- Pin 11: Ultrasonic TRIG
- Pin 12: Ultrasonic ECHO
- GND:    Common ground (tie external 5V ground to Arduino GND)
- 5V:     Power supply (if servos need external power)
```

#### 6.2 Upload Firmware
```bash
# Open Arduino IDE
# Load tomato_sorter_arduino.ino
# Select your Arduino board
# Upload the code

# Test serial communication
python -c "
import serial
import time
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)
ser.write(b'STATUS\n')
response = ser.readline()
print('Arduino response:', response.decode().strip())
"
```

### Step 7: Camera Calibration

#### 7.1 Calibrate Coordinate Mapping
```bash
# Run calibration script
python pi/calibration.py \
    --camera 0 \
    --output calibration.json

# Follow interactive calibration:
# 1. Click on 4 corners of your workspace
# 2. Enter real-world coordinates for each point
# 3. Save calibration data
```

#### 7.2 Test Calibration
```bash
# Test coordinate transformation
python -c "
import json
import numpy as np
from pi.calibration import CameraCalibrator

calibrator = CameraCalibrator()
calibrator.load_calibration('calibration.json')

# Test pixel to world conversion
world_coords = calibrator.pixel_to_world(320, 240)
print('World coordinates:', world_coords)
"
```

### Step 8: System Integration

#### 8.1 Test Individual Components
```bash
# Test camera
python pi/inference_pi.py \
    --model exported_models/tomato_sorter.onnx \
    --camera 0 \
    --no_display

# Test Arduino communication
python -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
ser.write(b'ANGLE 90 60 120 95 150\n')
print('Command sent to Arduino')
"
```

#### 8.2 Run Full System
```bash
# Start the complete system
python pi/inference_pi.py \
    --model exported_models/tomato_sorter.onnx \
    --camera 0 \
    --arduino_port /dev/ttyUSB0 \
    --calibration calibration.json
```

### Step 9: Web Interface Setup

#### 9.1 Start Web Interface
```bash
# Start web interface
python pi/web_interface.py \
    --host 0.0.0.0 \
    --port 5000

# Access from browser: http://<pi-ip>:5000
```

#### 9.2 Configure Auto-start (Optional)
```bash
# Create systemd service for auto-start
sudo nano /etc/systemd/system/tomato-sorter.service

# Add service configuration:
[Unit]
Description=AI Tomato Sorter
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/emebeded
ExecStart=/home/pi/emebeded/tomato_sorter_env/bin/python pi/inference_pi.py --model exported_models/tomato_sorter.onnx --camera 0 --arduino_port /dev/ttyUSB0
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable tomato-sorter.service
sudo systemctl start tomato-sorter.service
```

### Step 10: Testing and Validation

#### 10.1 Run System Tests
```bash
# Run comprehensive evaluation
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --test_data tomato_dataset \
    --num_images 50 \
    --num_trials 30 \
    --benchmark_duration 120
```

#### 10.2 Performance Monitoring
```bash
# Monitor system performance
htop

# Check GPU memory usage
vcgencmd get_mem gpu

# Monitor temperature
vcgencmd measure_temp

# Check disk usage
df -h
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Camera Issues
```bash
# Check camera detection
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Fix camera permissions
sudo usermod -a -G video pi
```

#### 2. Serial Communication Issues
```bash
# Check serial ports
ls /dev/ttyUSB* /dev/ttyACM*

# Add user to dialout group
sudo usermod -a -G dialout pi

# Test serial communication
python -c "
import serial
ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
for port in ports:
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f'Port {port} is available')
        ser.close()
    except:
        print(f'Port {port} not available')
"
```

#### 3. Model Loading Issues
```bash
# Check model file
ls -la exported_models/

# Test model loading
python -c "
import cv2
try:
    net = cv2.dnn.readNetFromONNX('exported_models/tomato_sorter.onnx')
    print('ONNX model OK')
except Exception as e:
    print('ONNX error:', e)
"
```

#### 4. Performance Issues
```bash
# Check CPU usage
htop

# Check memory usage
free -h

# Check GPU memory
vcgencmd get_mem gpu

# Optimize Pi performance
sudo raspi-config
# Advanced Options → Memory Split → 128
```

#### 5. Arduino Communication Issues
```bash
# Check Arduino connection
dmesg | grep tty

# Test Arduino response
python -c "
import serial
import time
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)
    ser.write(b'STATUS\n')
    response = ser.readline()
    print('Arduino response:', response.decode().strip())
    ser.close()
except Exception as e:
    print('Arduino error:', e)
"
```

## 📊 Performance Optimization

### Model Optimization
```bash
# Quantize model for faster inference
python export/export_models.py \
    --model best.pt \
    --quantize \
    --data_yaml data.yaml
```

### System Optimization
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options → Memory Split → 128

# Overclock Pi (optional)
sudo raspi-config
# Advanced Options → Overclock → Pi 4 2GHz

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable hciuart
```

### Network Optimization
```bash
# Optimize WiFi
sudo nano /etc/dhcpcd.conf
# Add:
# interface wlan0
# static ip_address=192.168.1.100/24
# static routers=192.168.1.1
# static domain_name_servers=8.8.8.8
```

## 🎯 Final Validation

### System Checklist
- [ ] Camera captures images correctly
- [ ] Model loads and runs inference
- [ ] Arduino responds to commands
- [ ] Coordinate calibration works
- [ ] Web interface accessible
- [ ] Sorting accuracy meets targets
- [ ] System runs stably for extended periods

### Performance Targets
- [ ] Inference time ≤ 300ms
- [ ] FPS ≥ 3
- [ ] Sorting accuracy ≥ 85%
- [ ] System uptime ≥ 95%

## 🎉 Congratulations!

Your AI Tomato Sorter system is now fully set up and ready to sort tomatoes! 

### Next Steps
1. **Fine-tune**: Adjust confidence thresholds based on performance
2. **Optimize**: Implement additional performance optimizations
3. **Scale**: Consider multiple sorting stations
4. **Monitor**: Set up continuous monitoring and logging
5. **Improve**: Collect more data and retrain model as needed

### Support
- Check the troubleshooting section for common issues
- Review logs in `tomato_sorter.log`
- Monitor system performance with the web interface
- Contact support for advanced issues

**Happy Sorting! 🍅🤖**
