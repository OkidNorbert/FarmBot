# 🍅 AI Tomato Sorter

A complete AI-powered tomato sorting system that uses computer vision and robotics to automatically classify and sort tomatoes into three categories: not ready, ready, and spoilt.

## 📋 Project Overview

This project implements an end-to-end tomato sorting system using:
- **Computer Vision**: YOLOv8 object detection and classification
- **Edge Computing**: Raspberry Pi 5 for real-time inference
- **Robotics**: Arduino-controlled 5-DOF robotic arm with gripper and distance sensing
- **Web Interface**: Real-time monitoring and control

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera        │    │  Raspberry Pi 5 │    │    Arduino      │
│   (Vision)      │───▶│  (AI Inference) │───▶│  (Servo Control)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Web Interface  │
                       │  (Monitoring)   │
                       └─────────────────┘
```

## 📁 Project Structure

```
emebeded/
├── train/                    # Training scripts and data preparation
│   ├── train_tomato_detector.py
│   └── data_preparation.py
├── export/                   # Model export and optimization
│   └── export_models.py
├── pi/                       # Raspberry Pi inference system
│   ├── inference_pi.py
│   ├── web_interface.py
│   └── calibration.py
├── arduino/                  # Arduino firmware
│   └── tomato_sorter_arduino.ino
├── test/                     # Testing and evaluation
│   └── evaluation.py
├── docs/                     # Documentation
│   └── README.md
├── tomato_dataset/           # Dataset (created during setup)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── requirements.txt          # Python dependencies
└── data.yaml                # Dataset configuration
```

## 🚀 Quick Start

### **Two Deployment Paths Available:**

#### **🚀 Path 1: Quick Deployment (Recommended for Beginners)**

**1. Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd emebeded

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Prepare Dataset**
```bash
# Organize your images into:
# tomato_dataset/images/{train,val,test}/
# tomato_dataset/labels/{train,val,test}/

# Validate dataset (optional)
python train/data_preparation.py \
    --source_images /path/to/your/images \
    --source_labels /path/to/your/labels \
    --output tomato_dataset \
    --validate --analyze --visualize
```

**3. Train Model (Laptop/Colab)**
```bash
# Local training (GPU recommended)
python train.py --data ./data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0

# Colab training (paste in cell)
!pip install -U ultralytics opencv-python-headless
!python train.py --data /content/tomato_dataset/data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0
```

**4. Deploy to Raspberry Pi**
```bash
# Copy model to Pi
scp runs/tomato/*/weights/best.pt pi@<pi-ip>:~/tomato_sorter/

# On Pi: Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Upload Arduino firmware
# Load arduino_servo.ino in Arduino IDE and upload to Arduino
```

**5. Calibrate Camera**
```bash
# On Pi: Run interactive calibration
python calibrate_homography.py

# Click 4 corners of your workspace rectangle
# This creates homography.npy for coordinate mapping
```

**6. Run System**
```bash
# On Pi: Start inference
python inference_pi.py --model best.pt --source 0 --serial /dev/ttyUSB0

# Test system
python test_run.py  # Logs detections to test_log.csv
```

#### **🔧 Path 2: Comprehensive System (Advanced Features)**

**1. Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd emebeded

# Create virtual environment
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Dataset Preparation**
```bash
# Prepare your dataset
python train/data_preparation.py \
    --source_images /path/to/your/images \
    --source_labels /path/to/your/labels \
    --output tomato_dataset \
    --validate \
    --analyze \
    --visualize
```

**3. Model Training**
```bash
# Train the YOLOv8 model
python train/train_tomato_detector.py \
    --data data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --plot
```

**4. Model Export**
```bash
# Export model for Raspberry Pi
python export/export_models.py \
    --model runs/detect/tomato_sorter/weights/best.pt \
    --formats onnx tflite \
    --quantize \
    --benchmark
```

**5. Raspberry Pi Deployment**
```bash
# Run inference on Raspberry Pi
python pi/inference_pi.py \
    --model exported_models/tomato_sorter.onnx \
    --camera 0 \
    --arduino_port /dev/ttyUSB0
```

**6. Web Interface**
```bash
# Start web interface
python pi/web_interface.py \
    --host 0.0.0.0 \
    --port 5000
```

## 🔧 Hardware Requirements

### Raspberry Pi 5
- **CPU**: ARM Cortex-A76 quad-core
- **RAM**: 4GB+ recommended
- **Storage**: 32GB+ microSD card
- **Camera**: Pi Camera v2 or USB webcam
- **OS**: Raspberry Pi OS (64-bit)

### Arduino/ESP32
- **Microcontroller**: Arduino Uno/Nano or ESP32
- **Servos**: 3x SG90 or similar (3-5kg torque)
- **Power**: 5V/2A power supply
- **Connections**: Serial communication with Pi

### Mechanical Components
- **Robotic Arm**: 5-DOF arm (base, shoulder, elbow, wrist, gripper) with HC-SR04 distance sensor
- **Gripper**: Soft gripper for tomatoes
- **Sorting Bins**: 3 bins for different categories
- **Workspace**: 30cm x 30cm sorting area

## 📊 Dataset Requirements

### Image Collection
- **Total Images**: 2,000-4,000 images
- **Class Distribution**:
  - Not Ready: 600-1,200 images
  - Ready: 800-1,600 images  
  - Spoilt: 400-800 images
- **Variations**: Different lighting, backgrounds, angles
- **Format**: JPG/PNG, 640x640 pixels recommended

### Annotation Format
- **Tool**: LabelImg or Roboflow
- **Format**: YOLO Darknet format
- **Classes**: 0=not_ready, 1=ready, 2=spoilt
- **Quality**: Accurate bounding boxes, consistent labeling

## 🎯 Performance Targets

### Model Performance
- **mAP@0.5**: ≥ 0.75
- **Inference Time**: ≤ 300ms per frame
- **FPS**: ≥ 3 FPS on Raspberry Pi 5
- **Model Size**: ≤ 50MB (quantized)

### System Performance
- **Sorting Accuracy**: ≥ 85%
- **Pick Success Rate**: ≥ 80%
- **System Uptime**: ≥ 95%
- **Response Time**: ≤ 1 second end-to-end

## 🔍 Testing and Evaluation

### **Quick Testing (Path 1)**
```bash
# Test inference performance
python test_run.py  # Creates test_log.csv with detection data

# Test IK solver
python ik_solver.py  # Should output: (8.03, 51.32)

# Test camera calibration
python calibrate_homography.py
```

### **Comprehensive Testing (Path 2)**
```bash
# Detection evaluation
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --test_data tomato_dataset \
    --num_images 100

# System benchmark
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --benchmark_duration 300

# Sorting evaluation
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --arduino_port /dev/ttyUSB0 \
    --num_trials 50
```

### **System Demo**
```bash
# Run complete system demo
python run_demo.py

# Check system requirements
python run_demo.py --check-requirements

# Quick system test
python run_demo.py --quick-test
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera devices
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

#### 2. Arduino Connection Issues
```bash
# Check serial ports
ls /dev/ttyUSB* /dev/ttyACM*

# Test serial communication
python -c "import serial; ser = serial.Serial('/dev/ttyUSB0', 115200); print('Arduino OK')"
```

#### 3. Model Loading Errors
```bash
# Check model file
ls -la exported_models/

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('exported_models/tomato_sorter.onnx'); print('Model OK')"
```

#### 4. Performance Issues
- **Slow Inference**: Try model quantization
- **High Memory Usage**: Reduce batch size
- **Poor Accuracy**: Check dataset quality and augmentation

### Performance Optimization

#### Model Optimization
```bash
# Quantize model for faster inference
python export/export_models.py \
    --model best.pt \
    --quantize \
    --data_yaml data.yaml
```

#### System Optimization
```bash
# Optimize Raspberry Pi performance
sudo raspi-config
# Enable GPU memory split
# Overclock CPU if needed
```

## 📈 Monitoring and Logging

### Real-time Monitoring
- **Web Interface**: http://raspberry-pi-ip:5000
- **Logs**: `tomato_sorter.log`
- **Metrics**: Detection count, accuracy, inference time

### Performance Metrics
- **Detection Rate**: Tomatoes detected per minute
- **Classification Accuracy**: Per-class precision/recall
- **System Latency**: End-to-end processing time
- **Hardware Utilization**: CPU, memory, GPU usage

## 🔒 Safety Considerations

### Hardware Safety
- **Emergency Stop**: Arduino emergency stop functionality
- **Servo Limits**: Hardware limits to prevent damage
- **Power Management**: Proper power supply and fuses
- **Mechanical Safety**: Smooth movements, soft grippers

### Software Safety
- **Error Handling**: Graceful failure handling
- **Input Validation**: Sanitize all inputs
- **Logging**: Comprehensive error logging
- **Recovery**: Automatic system recovery

## 📚 Additional Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Raspberry Pi Camera Guide](https://www.raspberrypi.org/documentation/accessories/camera/)
- [Arduino Servo Control](https://www.arduino.cc/en/Tutorial/Sweep)

### Datasets
- [Roboflow Tomato Dataset](https://roboflow.com/datasets/tomato)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [COCO Dataset](https://cocodataset.org/)

### Hardware Suppliers
- **Servos**: HobbyKing, Adafruit, SparkFun
- **Arduino**: Arduino Store, Amazon
- **Raspberry Pi**: Official Store, Adafruit
- **Mechanical Parts**: 3D printing, local fabrication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [your-email@domain.com]

## 🎉 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **OpenCV**: Computer vision library
- **Raspberry Pi Foundation**: Hardware platform
- **Arduino**: Microcontroller platform

## 📋 **Complete Usage Instructions**

### **🚀 For Beginners - Quick Path**

**Step 1: Setup Environment**
```bash
# Clone repository
git clone <repository-url>
cd emebeded

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Prepare Your Dataset**

**Option A: Extract from Archive**
```bash
# Extract and organize dataset from ZIP/TAR
python extract_dataset.py your_dataset.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup

# This will:
# - Extract your archive
# - Organize into proper structure
# - Create data.yaml automatically
# - Validate the dataset
# - Clean up temporary files
```

**Option B: Manual Dataset Preparation**
```bash
# Create dataset structure
mkdir -p tomato_dataset/{images/{train,val,test},labels/{train,val,test}}

# Add your images to appropriate folders
# Annotate with LabelImg (save as YOLO format)
# Classes: 0=not_ready, 1=ready, 2=spoilt
```

**Step 3: Train Model**
```bash
# Local training (GPU recommended)
python train.py --data ./data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0

# Colab training (paste in cell)
!pip install -U ultralytics opencv-python-headless
!python train.py --data /content/tomato_dataset/data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0
```

**Step 4: Deploy to Raspberry Pi**
```bash
# Copy model to Pi
scp runs/tomato/*/weights/best.pt pi@<pi-ip>:~/tomato_sorter/

# On Pi: Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Upload Arduino firmware
# Load arduino_servo.ino in Arduino IDE and upload
```

**Step 5: Calibrate Camera**
```bash
# On Pi: Interactive calibration
python calibrate_homography.py

# Click 4 corners of your workspace rectangle
# This creates homography.npy for coordinate mapping
```

**Step 6: Run System**
```bash
# On Pi: Start inference
python inference_pi.py --model best.pt --source 0 --serial /dev/ttyUSB0

# Test system
python test_run.py  # Creates test_log.csv
```

### **🔧 For Advanced Users - Comprehensive Path**

**Step 1: Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd emebeded

# Create virtual environment
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Dataset Preparation**

**Option A: Extract from Archive**
```bash
# Extract and organize dataset from ZIP/TAR
python extract_dataset.py your_dataset.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup
```

**Option B: Manual Dataset Preparation**
```bash
# Prepare dataset with validation
python train/data_preparation.py \
    --source_images /path/to/your/images \
    --source_labels /path/to/your/labels \
    --output tomato_dataset \
    --validate --analyze --visualize
```

**Step 3: Model Training**
```bash
# Advanced training with monitoring
python train/train_tomato_detector.py \
    --data data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --plot
```

**Step 4: Model Export**
```bash
# Export with quantization
python export/export_models.py \
    --model runs/detect/tomato_sorter/weights/best.pt \
    --formats onnx tflite \
    --quantize --benchmark
```

**Step 5: Pi Deployment**
```bash
# Run inference with calibration
python pi/inference_pi.py \
    --model exported_models/tomato_sorter.onnx \
    --camera 0 \
    --arduino_port /dev/ttyUSB0 \
    --calibration calibration.json
```

**Step 6: Web Interface**
```bash
# Start monitoring interface
python pi/web_interface.py --host 0.0.0.0 --port 5000
# Access at: http://<pi-ip>:5000
```

### **📦 Dataset Extraction**

**Extract from ZIP Archive:**
```bash
# Basic extraction
python extract_dataset.py dataset.zip

# Full extraction with organization
python extract_dataset.py dataset.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup
```

**Extract from TAR Archive:**
```bash
# Extract TAR/TAR.GZ/TAR.BZ2
python extract_dataset.py dataset.tar.gz \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup
```

**Supported Archive Formats:**
- ✅ **ZIP**: `.zip`
- ✅ **TAR**: `.tar`
- ✅ **TAR.GZ**: `.tar.gz`, `.tgz`
- ✅ **TAR.BZ2**: `.tar.bz2`, `.tbz2`

**What the extraction utility does:**
1. **Extracts** your archive to a temporary directory
2. **Organizes** images and labels into proper train/val/test structure
3. **Creates** data.yaml configuration file automatically
4. **Validates** the dataset structure and content
5. **Cleans up** temporary files (optional)

### **🧪 Testing Your System**

**Quick Tests:**
```bash
# Test IK solver
python ik_solver.py  # Should output: (8.03, 51.32)

# Test camera
python calibrate_homography.py

# Test inference
python test_run.py
```

**Comprehensive Tests:**
```bash
# Run system demo
python run_demo.py

# Check requirements
python run_demo.py --check-requirements

# Quick test
python run_demo.py --quick-test
```

**Performance Evaluation:**
```bash
# Detection evaluation
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --test_data tomato_dataset \
    --num_images 100

# System benchmark
python test/evaluation.py \
    --model exported_models/tomato_sorter.onnx \
    --benchmark_duration 300
```

### **🔧 Hardware Setup**

**Arduino Connections:**
```
Pin 3:  Servo 1 (Base rotation)
Pin 5:  Servo 2 (Shoulder joint)  
Pin 6:  Servo 3 (Elbow joint)
Pin 9:  Servo 4 (Wrist pitch)
Pin 10: Servo 5 (Gripper)
Pin 11: HC-SR04 TRIG
Pin 12: HC-SR04 ECHO
GND:    Common ground (external 5V ↔ Arduino)
5V:     Power supply
```

**Raspberry Pi:**
- **Camera**: USB webcam or Pi Camera
- **Serial**: USB connection to Arduino
- **Power**: 5V/3A power supply

### **⚙️ Configuration**

**Adjust in `inference_pi.py`:**
```python
L1 = 10.0  # Link 1 length in cm
L2 = 10.0  # Link 2 length in cm  
CONF_THRESH = 0.35  # Detection confidence
```

**Adjust in `arduino_servo.ino`:**
```cpp
const int stepDelay = 20;  // ms between steps
const int stepSize = 1;    // degrees per step
```

### **📊 Performance Targets**

- ✅ **Detection**: ≥3 FPS on Raspberry Pi 5
- ✅ **Accuracy**: mAP@0.5 ≥ 0.75
- ✅ **Sorting**: ≥85% correct classification
- ✅ **Stability**: Continuous operation without crashes

### **🆘 Troubleshooting**

**Common Issues:**
1. **Camera not detected**: Check `/dev/video*` devices
2. **Arduino not connected**: Check `/dev/ttyUSB*` ports
3. **Model loading fails**: Verify model file exists
4. **Poor performance**: Adjust confidence thresholds

**Get Help:**
- Check `QUICK_START.md` for quick deployment
- Check `docs/SETUP_GUIDE.md` for detailed setup
- Run `python run_demo.py` for system status
- Check logs in `tomato_sorter.log`

---

**Happy Sorting! 🍅🤖**
