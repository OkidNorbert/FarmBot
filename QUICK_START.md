# 🚀 AI Tomato Sorter - Quick Start Guide

## 📋 **Enhanced System with New Components**

The system now includes additional streamlined components for faster deployment:

### 🆕 **New Files Added:**
- `train.py` - Simplified training wrapper using Ultralytics API
- `inference_pi.py` - Enhanced Pi-side inference with robust model API
- `ik_solver.py` - Simple 2-link inverse kinematics helper
- `arduino_servo.ino` - Streamlined Arduino servo control
- `calibrate_homography.py` - Interactive camera calibration
- `test_run.py` - Quick testing and evaluation script

## 🚀 **Quick Start Commands**

### **1. Setup Environment**
```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Prepare Dataset**
```bash
# Organize your images into:
# tomato_dataset/images/{train,val,test}/
# tomato_dataset/labels/{train,val,test}/

# Validate dataset
python train/data_preparation.py \
    --source_images /path/to/images \
    --source_labels /path/to/labels \
    --output tomato_dataset \
    --validate --analyze --visualize
```

### **3. Train Model (Laptop/Colab)**
```bash
# Local training (GPU recommended)
python train.py --data ./data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0

# Colab training (paste in cell)
!pip install -U ultralytics opencv-python-headless
!python train.py --data /content/tomato_dataset/data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0
```

### **4. Deploy to Raspberry Pi**
```bash
# Copy model to Pi
scp runs/tomato/*/weights/best.pt pi@<pi-ip>:~/tomato_sorter/

# On Pi: Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Upload Arduino firmware
# Load arduino_servo.ino in Arduino IDE and upload to Arduino
```

### **5. Calibrate Camera**
```bash
# On Pi: Run interactive calibration
python calibrate_homography.py

# Click 4 corners of your workspace rectangle
# This creates homography.npy for coordinate mapping
```

### **6. Run System**
```bash
# On Pi: Start inference
python inference_pi.py --model best.pt --source 0 --serial /dev/ttyUSB0

# Test system
python test_run.py  # Logs detections to test_log.csv
```

## 🔧 **Hardware Setup**

### **Arduino Connections:**
```
Pin 3:  Servo 1 (Base rotation)
Pin 5:  Servo 2 (Arm joint)  
Pin 6:  Servo 3 (Gripper)
GND:    Common ground
5V:     Power supply
```

### **Raspberry Pi:**
- **Camera**: USB webcam or Pi Camera
- **Serial**: USB connection to Arduino
- **Power**: 5V/3A power supply

## ⚙️ **Configuration**

### **Adjust in `inference_pi.py`:**
```python
L1 = 10.0  # Link 1 length in cm
L2 = 10.0  # Link 2 length in cm
CONF_THRESH = 0.35  # Detection confidence threshold
```

### **Adjust in `arduino_servo.ino`:**
```cpp
const int stepDelay = 20;  // ms between servo steps
const int stepSize = 1;    // degrees per step
```

## 🎯 **Usage Workflow**

1. **Collect Data**: Take 2000-4000 tomato images with varied lighting
2. **Annotate**: Use LabelImg to create YOLO format annotations
3. **Train**: Run training on laptop/Colab with GPU
4. **Deploy**: Copy model to Raspberry Pi
5. **Calibrate**: Run camera calibration for coordinate mapping
6. **Test**: Run inference and adjust parameters
7. **Deploy**: Start full sorting system

## 📊 **Performance Monitoring**

### **Test System:**
```bash
python test_run.py  # Creates test_log.csv with detection data
```

### **Monitor Performance:**
- **Inference Time**: Should be ≤300ms per frame
- **Detection Accuracy**: Monitor confidence scores
- **Sorting Success**: Track Arduino command execution
- **System Stability**: Check for memory leaks or crashes

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **Camera Not Detected:**
   ```bash
   ls /dev/video*
   # Try different camera indices: 0, 1, 2
   ```

2. **Arduino Not Connected:**
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM*
   # Check serial port permissions
   sudo usermod -a -G dialout pi
   ```

3. **Model Loading Issues:**
   ```bash
   # Check model file exists
   ls -la best.pt
   # Test model loading
   python -c "from ultralytics import YOLO; model = YOLO('best.pt')"
   ```

4. **Poor Detection:**
   - Adjust confidence threshold in `inference_pi.py`
   - Check lighting conditions
   - Retrain with more diverse data

## 🎉 **Success Metrics**

- ✅ **Detection**: ≥3 FPS on Raspberry Pi 5
- ✅ **Accuracy**: mAP@0.5 ≥ 0.75
- ✅ **Sorting**: ≥85% correct classification
- ✅ **Stability**: Continuous operation without crashes

## 📚 **Additional Resources**

- **Full Documentation**: `docs/README.md`
- **Setup Guide**: `docs/SETUP_GUIDE.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Demo System**: `python run_demo.py`

---

**Your AI Tomato Sorter is ready to revolutionize agricultural automation! 🍅🤖**
