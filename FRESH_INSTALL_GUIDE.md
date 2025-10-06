# 🚀 Fresh Installation Guide - AI Tomato Sorter

## 📋 **Complete Setup Instructions**

### **System Requirements:**
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Camera**: USB camera or built-in webcam

## 🔧 **Step 1: System Preparation**

### **Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Install camera support
sudo apt install -y v4l-utils

# Install development tools
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module
sudo apt install -y libv4l-dev v4l-utils
```

### **macOS:**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python3 git curl wget
brew install opencv
```

### **Windows:**
```bash
# Install Python from python.org
# Install Git from git-scm.com
# Install Visual Studio Build Tools
```

## 🐍 **Step 2: Python Environment Setup**

### **Create Project Directory:**
```bash
# Create project directory
mkdir -p ~/ai_tomato_sorter
cd ~/ai_tomato_sorter

# Clone or download project files
# (Copy all project files to this directory)
```

### **Create Virtual Environment:**
```bash
# Create virtual environment
python3 -m venv tomato_sorter_env

# Activate virtual environment
source tomato_sorter_env/bin/activate  # Linux/macOS
# OR
tomato_sorter_env\Scripts\activate     # Windows
```

### **Upgrade pip:**
```bash
pip install --upgrade pip setuptools wheel
```

## 📦 **Step 3: Install Dependencies**

### **Install Core Requirements:**
```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install individually
pip install torch torchvision
pip install opencv-python
pip install Flask
pip install numpy pandas
pip install Pillow matplotlib
pip install scikit-learn
pip install pyserial
```

### **Verify Installation:**
```bash
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import flask; print('Flask:', flask.__version__)"
```

## 🗂️ **Step 4: Project Structure Setup**

### **Create Directory Structure:**
```bash
# Create necessary directories
mkdir -p datasets/tomato
mkdir -p models/tomato
mkdir -p temp
mkdir -p learning_data
mkdir -p templates
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images
mkdir -p arduino
mkdir -p docs
```

### **Set Permissions:**
```bash
# Make scripts executable
chmod +x *.py
chmod +x *.sh

# Set directory permissions
chmod 755 datasets models temp learning_data
```

## 🎯 **Step 5: Dataset Setup**

### **Prepare Dataset:**
```bash
# Create dataset structure
mkdir -p datasets/tomato/train/ripe
mkdir -p datasets/tomato/train/unripe
mkdir -p datasets/tomato/train/old
mkdir -p datasets/tomato/train/damaged

mkdir -p datasets/tomato/val/ripe
mkdir -p datasets/tomato/val/unripe
mkdir -p datasets/tomato/val/old
mkdir -p datasets/tomato/val/damaged
```

### **Add Your Dataset:**
```bash
# Copy your tomato images to appropriate folders
# Example structure:
# datasets/tomato/train/ripe/*.jpg
# datasets/tomato/train/unripe/*.jpg
# datasets/tomato/train/old/*.jpg
# datasets/tomato/train/damaged/*.jpg
```

## 🔧 **Step 6: Configuration**

### **Create Configuration File:**
```bash
# Create .env file
cat > .env << EOF
# AI Tomato Sorter Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=models/tomato/
DATASET_PATH=datasets/tomato/
TEMP_PATH=temp/

# Camera Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Arduino Configuration
ARDUINO_PORT=/dev/ttyUSB0
ARDUINO_BAUD=9600

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=5001
EOF
```

### **Create Startup Script:**
```bash
# Create start.sh
cat > start.sh << 'EOF'
#!/bin/bash
# AI Tomato Sorter Startup Script

echo "🌐 Starting AI Tomato Sorter"
echo "================================"

# Activate virtual environment
source tomato_sorter_env/bin/activate

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check Python version
python --version

# Check required packages
echo "🔍 Checking dependencies..."
python -c "import torch, cv2, flask; print('✅ All dependencies available')"

# Start web interface
echo "🚀 Starting web interface..."
python web_interface.py
EOF

chmod +x start.sh
```

## 🚀 **Step 7: First Run**

### **Test Installation:**
```bash
# Activate environment
source tomato_sorter_env/bin/activate

# Test basic functionality
python -c "
import sys
sys.path.append('.')
from web_interface import app
print('✅ Web interface imported successfully')
"

# Test camera
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera available')
    cap.release()
else:
    print('⚠️  Camera not available')
"
```

### **Start the System:**
```bash
# Method 1: Using startup script
./start.sh

# Method 2: Direct Python
source tomato_sorter_env/bin/activate
python web_interface.py

# Method 3: Background process
nohup python web_interface.py > tomato_sorter.log 2>&1 &
```

## 🌐 **Step 8: Access Web Interface**

### **Open in Browser:**
- **Local**: http://localhost:5001
- **Network**: http://YOUR_IP:5001

### **Test Features:**
1. **Home Page**: Check system status
2. **Dataset Management**: Upload and organize datasets
3. **Model Training**: Train new models
4. **Live Camera**: Test camera feed and detection
5. **Inference**: Test image classification

## 🔧 **Step 9: Arduino Setup (Optional)**

### **Install Arduino IDE:**
```bash
# Download from arduino.cc
# Install Arduino IDE
# Install required libraries:
# - Servo
# - Wire
# - SoftwareSerial
```

### **Upload Arduino Code:**
```bash
# Open tomato_sorter_arduino.ino in Arduino IDE
# Select your board (Arduino Uno/Nano)
# Upload the code
# Connect to Raspberry Pi via USB
```

## 📊 **Step 10: Production Deployment**

### **System Service (Linux):**
```bash
# Create systemd service
sudo tee /etc/systemd/system/tomato-sorter.service > /dev/null << EOF
[Unit]
Description=AI Tomato Sorter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ai_tomato_sorter
ExecStart=$HOME/ai_tomato_sorter/tomato_sorter_env/bin/python web_interface.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tomato-sorter
sudo systemctl start tomato-sorter
```

### **Production Server:**
```bash
# Install production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 web_interface:app
```

## 🔍 **Step 11: Verification**

### **Check System Status:**
```bash
# Check if service is running
curl http://localhost:5001

# Check camera status
curl http://localhost:5001/camera_status

# Check detection status
curl http://localhost:5001/tomato_detection_status
```

### **Test All Features:**
1. ✅ **Web Interface**: Accessible at http://localhost:5001
2. ✅ **Camera Feed**: Live video streaming
3. ✅ **Object Detection**: Real-time tomato detection
4. ✅ **Model Training**: Dataset upload and training
5. ✅ **Inference**: Image classification
6. ✅ **Continuous Learning**: Automatic model updates

## 🛠️ **Troubleshooting**

### **Common Issues:**

#### **Camera Not Working:**
```bash
# Check camera devices
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Install camera drivers if needed
sudo apt install v4l2loopback-dkms
```

#### **Python Import Errors:**
```bash
# Reinstall problematic packages
pip uninstall opencv-python
pip install opencv-python-headless

# Or install system packages
sudo apt install python3-opencv
```

#### **Permission Issues:**
```bash
# Fix permissions
sudo chown -R $USER:$USER ~/ai_tomato_sorter
chmod +x *.py *.sh
```

#### **Memory Issues:**
```bash
# Monitor memory usage
htop
free -h

# Reduce model complexity if needed
```

## 📚 **Additional Resources**

### **Documentation:**
- `README.md` - Main project documentation
- `SETUP_GUIDE.md` - Detailed setup instructions
- `API_DOCUMENTATION.md` - API reference
- `TROUBLESHOOTING.md` - Common issues and solutions

### **Support:**
- Check logs: `tail -f tomato_sorter.log`
- Monitor system: `htop`, `iotop`
- Test components individually
- Check network connectivity

---

## 🎉 **Installation Complete!**

**Your AI Tomato Sorter is now ready for production use!**

**Access the system at: http://localhost:5001**

**Features available:**
- ✅ Real-time object detection
- ✅ Live camera feed
- ✅ Model training and inference
- ✅ Continuous learning
- ✅ Web-based management interface
- ✅ Arduino integration ready

**Happy sorting! 🍅🤖✨**
