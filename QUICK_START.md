# 🚀 Quick Start Guide - AI Tomato Sorter

## ⚡ **One-Command Installation**

### **For Fresh Installation:**
```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/your-repo/ai-tomato-sorter/main/setup.sh | bash

# OR if you have the project files
chmod +x setup.sh
./setup.sh
```

### **Manual Installation:**
```bash
# 1. Create virtual environment
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the system
python web_interface.py
```

## 🌐 **Access the System**

### **Web Interface:**
- **URL**: http://localhost:5001
- **Features**: Live camera, object detection, model training
- **Mobile**: Works on phones and tablets

### **API Endpoints:**
- **Status**: `GET /` - System overview
- **Camera**: `GET /camera_feed` - Live camera feed
- **Detection**: `GET /tomato_detection_status` - Detection status
- **Training**: `POST /train_model` - Train new models

## 🎯 **Quick Setup Checklist**

### **✅ System Requirements:**
- [ ] Python 3.8+ installed
- [ ] 4GB+ RAM available
- [ ] 10GB+ free disk space
- [ ] USB camera or webcam
- [ ] Internet connection

### **✅ Installation Steps:**
- [ ] Run `./setup.sh` or install manually
- [ ] Activate virtual environment
- [ ] Install dependencies from requirements.txt
- [ ] Create directory structure
- [ ] Configure system settings

### **✅ First Run:**
- [ ] Start web interface: `python web_interface.py`
- [ ] Open browser: http://localhost:5001
- [ ] Test camera feed
- [ ] Upload tomato dataset
- [ ] Train initial model

## 🔧 **Essential Commands**

### **Start System:**
```bash
# Method 1: Direct start
source tomato_sorter_env/bin/activate
python web_interface.py

# Method 2: Using startup script
./start.sh

# Method 3: Background process
nohup python web_interface.py > tomato_sorter.log 2>&1 &
```

### **Stop System:**
```bash
# Find and kill process
pkill -f "python.*web_interface"

# Or if running as service
sudo systemctl stop tomato-sorter
```

### **Check Status:**
```bash
# Test web interface
curl http://localhost:5001

# Check camera
curl http://localhost:5001/camera_status

# Check detection
curl http://localhost:5001/tomato_detection_status
```

## 📊 **System Monitoring**

### **Check Logs:**
```bash
# View application logs
tail -f tomato_sorter.log

# Check system resources
htop
free -h
df -h
```

### **Test Components:**
```bash
# Test Python environment
source tomato_sorter_env/bin/activate
python -c "import torch, cv2, flask; print('All OK')"

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'No Camera')"
```

## 🎯 **Production Deployment**

### **System Service:**
```bash
# Install as system service
sudo cp tomato-sorter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tomato-sorter
sudo systemctl start tomato-sorter

# Check service status
sudo systemctl status tomato-sorter
```

### **Production Server:**
```bash
# Install production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 web_interface:app
```

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **Camera Not Working:**
```bash
# Check camera devices
ls /dev/video*

# Test with v4l2
v4l2-ctl --list-devices

# Install camera support
sudo apt install v4l-utils
```

#### **Python Import Errors:**
```bash
# Reinstall packages
pip uninstall opencv-python
pip install opencv-python-headless

# Or install system packages
sudo apt install python3-opencv
```

#### **Permission Issues:**
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x *.py *.sh
```

#### **Memory Issues:**
```bash
# Monitor memory
htop
free -h

# Reduce model complexity if needed
```

## 📚 **Documentation**

### **Complete Guides:**
- `README.md` - Main project documentation
- `FRESH_INSTALL_GUIDE.md` - Detailed installation
- `SETUP_GUIDE.md` - System setup instructions
- `API_DOCUMENTATION.md` - API reference
- `TROUBLESHOOTING.md` - Common issues

### **Configuration Files:**
- `requirements.txt` - Python dependencies
- `.env` - Environment configuration
- `setup.sh` - Automated setup script
- `start.sh` - Startup script

## 🎉 **You're Ready!**

**Your AI Tomato Sorter is now running!**

**Access at: http://localhost:5001**

**Features available:**
- ✅ Real-time object detection
- ✅ Live camera feed
- ✅ Model training and inference
- ✅ Continuous learning
- ✅ Web-based management
- ✅ Arduino integration ready

**Happy sorting! 🍅🤖✨**
