# Raspberry Pi Quick Reference
## AI Tomato Sorter - Essential Commands

## 🚀 **Quick Setup Commands**

### **Initial Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-venv python3-dev git v4l-utils ffmpeg libopencv-dev python3-opencv python3-serial

# Create project directory
mkdir -p /home/pi/tomato_sorter
cd /home/pi/tomato_sorter
```

### **Python Environment**
```bash
# Create virtual environment
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# Install PyTorch (CPU version for Pi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy opencv-python Flask pandas scikit-learn matplotlib pyyaml pyserial python-dotenv loguru
```

## 🔧 **Hardware Testing**

### **Test Camera**
```bash
# Pi Camera
raspistill -o test.jpg
vcgencmd get_camera

# USB Camera
ls /dev/video*
v4l2-ctl --list-devices
```

### **Test Arduino**
```bash
# Check connection
ls /dev/ttyUSB* /dev/ttyACM*

# Test communication
python3 -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
ser.write(b'STATUS\n')
print(ser.readline().decode().strip())
ser.close()
"
```

## 🎮 **System Control**

### **Start System**
```bash
cd /home/pi/tomato_sorter
source tomato_sorter_env/bin/activate
python pi_controller.py
```

### **Web Interface**
```bash
# Start web interface
python pi_web_interface.py

# Access from browser
# http://<PI_IP>:5000
```

### **Service Management**
```bash
# Start service
sudo systemctl start tomato-sorter

# Stop service
sudo systemctl stop tomato-sorter

# Check status
sudo systemctl status tomato-sorter

# View logs
sudo journalctl -u tomato-sorter -f
```

## 🔍 **Troubleshooting**

### **Check System Status**
```bash
# Check all components
python3 -c "import torch, cv2, flask, yaml; print('✅ Dependencies OK')"
ls /dev/ttyUSB* /dev/ttyACM*  # Arduino
ls /dev/video*                # Camera
vcgencmd get_camera           # Pi Camera
```

### **View Logs**
```bash
# System logs
sudo journalctl -u tomato-sorter -f

# Application logs
tail -f pi_controller.log

# System resources
htop
df -h
free -h
```

### **Reset System**
```bash
# Stop service
sudo systemctl stop tomato-sorter

# Reset environment
rm -rf tomato_sorter_env
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate
pip install -r requirements.txt
```

## 🌐 **Network Commands**

### **Get IP Address**
```bash
hostname -I
ip addr show
```

### **Test Web Interface**
```bash
# Local test
curl http://localhost:5000/pi/status

# Remote test
curl http://<PI_IP>:5000/pi/status
```

## 📱 **Web Interface URLs**

| URL | Description |
|-----|-------------|
| `http://<PI_IP>:5000/` | Main dashboard |
| `http://<PI_IP>:5000/control` | Control panel |
| `http://<PI_IP>:5000/monitor` | Monitoring |
| `http://<PI_IP>:5000/calibrate` | Calibration |
| `http://<PI_IP>:5000/pi/status` | API status |

## 🔧 **Arduino Commands**

### **Test Arduino**
```bash
# Send commands to Arduino
echo "STATUS" | sudo tee /dev/ttyUSB0
echo "HOME" | sudo tee /dev/ttyUSB0
echo "ANGLE 90 90 90 90 30" | sudo tee /dev/ttyUSB0
```

### **Arduino Commands**
- `STATUS` - Get system status
- `HOME` - Move to home position
- `ANGLE 90 90 90 90 30` - Set servo angles (base, shoulder, elbow, wrist, gripper)
- `GRIP OPEN` - Open gripper
- `GRIP CLOSE` - Close gripper
- `MOVE 100 150 1` - Move and sort
- `MOVE` commands trigger the full pick → distance adjust → bin drop sequence autonomously on the Arduino

## 📊 **System Monitoring**

### **Check Performance**
```bash
# Temperature
vcgencmd measure_temp

# Memory usage
vcgencmd get_mem arm
vcgencmd get_mem gpu

# CPU usage
top
htop
```

### **Check Services**
```bash
# All services
sudo systemctl list-units --type=service

# Specific service
sudo systemctl status tomato-sorter
sudo systemctl is-active tomato-sorter
sudo systemctl is-enabled tomato-sorter
```

## 🚨 **Emergency Commands**

### **Emergency Stop**
```bash
# Stop system immediately
sudo systemctl stop tomato-sorter

# Send emergency stop to Arduino
echo "STOP" | sudo tee /dev/ttyUSB0

# Home Arduino
echo "HOME" | sudo tee /dev/ttyUSB0
```

### **System Recovery**
```bash
# Restart service
sudo systemctl restart tomato-sorter

# Reboot system
sudo reboot

# Check system health
sudo systemctl status tomato-sorter
sudo journalctl -u tomato-sorter --since "5 minutes ago"
```

## 📁 **File Locations**

| File | Location |
|------|----------|
| **Main Controller** | `/home/pi/tomato_sorter/pi_controller.py` |
| **Web Interface** | `/home/pi/tomato_sorter/pi_web_interface.py` |
| **Configuration** | `/home/pi/tomato_sorter/pi_config.yaml` |
| **Logs** | `/home/pi/tomato_sorter/pi_controller.log` |
| **Service** | `/etc/systemd/system/tomato-sorter.service` |
| **Models** | `/home/pi/tomato_sorter/models/tomato/` |
| **Datasets** | `/home/pi/tomato_sorter/datasets/` |

## 🔄 **Daily Operations**

### **Start System**
```bash
cd /home/pi/tomato_sorter
./start_pi.sh
```

### **Stop System**
```bash
sudo systemctl stop tomato-sorter
```

### **Check Status**
```bash
curl http://localhost:5000/pi/status
```

### **View Logs**
```bash
sudo journalctl -u tomato-sorter -f
```

## 🎯 **Quick Test Sequence**

```bash
# 1. Test camera
raspistill -o test.jpg

# 2. Test Arduino
echo "STATUS" | sudo tee /dev/ttyUSB0

# 3. Test Python
source tomato_sorter_env/bin/activate
python -c "import torch, cv2, flask; print('✅ All OK')"

# 4. Test web interface
curl http://localhost:5000/pi/status

# 5. Start system
sudo systemctl start tomato-sorter
```

## 📞 **Emergency Contacts**

- **System Logs**: `sudo journalctl -u tomato-sorter -f`
- **Hardware Status**: `vcgencmd get_camera && ls /dev/ttyUSB*`
- **Network**: `hostname -I && curl http://localhost:5000/pi/status`
- **Service**: `sudo systemctl status tomato-sorter`

---

**Remember**: Always test in a safe environment first! 🍅🤖✨
