#!/bin/bash
# Raspberry Pi Deployment Script for AI Tomato Sorter
# Deploys the complete system for offline operation

set -e

echo "🍅 Deploying AI Tomato Sorter to Raspberry Pi"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_warning "This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get Pi IP address
PI_IP=$(hostname -I | awk '{print $1}')
print_info "Raspberry Pi IP: $PI_IP"

# Update system
print_info "Updating Raspberry Pi system..."
sudo apt update && sudo apt upgrade -y

# Install essential packages for Pi
print_info "Installing Raspberry Pi specific packages..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    v4l-utils \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    python3-serial \
    python3-rpi.gpio \
    i2c-tools \
    libi2c-dev \
    wiringpi \
    arduino-core \
    arduino-mk

# Install camera tools
sudo apt install -y \
    raspberrypi-camera \
    raspistill \
    raspivid \
    vcgencmd

print_status "System packages installed"

# Create project directory
PROJECT_DIR="/home/$USER/tomato_sorter"
print_info "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Copy project files
print_info "Copying project files..."
cp -r /home/okidi6/Documents/GitHub/emebeded/* "$PROJECT_DIR/" 2>/dev/null || {
    print_warning "Could not copy files. Please copy manually to $PROJECT_DIR"
}

# Set up virtual environment
print_info "Setting up Python virtual environment..."
python3 -m venv tomato_sorter_env
source tomato_sorter_env/bin/activate

# Install Python dependencies optimized for Pi
print_info "Installing Python dependencies for Raspberry Pi..."
pip install --upgrade pip setuptools wheel

# Install CPU-optimized PyTorch for Pi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install \
    numpy \
    opencv-python \
    Pillow \
    Flask \
    Werkzeug \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    psutil \
    python-dateutil \
    pyyaml \
    pyserial \
    python-dotenv \
    loguru \
    pathlib2 \
    watchdog \
    flask-restx \
    gunicorn \
    waitress

print_status "Python dependencies installed"

# Create systemd service for auto-start
print_info "Creating systemd service for auto-start..."
sudo tee /etc/systemd/system/tomato-sorter.service > /dev/null << EOF
[Unit]
Description=AI Tomato Sorter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/tomato_sorter_env/bin/python $PROJECT_DIR/pi_controller.py
Restart=always
RestartSec=10
Environment=PATH=$PROJECT_DIR/tomato_sorter_env/bin
Environment=PYTHONPATH=$PROJECT_DIR

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
sudo systemctl daemon-reload
sudo systemctl enable tomato-sorter

print_status "Systemd service created and enabled"

# Create startup script
cat > start_pi.sh << 'EOF'
#!/bin/bash
# Raspberry Pi Startup Script for AI Tomato Sorter

echo "🍅 Starting AI Tomato Sorter on Raspberry Pi"
echo "============================================="

# Activate virtual environment
source tomato_sorter_env/bin/activate

# Check camera
echo "📷 Checking camera..."
if vcgencmd get_camera | grep -q "detected=1"; then
    echo "✅ Camera detected"
else
    echo "⚠️  Camera not detected - check camera connection"
fi

# Check Arduino connection
echo "🔌 Checking Arduino connection..."
if ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM* 2>/dev/null; then
    echo "✅ Arduino detected"
else
    echo "⚠️  Arduino not detected - check USB connection"
fi

# Start the main controller
echo "🚀 Starting AI Tomato Sorter..."
python pi_controller.py
EOF

chmod +x start_pi.sh

# Create configuration file
cat > pi_config.yaml << EOF
# Raspberry Pi Configuration
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

arduino:
  port: /dev/ttyUSB0
  baudrate: 115200

arm:
  home_position: [90, 90, 90, 90, 30]
  bin_positions:
    not_ready: [20, 55, 120, 80, 150]
    ready: [100, 50, 110, 80, 150]
    spoilt: [160, 60, 115, 80, 150]
  
  # Arm dimensions (mm)
  arm_length_1: 100.0
  arm_length_2: 80.0
  
  # Workspace limits
  workspace_x: [-150, 150]
  workspace_y: [50, 200]

detection:
  confidence_threshold: 0.7
  detection_interval: 2.0  # seconds

web_interface:
  host: 0.0.0.0
  port: 5000
  debug: false

# Auto-start settings
auto_start:
  enabled: true
  delay: 10  # seconds after boot
EOF

print_status "Configuration files created"

# Set up auto-start
print_info "Setting up auto-start..."
cat > ~/.bashrc << 'EOF'
# Auto-start Tomato Sorter
if [ -z "$SSH_CLIENT" ] && [ -z "$SSH_TTY" ]; then
    # Only auto-start if not SSH session
    if [ -f "/home/$USER/tomato_sorter/start_pi.sh" ]; then
        cd /home/$USER/tomato_sorter
        ./start_pi.sh &
    fi
fi
EOF

# Create desktop shortcut (if desktop environment exists)
if [ -d "/home/$USER/Desktop" ]; then
    cat > "/home/$USER/Desktop/Tomato Sorter.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=AI Tomato Sorter
Comment=Start the AI Tomato Sorter
Exec=$PROJECT_DIR/start_pi.sh
Icon=applications-utilities
Terminal=true
Categories=Utility;
EOF
    chmod +x "/home/$USER/Desktop/Tomato Sorter.desktop"
fi

print_status "Auto-start configured"

# Final setup
print_info "Final setup..."
chmod +x *.py *.sh 2>/dev/null || true

# Display completion message
echo ""
echo "🎉 Raspberry Pi Deployment Complete!"
echo "===================================="
echo ""
echo "📁 Project Directory: $PROJECT_DIR"
echo "🌐 Web Interface: http://$PI_IP:5000"
echo "🔌 Arduino Port: /dev/ttyUSB0 (or /dev/ttyACM0)"
echo "📷 Camera: /dev/video0"
echo ""
echo "🚀 To start manually:"
echo "   cd $PROJECT_DIR && ./start_pi.sh"
echo ""
echo "🔧 To check service status:"
echo "   sudo systemctl status tomato-sorter"
echo ""
echo "📊 To view logs:"
echo "   sudo journalctl -u tomato-sorter -f"
echo ""
echo "🌐 Access web interface from any device:"
echo "   http://$PI_IP:5000"
echo ""
echo "The system will auto-start on boot!"
echo "Happy sorting! 🍅🤖✨"