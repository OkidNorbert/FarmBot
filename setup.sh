#!/bin/bash
# AI Tomato Sorter - Automated Setup Script
# For fresh installation on any Linux system

set -e  # Exit on any error

echo "🌐 AI Tomato Sorter - Automated Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Get current directory
PROJECT_DIR=$(pwd)
print_info "Project directory: $PROJECT_DIR"

# Step 1: Update system packages
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System packages updated"

# Step 2: Install essential packages
print_info "Installing essential packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libv4l-dev \
    v4l-utils \
    v4l2loopback-dkms \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0

print_status "Essential packages installed"

# Step 3: Create virtual environment
print_info "Creating Python virtual environment..."
if [ -d "tomato_sorter_env" ]; then
    print_warning "Virtual environment already exists, removing..."
    rm -rf tomato_sorter_env
fi

python3 -m venv tomato_sorter_env
print_status "Virtual environment created"

# Step 4: Activate virtual environment
print_info "Activating virtual environment..."
source tomato_sorter_env/bin/activate
print_status "Virtual environment activated"

# Step 5: Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "Pip upgraded"

# Step 6: Install Python dependencies
print_info "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing core packages..."
    pip install torch torchvision
    pip install opencv-python
    pip install Flask
    pip install numpy pandas
    pip install Pillow matplotlib
    pip install scikit-learn
    pip install pyserial
    pip install python-dotenv
    pip install loguru
    print_status "Core dependencies installed"
fi

# Step 7: Create directory structure
print_info "Creating project directory structure..."
mkdir -p datasets/tomato/{train,val}/{ripe,unripe,old,damaged}
mkdir -p models/tomato
mkdir -p temp
mkdir -p learning_data
mkdir -p templates
mkdir -p static/{css,js,images}
mkdir -p arduino
mkdir -p docs
mkdir -p logs
print_status "Directory structure created"

# Step 8: Set permissions
print_info "Setting permissions..."
chmod +x *.py 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true
chmod 755 datasets models temp learning_data templates static
print_status "Permissions set"

# Step 9: Create configuration files
print_info "Creating configuration files..."

# Create .env file
cat > .env << EOF
# AI Tomato Sorter Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=$(openssl rand -hex 32)

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

# Create startup script
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
python -c "import torch, cv2, flask; print('✅ All dependencies available')" || {
    echo "❌ Missing dependencies, please run setup.sh again"
    exit 1
}

# Start web interface
echo "🚀 Starting web interface..."
python web_interface.py
EOF

chmod +x start.sh

# Create systemd service
cat > tomato-sorter.service << EOF
[Unit]
Description=AI Tomato Sorter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/tomato_sorter_env/bin/python web_interface.py
Restart=always
RestartSec=10
Environment=PATH=$PROJECT_DIR/tomato_sorter_env/bin

[Install]
WantedBy=multi-user.target
EOF

print_status "Configuration files created"

# Step 10: Test installation
print_info "Testing installation..."

# Test Python imports
python -c "
import sys
sys.path.append('.')
try:
    from web_interface import app
    print('✅ Web interface imported successfully')
except Exception as e:
    print(f'❌ Web interface import failed: {e}')
    sys.exit(1)
"

# Test camera
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera available')
    cap.release()
else:
    print('⚠️  Camera not available (this is normal if no camera is connected)')
"

print_status "Installation test completed"

# Step 11: Create systemd service (optional)
print_info "Setting up system service..."
if command -v systemctl &> /dev/null; then
    sudo cp tomato-sorter.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tomato-sorter
    print_status "System service created and enabled"
    print_info "To start service: sudo systemctl start tomato-sorter"
    print_info "To check status: sudo systemctl status tomato-sorter"
else
    print_warning "systemctl not available, skipping service setup"
fi

# Step 12: Final verification
print_info "Final verification..."

# Check if all required files exist
required_files=("web_interface.py" "requirements.txt" "start.sh")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "Found $file"
    else
        print_error "Missing $file"
    fi
done

# Check virtual environment
if [ -d "tomato_sorter_env" ]; then
    print_status "Virtual environment exists"
else
    print_error "Virtual environment not found"
fi

# Check Python packages
source tomato_sorter_env/bin/activate
python -c "
import torch, cv2, flask, numpy, pandas
print('✅ All core packages available')
"

print_status "Final verification completed"

# Step 13: Display completion message
echo ""
echo "🎉 AI Tomato Sorter Setup Complete!"
echo "=================================="
echo ""
echo "📁 Project Directory: $PROJECT_DIR"
echo "🐍 Virtual Environment: tomato_sorter_env"
echo "🌐 Web Interface: http://localhost:5001"
echo ""
echo "🚀 To start the system:"
echo "   ./start.sh"
echo ""
echo "🔧 To start as service:"
echo "   sudo systemctl start tomato-sorter"
echo ""
echo "📊 To check status:"
echo "   curl http://localhost:5001"
echo ""
echo "📚 Documentation:"
echo "   - README.md"
echo "   - FRESH_INSTALL_GUIDE.md"
echo "   - API_DOCUMENTATION.md"
echo ""
echo "🎯 Next steps:"
echo "   1. Add your tomato dataset to datasets/tomato/"
echo "   2. Train your model using the web interface"
echo "   3. Test the live camera feed"
echo "   4. Configure Arduino integration (optional)"
echo ""
echo "Happy sorting! 🍅🤖✨"
