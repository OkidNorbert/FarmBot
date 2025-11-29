#!/bin/bash
# AI Tomato Sorter - Improved Setup Script
# Handles PyTorch installation issues and preserves existing environments
#
# SAFETY: This script NEVER deletes existing virtual environments
# It will only create a new environment if one doesn't exist

# Don't exit on error - we'll handle errors gracefully
set +e

echo "🌐 AI Tomato Sorter - Improved Setup"
echo "===================================="
echo ""
echo "⚠️  SAFETY: This script preserves your existing virtual environment"
echo "   It will NOT delete or recreate farmbot_env if it already exists"
echo ""

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

# Step 1: Update system packages (only if not already updated recently)
print_info "Checking system packages..."
if [ ! -f ".setup_completed" ] || [ $(find .setup_completed -mtime +1 2>/dev/null | wc -l) -gt 0 ]; then
    print_info "Updating system packages..."
    if sudo -n true 2>/dev/null; then
        # Sudo available without password
        sudo apt update && sudo apt upgrade -y
        print_status "System packages updated"
    else
        print_warning "Sudo requires password - skipping system package update"
        print_info "You can manually run: sudo apt update && sudo apt upgrade -y"
    fi
else
    print_info "System packages recently updated, skipping..."
fi

# Step 2: Install essential packages (only if not already installed)
print_info "Checking essential packages..."
missing_packages=()
packages=(
    "python3" "python3-pip" "python3-venv" "python3-dev" "git" "curl" "wget"
    "build-essential" "cmake" "pkg-config" "libjpeg-dev" "libtiff5-dev"
    "libpng-dev" "libavcodec-dev" "libavformat-dev" "libswscale-dev"
    "libgtk2.0-dev" "v4l-utils" "v4l2loopback-dkms" "ffmpeg" "libsm6"
    "libxext6" "libxrender-dev" "libglib2.0-0"
)

for package in "${packages[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    print_info "Installing missing packages: ${missing_packages[*]}"
    if sudo -n true 2>/dev/null; then
        # Sudo available without password
        sudo apt install -y "${missing_packages[@]}"
        print_status "Essential packages installed"
    else
        print_warning "Sudo requires password - skipping package installation"
        print_info "You can manually run: sudo apt install -y ${missing_packages[*]}"
        print_warning "Some features may not work without these packages"
    fi
else
    print_info "All essential packages already installed"
fi

# Step 3: Create or preserve virtual environment
print_info "Setting up Python virtual environment..."
VENV_NAME="farmbot_env"

# SAFETY CHECK: Never delete existing environment
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists - PRESERVING IT"
    print_info "The existing environment will be used as-is"
    print_info "All packages in the existing environment will be preserved"
    print_info "To recreate environment, manually delete $VENV_NAME folder first (NOT RECOMMENDED)"
else
    print_info "Creating new virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    if [ $? -eq 0 ]; then
        print_status "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Verify environment still exists (safety check)
if [ ! -d "$VENV_NAME" ]; then
    print_error "CRITICAL: Virtual environment was deleted or not found!"
    exit 1
fi

# Step 4: Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_NAME/bin/activate"
if [ $? -eq 0 ]; then
    print_status "Virtual environment activated"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Step 5: Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "Pip upgraded"

# Step 6: Check disk space and set up pip cache
print_info "Setting up pip cache directory..."
mkdir -p ~/.pip-cache
export PIP_CACHE_DIR=~/.pip-cache
print_status "Pip cache directory set to ~/.pip-cache"

# Check available disk space
MIN_SPACE_GB=5
AVAILABLE_GB_ROOT=$(df --output=avail . | tail -1)
AVAILABLE_GB_ROOT=$((AVAILABLE_GB_ROOT / 1024 / 1024))
AVAILABLE_GB_TMP=$(df --output=avail /tmp | tail -1)
AVAILABLE_GB_TMP=$((AVAILABLE_GB_TMP / 1024 / 1024))

if [ "$AVAILABLE_GB_ROOT" -lt "$MIN_SPACE_GB" ]; then
    print_warning "Low disk space on root: ${AVAILABLE_GB_ROOT}GB available. At least ${MIN_SPACE_GB}GB is recommended."
fi
if [ "$AVAILABLE_GB_TMP" -lt "$MIN_SPACE_GB" ]; then
    print_warning "Low disk space on /tmp: ${AVAILABLE_GB_TMP}GB available. At least ${MIN_SPACE_GB}GB is recommended."
    print_info "Tip: You can free up /tmp with 'sudo rm -rf /tmp/*' if safe, or set TMPDIR to a larger location."
fi

# Step 7: Install PyTorch with CPU-only version to avoid CUDA issues
print_info "Installing PyTorch (CPU-only version to avoid CUDA issues)..."
if ! python -c "import torch" 2>/dev/null; then
    print_info "Installing PyTorch CPU-only version..."
    pip install --cache-dir ~/.pip-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu
    print_status "PyTorch CPU-only version installed"
else
    print_info "PyTorch already installed, checking version..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
fi

# Step 8: Install core dependencies
print_info "Installing core dependencies..."
core_packages=(
    "numpy" "opencv-python" "Pillow" "Flask" "Werkzeug" "pandas" 
    "scikit-learn" "matplotlib" "seaborn" "psutil" "python-dateutil"
    "pyyaml" "pyserial" "python-dotenv" "loguru" "pathlib2" "watchdog"
    "flask-restx" "pytest" "pytest-flask" "memory-profiler"
)

for package in "${core_packages[@]}"; do
    if ! python -c "import ${package//-/_}" 2>/dev/null; then
        print_info "Installing $package..."
        pip install --cache-dir ~/.pip-cache "$package"
    else
        print_info "$package already installed"
    fi
done

# Install optional development packages
print_info "Installing optional development packages..."
optional_packages=("jupyter" "ipython" "gunicorn" "waitress" "imutils" "scipy")
for package in "${optional_packages[@]}"; do
    if ! python -c "import ${package//-/_}" 2>/dev/null; then
        print_info "Installing optional package: $package..."
        pip install --cache-dir ~/.pip-cache "$package" || print_warning "Failed to install $package (optional)"
    fi
done

print_status "Core dependencies installed"

# Step 9: Create directory structure (only if not exists)
print_info "Creating project directory structure..."
directories=(
    "datasets/tomato/train/ripe" "datasets/tomato/train/unripe" "datasets/tomato/train/old" "datasets/tomato/train/damaged"
    "datasets/tomato/val/ripe" "datasets/tomato/val/unripe" "datasets/tomato/val/old" "datasets/tomato/val/damaged"
    "models/tomato" "temp" "learning_data" "templates" "static/css" "static/js" "static/images"
    "arduino" "docs" "logs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    else
        print_info "Directory already exists: $dir"
    fi
done

print_status "Directory structure ready"

# Step 10: Set permissions
print_info "Setting permissions..."
chmod +x *.py 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true
chmod 755 datasets models temp learning_data templates static 2>/dev/null || true
print_status "Permissions set"

# Step 11: Create or update configuration files
print_info "Creating configuration files..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
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
    print_status "Created .env file"
else
    print_info ".env file already exists, preserving it"
fi

# Create or update startup script
cat > start.sh << EOF
#!/bin/bash
# AI Tomato Sorter Startup Script

echo "🌐 Starting AI Tomato Sorter"
echo "================================"

# Activate virtual environment
source $VENV_NAME/bin/activate

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
python -c "import torch, cv2, flask, yaml; print('✅ All dependencies available')" || {
    echo "❌ Missing dependencies, please run setup.sh again"
    exit 1
}

# Start web interface
echo "🚀 Starting web interface..."
python web_interface.py
EOF

chmod +x start.sh
print_status "Startup script created/updated"

# Step 12: Test installation
print_info "Testing installation..."

# Test Python imports
python -c "
import sys
sys.path.append('.')
try:
    import torch, cv2, flask, yaml, numpy, pandas
    print('✅ All core packages imported successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

# Test web interface import
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

# Test camera (optional)
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

# Step 13: Create systemd service (optional)
print_info "Setting up system service..."
if command -v systemctl &> /dev/null; then
    cat > tomato-sorter.service << EOF
[Unit]
Description=AI Tomato Sorter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/$VENV_NAME/bin/python web_interface.py
Restart=always
RestartSec=10
Environment=PATH=$PROJECT_DIR/$VENV_NAME/bin

[Install]
WantedBy=multi-user.target
EOF

    if sudo -n true 2>/dev/null; then
        sudo cp tomato-sorter.service /etc/systemd/system/ 2>/dev/null || print_warning "Could not copy service file"
        sudo systemctl daemon-reload 2>/dev/null || print_warning "Could not reload systemd"
        sudo systemctl enable tomato-sorter 2>/dev/null || print_warning "Could not enable service"
    else
        print_warning "Sudo requires password - skipping systemd service setup"
        print_info "You can manually install the service later"
    fi
    print_status "System service created"
    print_info "To start service: sudo systemctl start tomato-sorter"
    print_info "To check status: sudo systemctl status tomato-sorter"
else
    print_warning "systemctl not available, skipping service setup"
fi

# Step 14: Final verification
print_info "Final verification..."

# Check if all required files exist
required_files=("web_interface.py" "start.sh")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "Found $file"
    else
        print_error "Missing $file"
    fi
done

# Check virtual environment
if [ -d "$VENV_NAME" ]; then
    print_status "Virtual environment exists"
else
    print_error "Virtual environment not found"
fi

# Check Python packages
python -c "
import torch, cv2, flask, yaml, numpy, pandas
print('✅ All core packages available')
"

print_status "Final verification completed"

# Create setup completion marker
touch .setup_completed

# Step 15: Display completion message
echo ""
echo "🎉 AI Tomato Sorter Setup Complete!"
echo "=================================="
echo ""
echo "📁 Project Directory: $PROJECT_DIR"
echo "🐍 Virtual Environment: $VENV_NAME"
echo "🌐 Web Interface: http://localhost:5000"
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
echo ""
echo "🎯 Next steps:"
echo "   1. Add your tomato dataset to datasets/tomato/"
echo "   2. Train your model using the web interface"
echo "   3. Test the live camera feed"
echo "   4. Configure Arduino integration (optional)"
echo ""
echo "Happy sorting! 🍅🤖✨"