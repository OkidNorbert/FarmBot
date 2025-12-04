#!/usr/bin/env python3
"""
Web Interface Launcher
======================

Launches the web interface for the AI Training System.
Handles dependencies and setup automatically.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependency(module_name, install_name=None):
    """Check if a dependency is installed, install if missing"""
    if install_name is None:
        install_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✅ {module_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {module_name}")
            return False

def setup_web_interface():
    """Setup the web interface with all dependencies"""
    print("🌐 Setting up Web Interface for AI Training System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check and install dependencies
    dependencies = [
        ("flask", "Flask>=2.0.0"),
        ("werkzeug", "Werkzeug>=2.0.0"),
        ("yaml", "PyYAML>=6.0"),
        ("PIL", "Pillow>=8.0.0")
    ]
    
    all_installed = True
    for module, install_name in dependencies:
        if not check_dependency(module, install_name):
            all_installed = False
    
    if not all_installed:
        print("❌ Some dependencies failed to install")
        return False
    
    # Create necessary directories
    directories = ['datasets', 'models', 'temp', 'templates', 'static/css', 'static/js']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Check if auto_train.py exists
    if not os.path.exists('auto_train.py'):
        print("❌ auto_train.py not found in current directory")
        print("Please ensure you're in the correct directory with all training files")
        return False
    
    print("✅ All dependencies and directories ready")
    return True

def start_web_server():
    """Start the web server"""
    print("\n🚀 Starting Web Interface...")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:5000")
    print("📁 Upload folder: datasets/")
    print("💾 Models folder: models/")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the web interface
        from web_interface import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🤖 AI Training System - Web Interface")
    print("=" * 60)
    
    # Setup
    if not setup_web_interface():
        print("❌ Setup failed")
        return 1
    
    # Start server
    if not start_web_server():
        print("❌ Failed to start web interface")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
