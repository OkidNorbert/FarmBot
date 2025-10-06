#!/usr/bin/env python3
"""
Complete Tomato Classification GUI Launcher
Ensures all dependencies and components are ready
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_environment():
    """Check if we're in the right environment"""
    print("🔍 Checking environment...")
    
    # Check if we're in the project directory
    if not os.path.exists("tomato_dataset"):
        print("❌ Dataset not found. Please run from the project directory.")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️ Virtual environment not detected. Continuing anyway...")
    
    print("✅ Environment check passed")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    
    dependencies = [
        "opencv-python",
        "pillow", 
        "numpy",
        "flask"
    ]
    
    for dep in dependencies:
        try:
            if dep == "opencv-python":
                import cv2
            elif dep == "pillow":
                from PIL import Image, ImageTk
            elif dep == "numpy":
                import numpy
            elif dep == "flask":
                import flask
            print(f"  ✅ {dep} already installed")
        except ImportError:
            print(f"  📦 Installing {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                              check=True, capture_output=True)
                print(f"  ✅ {dep} installed")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {dep}: {e}")
                return False
    
    return True

def verify_dataset():
    """Verify dataset structure"""
    print("🔍 Verifying dataset...")
    
    dataset_path = "tomato_dataset"
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        return False
    
    # Check structure
    required_dirs = ["train", "val"]
    required_classes = ["Unripe", "Ripe", "Old", "Damaged"]
    
    for split in required_dirs:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"❌ {split} directory not found!")
            return False
        
        for class_name in required_classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"❌ {split}/{class_name} directory not found!")
                return False
            
            # Count images
            image_count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            print(f"  {split}/{class_name}: {image_count} images")
    
    print("✅ Dataset structure verified")
    return True

def create_data_yaml():
    """Create data.yaml if it doesn't exist"""
    data_yaml_path = "tomato_dataset/data.yaml"
    if os.path.exists(data_yaml_path):
        print("✅ data.yaml already exists")
        return True
    
    print("📄 Creating data.yaml...")
    try:
        import yaml
        
        data_config = {
            'path': os.path.abspath('tomato_dataset'),
            'train': 'train',
            'val': 'val',
            'nc': 3,
            'names': {
                0: 'not_ready',
                1: 'ready', 
                2: 'spoilt'
            }
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print("✅ data.yaml created")
        return True
    except Exception as e:
        print(f"❌ Failed to create data.yaml: {e}")
        return False

def launch_gui():
    """Launch the GUI application"""
    print("🚀 Launching GUI...")
    
    try:
        from tomato_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        return False
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🍅 AI Tomato Sorter - Complete GUI Launcher")
    print("=" * 60)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed!")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        return False
    
    # Step 3: Verify dataset
    if not verify_dataset():
        print("\n❌ Dataset verification failed!")
        return False
    
    # Step 4: Create data.yaml
    if not create_data_yaml():
        print("\n❌ Failed to create data.yaml!")
        return False
    
    print("\n✅ All checks passed! Starting GUI...")
    print("=" * 60)
    
    # Step 5: Launch GUI
    try:
        launch_gui()
    except KeyboardInterrupt:
        print("\n👋 GUI closed by user")
    except Exception as e:
        print(f"\n❌ GUI error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Launcher failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n🎉 GUI session completed successfully!")
