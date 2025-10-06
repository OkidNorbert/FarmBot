#!/usr/bin/env python3
"""
Launch the Tomato Classification GUI
Simple launcher script with dependency checking
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['tkinter', 'PIL', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'PIL':
                from PIL import Image, ImageTk
            elif package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nTo install missing packages:")
        print("  pip install opencv-python pillow numpy")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🍅 AI Tomato Sorter - GUI Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    print("✅ All dependencies found!")
    print("🚀 Starting GUI application...")
    
    # Launch the GUI
    try:
        from tomato_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Error importing GUI: {e}")
        print("Make sure tomato_gui.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")

if __name__ == "__main__":
    main()
