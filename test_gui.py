#!/usr/bin/env python3
"""
Test script for the Tomato Classification GUI
Verifies all components are working correctly
"""

import os
import sys
import subprocess
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    required_modules = ['tkinter', 'PIL', 'cv2', 'numpy']
    missing = []
    
    for module in required_modules:
        try:
            if module == 'tkinter':
                import tkinter
            elif module == 'PIL':
                from PIL import Image, ImageTk
            elif module == 'cv2':
                import cv2
            elif module == 'numpy':
                import numpy
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install opencv-python pillow numpy")
        return False
    
    print("✅ All dependencies found!")
    return True

def test_dataset():
    """Test if dataset exists and is properly structured"""
    print("\n🔍 Testing dataset...")
    
    dataset_path = "tomato_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Check train/val structure
    train_path = Path(dataset_path) / "train"
    val_path = Path(dataset_path) / "val"
    
    if not train_path.exists():
        print("❌ Train directory not found")
        return False
    
    if not val_path.exists():
        print("❌ Val directory not found")
        return False
    
    # Check class folders
    expected_classes = ['Unripe', 'Ripe', 'Old', 'Damaged']
    for split in [train_path, val_path]:
        for class_name in expected_classes:
            class_path = split / class_name
            if not class_path.exists():
                print(f"❌ Class folder not found: {class_path}")
                return False
            
            # Count images
            image_count = len(list(class_path.glob('*.jpg')))
            print(f"  {split.name}/{class_name}: {image_count} images")
    
    print("✅ Dataset structure is correct!")
    return True

def test_scripts():
    """Test if all required scripts exist"""
    print("\n🔍 Testing scripts...")
    
    required_scripts = [
        'tomato_gui.py',
        'launch_gui.py',
        'train_tomato_classifier.py',
        'inference_classifier.py',
        'quick_train.py'
    ]
    
    missing = []
    for script in required_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
            missing.append(script)
    
    if missing:
        print(f"\n❌ Missing scripts: {', '.join(missing)}")
        return False
    
    print("✅ All scripts found!")
    return True

def test_gui_launch():
    """Test if GUI can be launched"""
    print("\n🔍 Testing GUI launch...")
    
    try:
        # Try to import the GUI
        import tomato_gui
        print("✅ GUI module imports successfully")
        
        # Test if GUI can be created (without showing)
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Try to create GUI instance
        gui = tomato_gui.TomatoGUI(root)
        print("✅ GUI instance created successfully")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🍅 Tomato Classification GUI - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Dataset", test_dataset),
        ("Scripts", test_scripts),
        ("GUI Launch", test_gui_launch)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GUI is ready to use.")
        print("\n🚀 Launch GUI with: python launch_gui.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
