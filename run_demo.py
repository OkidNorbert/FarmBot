#!/usr/bin/env python3
"""
AI Tomato Sorter - Demo Script
Complete demo of the AI Tomato Sorter system
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime

def print_banner():
    """Print project banner"""
    banner = """
    🍅 AI Tomato Sorter - Complete System Demo
    ==========================================
    
    A complete AI-powered tomato sorting system using:
    • Computer Vision (YOLOv8)
    • Edge Computing (Raspberry Pi 5)
    • Robotics (Arduino + 3-DOF Arm)
    • Web Interface (Real-time Monitoring)
    
    """
    print(banner)

def check_system_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        'Python': sys.version_info >= (3, 8),
        'OpenCV': False,
        'Ultralytics': False,
        'Serial': False,
        'Camera': False
    }
    
    # Check Python packages
    try:
        import cv2
        requirements['OpenCV'] = True
        print(f"   ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("   ❌ OpenCV not installed")
    
    try:
        import ultralytics
        requirements['Ultralytics'] = True
        print("   ✅ Ultralytics installed")
    except ImportError:
        print("   ❌ Ultralytics not installed")
    
    try:
        import serial
        requirements['Serial'] = True
        print("   ✅ PySerial installed")
    except ImportError:
        print("   ❌ PySerial not installed")
    
    # Check camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        requirements['Camera'] = True
        print("   ✅ Camera detected")
        cap.release()
    else:
        print("   ❌ Camera not detected")
    
    # Check if all requirements met
    all_met = all(requirements.values())
    if all_met:
        print("   ✅ All requirements met!")
    else:
        print("   ⚠️  Some requirements not met - install missing packages")
    
    return all_met

def demo_data_preparation():
    """Demo data preparation process"""
    print("\n📊 Data Preparation Demo")
    print("-" * 30)
    
    # Check if dataset exists
    dataset_path = Path("tomato_dataset")
    if dataset_path.exists():
        print("   ✅ Dataset directory exists")
        
        # Count images
        train_images = len(list((dataset_path / "images" / "train").glob("*.jpg")))
        val_images = len(list((dataset_path / "images" / "val").glob("*.jpg")))
        test_images = len(list((dataset_path / "images" / "test").glob("*.jpg")))
        
        print(f"   📈 Dataset statistics:")
        print(f"      Training images: {train_images}")
        print(f"      Validation images: {val_images}")
        print(f"      Test images: {test_images}")
    else:
        print("   ⚠️  Dataset not found - run data preparation first")
        print("   💡 Use: python train/data_preparation.py --help")

def demo_model_training():
    """Demo model training process"""
    print("\n🤖 Model Training Demo")
    print("-" * 30)
    
    # Check if model exists
    model_path = Path("runs/detect/tomato_sorter/weights/best.pt")
    if model_path.exists():
        print("   ✅ Trained model found")
        print(f"      Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Check if exported models exist
        onnx_path = Path("exported_models/tomato_sorter.onnx")
        tflite_path = Path("exported_models/tomato_sorter.tflite")
        
        if onnx_path.exists():
            print("   ✅ ONNX model exported")
        if tflite_path.exists():
            print("   ✅ TFLite model exported")
    else:
        print("   ⚠️  Trained model not found")
        print("   💡 Use: python train/train_tomato_detector.py --help")

def demo_inference():
    """Demo inference system"""
    print("\n🎯 Inference Demo")
    print("-" * 30)
    
    # Check if inference script exists
    inference_script = Path("pi/inference_pi.py")
    if inference_script.exists():
        print("   ✅ Inference script available")
        
        # Check if model exists
        model_path = Path("exported_models/tomato_sorter.onnx")
        if model_path.exists():
            print("   ✅ ONNX model ready for inference")
        else:
            print("   ⚠️  ONNX model not found - export model first")
    else:
        print("   ❌ Inference script not found")

def demo_arduino_integration():
    """Demo Arduino integration"""
    print("\n🤖 Arduino Integration Demo")
    print("-" * 30)
    
    # Check if Arduino firmware exists
    arduino_script = Path("arduino/tomato_sorter_arduino.ino")
    if arduino_script.exists():
        print("   ✅ Arduino firmware available")
        print("   💡 Upload to Arduino: Load tomato_sorter_arduino.ino in Arduino IDE")
    else:
        print("   ❌ Arduino firmware not found")
    
    # Check serial communication
    try:
        import serial
        ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
        available_ports = []
        
        for port in ports:
            try:
                ser = serial.Serial(port, 115200, timeout=1)
                ser.close()
                available_ports.append(port)
            except:
                pass
        
        if available_ports:
            print(f"   ✅ Serial ports available: {available_ports}")
        else:
            print("   ⚠️  No Arduino detected on serial ports")
    except ImportError:
        print("   ❌ PySerial not installed")

def demo_web_interface():
    """Demo web interface"""
    print("\n🌐 Web Interface Demo")
    print("-" * 30)
    
    # Check if web interface exists
    web_script = Path("pi/web_interface.py")
    if web_script.exists():
        print("   ✅ Web interface available")
        print("   💡 Start with: python pi/web_interface.py --host 0.0.0.0 --port 5000")
        print("   🌐 Access at: http://<pi-ip>:5000")
    else:
        print("   ❌ Web interface not found")

def demo_calibration():
    """Demo camera calibration"""
    print("\n🎯 Camera Calibration Demo")
    print("-" * 30)
    
    # Check if calibration script exists
    calib_script = Path("pi/calibration.py")
    if calib_script.exists():
        print("   ✅ Calibration script available")
        print("   💡 Run: python pi/calibration.py --camera 0 --output calibration.json")
        
        # Check if calibration file exists
        calib_file = Path("calibration.json")
        if calib_file.exists():
            print("   ✅ Calibration file found")
        else:
            print("   ⚠️  Calibration file not found - run calibration first")
    else:
        print("   ❌ Calibration script not found")

def demo_evaluation():
    """Demo evaluation system"""
    print("\n📊 Evaluation Demo")
    print("-" * 30)
    
    # Check if evaluation script exists
    eval_script = Path("test/evaluation.py")
    if eval_script.exists():
        print("   ✅ Evaluation script available")
        print("   💡 Run: python test/evaluation.py --model exported_models/tomato_sorter.onnx --test_data tomato_dataset")
    else:
        print("   ❌ Evaluation script not found")

def run_quick_test():
    """Run a quick system test"""
    print("\n🧪 Quick System Test")
    print("-" * 30)
    
    try:
        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("   ✅ Camera test passed")
            else:
                print("   ❌ Camera test failed - no frame captured")
            cap.release()
        else:
            print("   ❌ Camera test failed - cannot open camera")
        
        # Test model loading
        model_path = Path("exported_models/tomato_sorter.onnx")
        if model_path.exists():
            try:
                net = cv2.dnn.readNetFromONNX(str(model_path))
                print("   ✅ Model loading test passed")
            except Exception as e:
                print(f"   ❌ Model loading test failed: {e}")
        else:
            print("   ⚠️  Model not found - skip model test")
        
        # Test serial communication
        try:
            import serial
            ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
            for port in ports:
                try:
                    ser = serial.Serial(port, 115200, timeout=1)
                    ser.close()
                    print(f"   ✅ Serial communication test passed on {port}")
                    break
                except:
                    pass
            else:
                print("   ⚠️  No Arduino detected - serial test skipped")
        except ImportError:
            print("   ❌ PySerial not installed - serial test skipped")
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🚀 Next Steps")
    print("-" * 30)
    print("1. 📊 Prepare your dataset:")
    print("   python train/data_preparation.py --help")
    print()
    print("2. 🤖 Train the model:")
    print("   python train/train_tomato_detector.py --help")
    print()
    print("3. 📤 Export model for Pi:")
    print("   python export/export_models.py --help")
    print()
    print("4. 🎯 Calibrate camera:")
    print("   python pi/calibration.py --help")
    print()
    print("5. 🚀 Run the system:")
    print("   python pi/inference_pi.py --help")
    print()
    print("6. 🌐 Start web interface:")
    print("   python pi/web_interface.py --help")
    print()
    print("7. 📊 Evaluate performance:")
    print("   python test/evaluation.py --help")

def main():
    parser = argparse.ArgumentParser(description='AI Tomato Sorter Demo')
    parser.add_argument('--quick-test', action='store_true', help='Run quick system test')
    parser.add_argument('--check-requirements', action='store_true', help='Check system requirements')
    parser.add_argument('--show-steps', action='store_true', help='Show next steps')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.check_requirements:
        check_system_requirements()
    
    if args.quick_test:
        run_quick_test()
    
    if args.show_steps:
        show_next_steps()
    
    if not any([args.check_requirements, args.quick_test, args.show_steps]):
        # Run full demo
        print("🔍 System Status Check")
        print("=" * 50)
        
        # Check requirements
        requirements_met = check_system_requirements()
        
        # Demo each component
        demo_data_preparation()
        demo_model_training()
        demo_inference()
        demo_arduino_integration()
        demo_web_interface()
        demo_calibration()
        demo_evaluation()
        
        # Run quick test
        run_quick_test()
        
        # Show next steps
        show_next_steps()
        
        print("\n🎉 Demo Complete!")
        print("=" * 50)
        print("Your AI Tomato Sorter system is ready to use!")
        print("Follow the next steps above to get started.")

if __name__ == "__main__":
    main()
