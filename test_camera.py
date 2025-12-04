#!/usr/bin/env python3
"""
Camera Diagnostic Tool
Tests camera availability and functionality
"""

import sys
import os

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("ERROR: OpenCV (cv2) not installed")
    print("Install with: pip install opencv-python")
    sys.exit(1)

def test_camera(index):
    """Test a specific camera index"""
    print(f"\n{'='*50}")
    print(f"Testing Camera Index: {index}")
    print(f"{'='*50}")
    
    try:
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print(f"❌ Camera {index}: NOT OPENED")
            return False
        
        print(f"✅ Camera {index}: Opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Camera {index}: Cannot read frames")
            cap.release()
            return False
        
        if frame is None:
            print(f"❌ Camera {index}: Frame is None")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✅ Camera {index}: Frame read successfully")
        print(f"   Resolution: {width}x{height}")
        
        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"   FPS: {fps}")
        print(f"   Codec: {fourcc_str}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera {index}: Error - {e}")
        return False

def list_video_devices():
    """List available video devices on Linux"""
    video_devices = []
    if os.path.exists('/dev'):
        for item in os.listdir('/dev'):
            if item.startswith('video'):
                video_devices.append(f"/dev/{item}")
    return sorted(video_devices)

def main():
    print("="*50)
    print("CAMERA DIAGNOSTIC TOOL")
    print("="*50)
    
    # Check OpenCV version
    print(f"\nOpenCV Version: {cv2.__version__}")
    
    # List video devices
    print("\n📹 Available Video Devices:")
    video_devices = list_video_devices()
    if video_devices:
        for device in video_devices:
            print(f"   {device}")
    else:
        print("   No /dev/video* devices found")
    
    # Test common camera indices
    print("\n🔍 Testing Camera Indices:")
    working_cameras = []
    
    for i in range(5):  # Test indices 0-4
        if test_camera(i):
            working_cameras.append(i)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if working_cameras:
        print(f"✅ Working cameras found: {working_cameras}")
        print(f"   Recommended: Use camera index {working_cameras[0]}")
    else:
        print("❌ No working cameras found!")
        print("\nTroubleshooting:")
        print("1. Check if camera is connected (USB or built-in)")
        print("2. Check permissions: ls -l /dev/video*")
        print("3. Try: sudo chmod 666 /dev/video0")
        print("4. Check if another application is using the camera")
        print("5. On Linux, try: v4l2-ctl --list-devices")
        print("6. On Raspberry Pi, enable camera: sudo raspi-config")
    
    return 0 if working_cameras else 1

if __name__ == "__main__":
    sys.exit(main())

