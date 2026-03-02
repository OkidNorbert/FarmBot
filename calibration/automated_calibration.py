#!/usr/bin/env python3
"""
Automated Calibration Script
=============================

This script automates the calibration process by:
1. Connecting to the Arduino hardware
2. Moving the arm to predefined positions
3. Capturing camera frames at each position
4. Recording pixel and servo angle mapping

Use this instead of the manual wizard for faster, more accurate calibration.
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hardware_controller import HardwareController

class AutomatedCalibrator:
    def __init__(self, connection_type='auto'):
        """Initialize with hardware controller connection"""
        self.hc = HardwareController(connection_type=connection_type)
        self.calibration_points = []
        self.camera = None
        
        # Predefined calibration positions (servo angles)
        # Format: (base, shoulder, forearm, elbow, pitch, claw, description)
        self.calibration_positions = [
            (90, 90, 90, 90, 90, 100, "Center"),
            (45, 90, 90, 90, 90, 100, "Left"),
            (135, 90, 90, 90, 90, 100, "Right"),
            (90, 60, 90, 90, 90, 100, "Up"),
            (90, 120, 90, 90, 90, 100, "Down"),
            (90, 90, 60, 90, 90, 100, "Forward"),
            (90, 90, 120, 90, 90, 100, "Backward"),
        ]
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return False
        
        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    
    def wait_for_arm_stability(self, wait_time=2.0):
        """Wait for arm to reach target position"""
        print(f"   Waiting {wait_time}s for arm to stabilize...", end="", flush=True)
        time.sleep(wait_time)
        print(" ✓")
    
    def capture_at_position(self, base, shoulder, forearm, elbow, pitch, claw, description):
        """Move arm to position and capture frame"""
        print(f"\n📍 Calibration point: {description}")
        print(f"   Target angles: Base={base}, Shoulder={shoulder}, Forearm={forearm}, "
              f"Elbow={elbow}, Pitch={pitch}, Claw={claw}")
        
        # Send movement command
        command = f"ANGLE {base} {shoulder} {forearm} {elbow} {pitch} {claw}"
        print(f"   Sending: {command}")
        self.hc.send_command(command)
        
        # Wait for arm to move
        self.wait_for_arm_stability(wait_time=3.0)
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            print("   ❌ Failed to capture frame")
            return None
        
        # Display frame and ask user to click target point
        cv2.imshow('Click on target point (press any key when done)', frame)
        cv2.waitKey(1000)  # Auto-advance after 1 second, or wait for manual key
        
        # Ask for pixel coordinates
        print("   Click on the arm end-effector or target point in the window.")
        print("   (Image is displayed for 3 seconds)")
        
        # Simple approach: ask user to move mouse and click
        click_coords = None
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_coords
            if event == cv2.EVENT_LBUTTONDOWN:
                click_coords = (x, y)
                print(f"   ✓ Clicked at pixel: ({x}, {y})")
        
        cv2.setMouseCallback('Click on target point (press any key when done)', mouse_callback)
        
        # Display and wait
        start_time = time.time()
        while time.time() - start_time < 5:
            cv2.imshow('Click on target point (press any key when done)', frame)
            key = cv2.waitKey(100)
            if key != -1 or click_coords is not None:
                break
        
        cv2.setMouseCallback('Click on target point (press any key when done)', lambda *args: None)
        cv2.destroyAllWindows()
        
        if click_coords is None:
            # Ask for manual input
            pixel_x = int(input(f"   Enter pixel X coordinate (0-640): "))
            pixel_y = int(input(f"   Enter pixel Y coordinate (0-480): "))
            click_coords = (pixel_x, pixel_y)
        
        # Get ToF sensor reading if available
        tof_distance = self.hc.get_distance_sensor()
        if tof_distance is None or tof_distance <= 0:
            tof_distance = 100  # Default fallback
        
        point = {
            'pixel_x': click_coords[0],
            'pixel_y': click_coords[1],
            'tof_distance_mm': tof_distance,
            'base_angle': base,
            'shoulder_angle': shoulder,
            'forearm_angle': forearm,
            'elbow_angle': elbow,
            'pitch_angle': pitch,
            'claw_angle': claw,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✓ Recorded: Pixel({click_coords[0]}, {click_coords[1]}), ToF={tof_distance}mm")
        return point
    
    def run_automated_calibration(self):
        """Run full automated calibration"""
        print("=" * 70)
        print("AUTOMATED ARM CALIBRATION")
        print("=" * 70)
        print()
        
        # Check hardware connection
        if not self.hc.arduino_connected:
            print("⚠️  WARNING: Arduino not detected yet. Attempting to connect...")
            time.sleep(2)
        
        if not self.hc.arduino_connected and not self.hc.ble_client:
            print("❌ ERROR: No hardware connection available!")
            print("   Please connect Arduino via Serial or Bluetooth first.")
            return False
        
        # Initialize camera
        if not self.initialize_camera():
            print("⚠️  Camera not available - calibration cannot proceed (need visual feedback)")
            return False
        
        print(f"✅ Hardware connected, camera ready")
        print(f"📹 Will capture {len(self.calibration_positions)} positions\n")
        
        input("Press ENTER to start calibration (arm will move)...")
        
        # Move through each position
        for idx, (base, shoulder, forearm, elbow, pitch, claw, desc) in enumerate(self.calibration_positions, 1):
            print(f"\n[{idx}/{len(self.calibration_positions)}]", end="")
            point = self.capture_at_position(base, shoulder, forearm, elbow, pitch, claw, desc)
            
            if point:
                self.calibration_points.append(point)
                print(f"   Total points collected: {len(self.calibration_points)}")
            
            # Ask to continue
            if idx < len(self.calibration_positions):
                cont = input("\nContinue to next point? (y/n): ").strip().lower()
                if cont != 'y':
                    print("Stopping calibration early.")
                    break
        
        # Return to safe home position
        print("\n🏠 Returning arm to home position...")
        self.hc.send_command("ANGLE 90 90 90 90 90 100")
        self.wait_for_arm_stability(2.0)
        
        cv2.destroyAllWindows()
        
        if len(self.calibration_points) < 4:
            print("❌ Not enough calibration points. Need at least 4.")
            return False
        
        print(f"\n✅ Calibration complete with {len(self.calibration_points)} points!")
        return self.save_calibration()
    
    def save_calibration(self, filename="automated_calibration.json"):
        """Save calibration data"""
        data = {
            'calibration_points': self.calibration_points,
            'created': datetime.now().isoformat(),
            'point_count': len(self.calibration_points),
            'method': 'automated'
        }
        
        os.makedirs('calibration', exist_ok=True)
        filepath = os.path.join('calibration', filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Calibration saved to {filepath}")
        print(f"   You can now use this for pixel-to-servo mapping")
        return True

def main():
    print("\nAutomated Calibration Tool")
    print("-" * 70)
    print("This tool will move your arm automatically through calibration positions")
    print("and record the pixel coordinates for mapping.\n")
    
    # Choose connection type
    print("Connection type:")
    print("1. Auto (try serial first, then Bluetooth)")
    print("2. Serial (USB)")
    print("3. Bluetooth (BLE)")
    
    choice = input("\nSelect connection (1-3, default 1): ").strip() or "1"
    
    conn_map = {
        '1': 'auto',
        '2': 'serial',
        '3': 'bluetooth'
    }
    conn_type = conn_map.get(choice, 'auto')
    
    # Create calibrator and run
    calibrator = AutomatedCalibrator(connection_type=conn_type)
    success = calibrator.run_automated_calibration()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Calibration successful!")
        print("=" * 70)
        
        # Optional: Display calibration points
        yn = input("\nDisplay calibration summary? (y/n): ").strip().lower()
        if yn == 'y':
            print("\nCalibration Points:")
            for i, point in enumerate(calibrator.calibration_points, 1):
                print(f"  {i}. {point['description']}: "
                      f"Pixel({point['pixel_x']}, {point['pixel_y']}), "
                      f"ToF={point['tof_distance_mm']}mm, "
                      f"Base={point['base_angle']}°")
    else:
        print("\n❌ Calibration failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
