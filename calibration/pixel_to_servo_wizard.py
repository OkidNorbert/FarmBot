#!/usr/bin/env python3
"""
Pixel-to-Servo Calibration Wizard
=================================

Interactive wizard to generate pixel-to-servo angle mapping table.
Uses distance-based lookup approach as specified in requirements.
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

class PixelToServoWizard:
    def __init__(self):
        self.calibration_points = []
        self.camera = None
        self.lookup_table = {}
        # claw endpoint angles (determined by calibration or manual entry)
        self.claw_open_angle = None    # servo angle corresponding to fully open claw
        self.claw_closed_angle = None  # servo angle corresponding to fully closed claw
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            print(f"Failed to open camera {camera_index}")
            return False
        return True
    
    def add_calibration_point(self, pixel_x, pixel_y, tof_distance_mm, base_angle, shoulder_angle, forearm_angle):
        """Add a calibration point"""
        point = {
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'tof_distance_mm': tof_distance_mm,
            'base_angle': base_angle,
            'shoulder_angle': shoulder_angle,
            'forearm_angle': forearm_angle,
            'timestamp': datetime.now().isoformat()
        }
        self.calibration_points.append(point)
        print(f"✅ Calibration point added: Pixel({pixel_x}, {pixel_y}), ToF={tof_distance_mm}mm, Base={base_angle}°")
    
    def generate_lookup_table(self):
        """Generate distance-based lookup table"""
        if len(self.calibration_points) < 4:
            print("Error: Need at least 4 calibration points")
            return False
        
        # Group by distance ranges
        distance_ranges = {}
        for point in self.calibration_points:
            dist = point['tof_distance_mm']
            # Round to nearest 10mm for lookup
            range_key = (dist // 10) * 10
            if range_key not in distance_ranges:
                distance_ranges[range_key] = []
            distance_ranges[range_key].append(point)
        
        # Create lookup table
        self.lookup_table = {}
        for dist_range, points in distance_ranges.items():
            # Average angles for this distance range
            avg_base = np.mean([p['base_angle'] for p in points])
            avg_shoulder = np.mean([p['shoulder_angle'] for p in points])
            avg_forearm = np.mean([p['forearm_angle'] for p in points])
            
            self.lookup_table[dist_range] = {
                'base_angle': int(avg_base),
                'shoulder_angle': int(avg_shoulder),
                'forearm_angle': int(avg_forearm),
                'point_count': len(points)
            }
        
        print(f"✅ Lookup table generated with {len(self.lookup_table)} distance ranges")
        return True
    
    def lookup_servo_angles(self, pixel_x, pixel_y, tof_distance_mm):
        """Lookup servo angles for given pixel and distance"""
        # Find nearest distance range
        dist_range = (tof_distance_mm // 10) * 10
        
        # Find closest range in lookup table
        closest_range = min(self.lookup_table.keys(), key=lambda x: abs(x - dist_range))
        
        if closest_range in self.lookup_table:
            return self.lookup_table[closest_range]
        else:
            return None

    def calibrate_claw_endpoints(self):
        """Interactive assistance for claw open/closed angles.
        The user may manually move the claw or use the camera stream to
        choose endpoints.  If camera is available it will display live video
        and respond to `o` (open) and `c` (closed) key presses; otherwise the
        operator simply types the angles.
        """
        print("\n=== Claw End‑Point Calibration ===")
        if not self.initialize_camera():
            print("Camera not available – please enter angles manually.")
            self.claw_open_angle = int(input("Enter servo angle when claw is fully open: "))
            self.claw_closed_angle = int(input("Enter servo angle when claw is fully closed: "))
            return

        print("Press 'o' when claw is open, 'c' when claw is closed, ESC to finish.")
        open_val = None
        closed_val = None
        while open_val is None or closed_val is None:
            ret, frame = self.camera.read()
            if not ret:
                continue
            cv2.imshow('Claw Calibration - press o/c', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('o'):
                open_val = int(input("Servo angle for open state: "))
                print(f"Recorded open angle {open_val}")
            elif key == ord('c'):
                closed_val = int(input("Servo angle for closed state: "))
                print(f"Recorded closed angle {closed_val}")
            elif key == 27:  # ESC
                break
        cv2.destroyAllWindows()
        self.claw_open_angle = open_val
        self.claw_closed_angle = closed_val
        print(f"Claw endpoints: open={self.claw_open_angle}, closed={self.claw_closed_angle}")
    
    def save_calibration(self, filename="pixel_to_servo_calibration.json"):
        """Save calibration data"""
        data = {
            'calibration_points': self.calibration_points,
            'lookup_table': self.lookup_table,
            'created': datetime.now().isoformat(),
            'point_count': len(self.calibration_points)
        }
        # include claw endpoints if available
        if self.claw_open_angle is not None:
            data['claw_open_angle'] = self.claw_open_angle
        if self.claw_closed_angle is not None:
            data['claw_closed_angle'] = self.claw_closed_angle
        
        os.makedirs('calibration', exist_ok=True)
        filepath = os.path.join('calibration', filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Calibration saved to {filepath}")
        return filepath
    
    def load_calibration(self, filename="pixel_to_servo_calibration.json"):
        """Load calibration data"""
        filepath = os.path.join('calibration', filename)
        if not os.path.exists(filepath):
            print(f"Calibration file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.calibration_points = data.get('calibration_points', [])
        self.lookup_table = data.get('lookup_table', {})
        # load claw endpoints if present
        self.claw_open_angle = data.get('claw_open_angle')
        self.claw_closed_angle = data.get('claw_closed_angle')
        if self.claw_open_angle is not None and self.claw_closed_angle is not None:
            print(f"   Claw open angle: {self.claw_open_angle}°")
            print(f"   Claw closed angle: {self.claw_closed_angle}°")
        
        print(f"✅ Calibration loaded from {filepath}")
        print(f"   Points: {len(self.calibration_points)}")
        print(f"   Lookup ranges: {len(self.lookup_table)}")
        return True
    
    def interactive_calibration(self):
        """Interactive calibration wizard"""
        print("=" * 60)
        print("Pixel-to-Servo Calibration Wizard")
        print("=" * 60)
        print()
        print("This wizard will help you create a mapping between:")
        print("  - Camera pixel coordinates (X, Y)")
        print("  - ToF sensor distance (mm)")
        print("  - Servo angles (Base, Shoulder, Forearm)")
        print()
        print("You need at least 4 calibration points for a good mapping.")
        print()
        
        if not self.initialize_camera():
            print("Camera initialization failed. Continuing with manual entry...")
            use_camera = False
        else:
            use_camera = True
            print("Camera ready. Press 'c' to capture current frame, or 'm' for manual entry.")
        
        point_num = 1
        while True:
            print(f"\n--- Calibration Point {point_num} ---")
            
            if use_camera:
                ret, frame = self.camera.read()
                if ret:
                    cv2.imshow('Calibration - Click on target point', frame)
                    print("Click on the target point in the camera window, then press any key...")
                    cv2.waitKey(0)
                    # Get click coordinates (simplified - in real implementation use mouse callback)
                    pixel_x = int(input("Enter pixel X coordinate: "))
                    pixel_y = int(input("Enter pixel Y coordinate: "))
                else:
                    pixel_x = int(input("Enter pixel X coordinate: "))
                    pixel_y = int(input("Enter pixel Y coordinate: "))
            else:
                pixel_x = int(input("Enter pixel X coordinate: "))
                pixel_y = int(input("Enter pixel Y coordinate: "))
            
            tof_distance = float(input("Enter ToF distance (mm): "))
            base_angle = int(input("Enter Base servo angle (0-180): "))
            shoulder_angle = int(input("Enter Shoulder servo angle (0-180): "))
            forearm_angle = int(input("Enter Forearm servo angle (0-180): "))
            
            self.add_calibration_point(pixel_x, pixel_y, tof_distance, 
                                      base_angle, shoulder_angle, forearm_angle)
            
            point_num += 1
            
            if point_num > 4:
                more = input("\nAdd another point? (y/n): ").lower()
                if more != 'y':
                    break
            
            if use_camera:
                cv2.destroyAllWindows()
        
        if use_camera:
            cv2.destroyAllWindows()
        
        # Generate lookup table
        if self.generate_lookup_table():
            # optionally calibrate claw endpoints
            yn = input("\nWould you like to calibrate claw open/closed angles now? (y/n): ").strip().lower()
            if yn == 'y':
                self.calibrate_claw_endpoints()
            # Save calibration
            filename = input("\nEnter filename to save (default: pixel_to_servo_calibration.json): ").strip()
            if not filename:
                filename = "pixel_to_servo_calibration.json"
            self.save_calibration(filename)
            
            print("\n✅ Calibration complete!")
            return True
        else:
            print("\n❌ Calibration failed")
            return False

def main():
    wizard = PixelToServoWizard()
    
    print("Pixel-to-Servo Calibration Wizard")
    print("1. New calibration")
    print("2. Load existing calibration")
    print("3. Test calibration")
    print("4. Calibrate claw end‑points")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        wizard.interactive_calibration()
    elif choice == '2':
        filename = input("Enter calibration filename: ").strip()
        if wizard.load_calibration(filename):
            print("Calibration loaded successfully")
    elif choice == '3':
        filename = input("Enter calibration filename: ").strip()
        if wizard.load_calibration(filename):
            print("\nTesting lookup...")
            pixel_x = int(input("Enter pixel X: "))
            pixel_y = int(input("Enter pixel Y: "))
            tof_dist = float(input("Enter ToF distance (mm): "))
            
            angles = wizard.lookup_servo_angles(pixel_x, pixel_y, tof_dist)
            if angles:
                print(f"\nLookup result:")
                print(f"  Base: {angles['base_angle']}°")
                print(f"  Shoulder: {angles['shoulder_angle']}°")
                print(f"  Forearm: {angles['forearm_angle']}°")
                if wizard.claw_open_angle is not None:
                    print(f"  Claw open endpoint: {wizard.claw_open_angle}°")
                if wizard.claw_closed_angle is not None:
                    print(f"  Claw closed endpoint: {wizard.claw_closed_angle}°")
            else:
                print("No matching lookup entry found")
    elif choice == '4':
        wizard.calibrate_claw_endpoints()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()

