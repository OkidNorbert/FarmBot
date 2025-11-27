#!/usr/bin/env python3
"""
Coordinate Mapping System for AI Tomato Sorter
Maps camera pixel coordinates to robotic arm world coordinates
"""

import numpy as np
import cv2
import yaml
import json
from pathlib import Path

class CoordinateMapper:
    def __init__(self, config_file="pi_config.yaml"):
        """Initialize coordinate mapper"""
        self.config = self.load_config(config_file)
        self.calibration_points = []
        self.transformation_matrix = None
        self.calibrated = False
        
    def load_config(self, config_file):
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'camera': {'width': 640, 'height': 480},
            'arm': {
                'workspace_x': [-150, 150],
                'workspace_y': [50, 200],
                'arm_length_1': 100.0,
                'arm_length_2': 80.0
            }
        }
    
    def add_calibration_point(self, pixel_x, pixel_y, arm_x, arm_y):
        """Add a calibration point"""
        self.calibration_points.append({
            'pixel': (pixel_x, pixel_y),
            'arm': (arm_x, arm_y)
        })
        print(f"Added calibration point: Pixel({pixel_x}, {pixel_y}) -> Arm({arm_x}, {arm_y})")
    
    def calculate_transformation_matrix(self):
        """Calculate transformation matrix from calibration points"""
        if len(self.calibration_points) < 4:
            print("Need at least 4 calibration points")
            return False
        
        # Prepare data for transformation
        pixel_points = np.array([p['pixel'] for p in self.calibration_points], dtype=np.float32)
        arm_points = np.array([p['arm'] for p in self.calibration_points], dtype=np.float32)
        
        # Calculate homography matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            pixel_points, arm_points
        )
        
        self.calibrated = True
        print("Transformation matrix calculated successfully")
        return True
    
    def pixel_to_arm(self, pixel_x, pixel_y):
        """Convert pixel coordinates to arm coordinates"""
        if not self.calibrated:
            print("Not calibrated yet")
            return None
        
        # Convert to homogeneous coordinates
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        
        # Transform to arm coordinates
        arm_point = cv2.perspectiveTransform(pixel_point, self.transformation_matrix)
        
        return arm_point[0][0]
    
    def arm_to_pixel(self, arm_x, arm_y):
        """Convert arm coordinates to pixel coordinates"""
        if not self.calibrated:
            print("Not calibrated yet")
            return None
        
        # Convert to homogeneous coordinates
        arm_point = np.array([[[arm_x, arm_y]]], dtype=np.float32)
        
        # Transform to pixel coordinates
        pixel_point = cv2.perspectiveTransform(arm_point, np.linalg.inv(self.transformation_matrix))
        
        return pixel_point[0][0]
    
    def save_calibration(self, filename="calibration.json"):
        """Save calibration data"""
        if not self.calibrated:
            print("No calibration to save")
            return False
        
        calibration_data = {
            'calibration_points': self.calibration_points,
            'transformation_matrix': self.transformation_matrix.tolist(),
            'calibrated': self.calibrated
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
        return True
    
    def load_calibration(self, filename="calibration.json"):
        """Load calibration data"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.calibration_points = calibration_data['calibration_points']
            self.transformation_matrix = np.array(calibration_data['transformation_matrix'])
            self.calibrated = calibration_data['calibrated']
            
            print(f"Calibration loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found")
            return False
    
    def interactive_calibration(self):
        """Interactive calibration using camera feed"""
        print("Interactive Calibration Mode")
        print("============================")
        print("1. Position the robotic arm at known positions")
        print("2. Click on the corresponding point in the camera feed")
        print("3. Enter the arm coordinates when prompted")
        print("4. Repeat for at least 4 points")
        print("5. Press 'q' to quit calibration")
        print()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera")
            return False
        
        cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
        
        point_count = 0
        current_frame = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal point_count, current_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at pixel ({x}, {y})")
                
                # Get arm coordinates from user
                try:
                    arm_x = float(input("Enter arm X coordinate (mm): "))
                    arm_y = float(input("Enter arm Y coordinate (mm): "))
                    
                    self.add_calibration_point(x, y, arm_x, arm_y)
                    point_count += 1
                    
                    # Draw point on frame
                    cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(current_frame, f"Point {point_count}", (x+10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('Calibration', current_frame)
                    
                except ValueError:
                    print("Invalid input, skipping point")
        
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = frame.copy()
            
            # Draw instructions
            cv2.putText(frame, f"Calibration Points: {point_count}/4", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Click on points, then enter arm coordinates", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if point_count >= 4:
            if self.calculate_transformation_matrix():
                self.save_calibration()
                print("Calibration completed successfully!")
                return True
            else:
                print("Calibration failed")
                return False
        else:
            print("Need at least 4 calibration points")
            return False
    
    def test_calibration(self):
        """Test calibration accuracy"""
        if not self.calibrated:
            print("Not calibrated yet")
            return
        
        print("Testing Calibration")
        print("==================")
        
        # Test with known points
        test_points = [
            (320, 240),  # Center
            (160, 120),  # Top-left
            (480, 120),  # Top-right
            (160, 360),  # Bottom-left
            (480, 360),  # Bottom-right
        ]
        
        for pixel_x, pixel_y in test_points:
            arm_coords = self.pixel_to_arm(pixel_x, pixel_y)
            if arm_coords is not None:
                print(f"Pixel({pixel_x}, {pixel_y}) -> Arm({arm_coords[0]:.1f}, {arm_coords[1]:.1f})")

def main():
    """Main function for calibration"""
    mapper = CoordinateMapper()
    
    print("Coordinate Mapping System")
    print("========================")
    print("1. Interactive calibration")
    print("2. Load existing calibration")
    print("3. Test calibration")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            mapper.interactive_calibration()
        elif choice == '2':
            filename = input("Enter calibration filename (default: calibration.json): ").strip()
            if not filename:
                filename = "calibration.json"
            mapper.load_calibration(filename)
        elif choice == '3':
            mapper.test_calibration()
        elif choice == '4':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
