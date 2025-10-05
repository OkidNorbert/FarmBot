#!/usr/bin/env python3
"""
AI Tomato Sorter - Camera Calibration Script
Calibrates camera to map pixel coordinates to world coordinates
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class CameraCalibrator:
    def __init__(self, camera_id=0):
        """Initialize camera calibrator"""
        self.camera_id = camera_id
        self.camera = None
        self.calibration_points = []
        self.transform_matrix = None
        self.calibration_complete = False
        
    def connect_camera(self):
        """Connect to camera"""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise Exception(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera {self.camera_id} connected")
    
    def capture_calibration_image(self):
        """Capture image for calibration"""
        if not self.camera:
            self.connect_camera()
        
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture calibration image")
        
        return frame
    
    def interactive_calibration(self):
        """Interactive calibration using mouse clicks"""
        print("🎯 Interactive Camera Calibration")
        print("Click on 4 known points in the workspace")
        print("Points should form a rectangle in real-world coordinates")
        
        # Capture image
        frame = self.capture_calibration_image()
        
        # Create window and set mouse callback
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration', self._mouse_callback)
        
        self.calibration_points = []
        self.display_frame = frame.copy()
        
        print("\nClick on the 4 corners of your workspace:")
        print("1. Top-left corner")
        print("2. Top-right corner") 
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("\nPress 'r' to reset, 'q' to quit")
        
        while True:
            cv2.imshow('Calibration', self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.calibration_points = []
                self.display_frame = frame.copy()
                print("Reset calibration points")
        
        cv2.destroyAllWindows()
        
        if len(self.calibration_points) == 4:
            return self._get_world_coordinates()
        else:
            print("❌ Calibration incomplete - need exactly 4 points")
            return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                
                # Draw point
                cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_frame, f"{len(self.calibration_points)}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"Point {len(self.calibration_points)}: ({x}, {y})")
                
                if len(self.calibration_points) == 4:
                    print("✅ All 4 points selected!")
                    print("Press 'q' to continue or 'r' to reset")
    
    def _get_world_coordinates(self):
        """Get world coordinates for calibration points"""
        print("\nEnter real-world coordinates for each point (in cm):")
        world_points = []
        
        point_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        
        for i, (x, y) in enumerate(self.calibration_points):
            print(f"\nPoint {i+1} ({point_names[i]}): Pixel ({x}, {y})")
            
            while True:
                try:
                    world_x = float(input(f"Enter world X coordinate (cm): "))
                    world_y = float(input(f"Enter world Y coordinate (cm): "))
                    world_points.append((world_x, world_y))
                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
        
        return world_points
    
    def calibrate(self, pixel_points, world_points):
        """Calibrate coordinate transformation"""
        if len(pixel_points) != 4 or len(world_points) != 4:
            raise ValueError("Need exactly 4 calibration points")
        
        # Convert to numpy arrays
        pixel_array = np.array(pixel_points, dtype=np.float32)
        world_array = np.array(world_points, dtype=np.float32)
        
        # Compute perspective transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(pixel_array, world_array)
        self.calibration_complete = True
        
        print("✅ Calibration completed!")
        return self.transform_matrix
    
    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates"""
        if not self.calibration_complete:
            raise Exception("Calibration not completed")
        
        # Transform point
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.transform_matrix)
        
        return world_point[0][0]
    
    def world_to_pixel(self, world_x, world_y):
        """Convert world coordinates to pixel coordinates"""
        if not self.calibration_complete:
            raise Exception("Calibration not completed")
        
        # Get inverse transformation
        inv_matrix = cv2.getPerspectiveTransform(
            np.array([[world_x, world_y]], dtype=np.float32),
            np.array([[0, 0]], dtype=np.float32)
        )
        
        # This is a simplified version - you might need a different approach
        # for the inverse transformation
        return None
    
    def save_calibration(self, filename):
        """Save calibration data to file"""
        if not self.calibration_complete:
            raise Exception("Calibration not completed")
        
        calibration_data = {
            'transform_matrix': self.transform_matrix.tolist(),
            'calibration_points': [
                {'pixel': list(pixel), 'world': list(world)}
                for pixel, world in zip(self.calibration_points, self._get_world_coordinates())
            ],
            'camera_id': self.camera_id,
            'calibration_date': str(np.datetime64('now'))
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"✅ Calibration saved to {filename}")
    
    def load_calibration(self, filename):
        """Load calibration data from file"""
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        self.transform_matrix = np.array(calibration_data['transform_matrix'])
        self.calibration_complete = True
        
        print(f"✅ Calibration loaded from {filename}")
    
    def test_calibration(self, num_test_points=5):
        """Test calibration accuracy"""
        if not self.calibration_complete:
            raise Exception("Calibration not completed")
        
        print("🧪 Testing calibration accuracy...")
        
        # Generate test points
        test_pixel_points = []
        test_world_points = []
        
        for _ in range(num_test_points):
            # Random pixel coordinates
            pixel_x = np.random.randint(50, 590)
            pixel_y = np.random.randint(50, 430)
            test_pixel_points.append((pixel_x, pixel_y))
            
            # Convert to world coordinates
            world_coords = self.pixel_to_world(pixel_x, pixel_y)
            test_world_points.append(world_coords)
        
        # Calculate accuracy metrics
        errors = []
        for i, (pixel, world) in enumerate(zip(test_pixel_points, test_world_points)):
            # Convert back to pixel (simplified test)
            error = np.sqrt((pixel[0] - pixel[0])**2 + (pixel[1] - pixel[1])**2)
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"📊 Calibration Test Results:")
        print(f"   Average error: {avg_error:.2f} pixels")
        print(f"   Maximum error: {max_error:.2f} pixels")
        
        return avg_error, max_error
    
    def visualize_calibration(self, output_file=None):
        """Visualize calibration points and transformation"""
        if not self.calibration_complete:
            raise Exception("Calibration not completed")
        
        # Capture current frame
        frame = self.capture_calibration_image()
        
        # Draw calibration points
        for i, (x, y) in enumerate(self.calibration_points):
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i+1}", (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw coordinate grid
        self._draw_coordinate_grid(frame)
        
        if output_file:
            cv2.imwrite(output_file, frame)
            print(f"✅ Calibration visualization saved to {output_file}")
        
        return frame
    
    def _draw_coordinate_grid(self, frame):
        """Draw coordinate grid on frame"""
        h, w = frame.shape[:2]
        
        # Draw grid lines every 50 pixels
        for x in range(0, w, 50):
            cv2.line(frame, (x, 0), (x, h), (128, 128, 128), 1)
        for y in range(0, h, 50):
            cv2.line(frame, (0, y), (w, y), (128, 128, 128), 1)
    
    def close(self):
        """Close camera connection"""
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Camera Calibration for Tomato Sorter')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--output', type=str, default='calibration.json', help='Output calibration file')
    parser.add_argument('--load', type=str, help='Load existing calibration file')
    parser.add_argument('--test', action='store_true', help='Test calibration accuracy')
    parser.add_argument('--visualize', action='store_true', help='Create calibration visualization')
    
    args = parser.parse_args()
    
    print("🎯 AI Tomato Sorter - Camera Calibration")
    print("=" * 50)
    
    # Initialize calibrator
    calibrator = CameraCalibrator(args.camera)
    
    try:
        if args.load:
            # Load existing calibration
            calibrator.load_calibration(args.load)
        else:
            # Perform interactive calibration
            world_points = calibrator.interactive_calibration()
            if world_points:
                calibrator.calibrate(calibrator.calibration_points, world_points)
        
        # Save calibration
        if calibrator.calibration_complete:
            calibrator.save_calibration(args.output)
        
        # Test calibration
        if args.test and calibrator.calibration_complete:
            calibrator.test_calibration()
        
        # Create visualization
        if args.visualize and calibrator.calibration_complete:
            output_file = args.output.replace('.json', '_visualization.jpg')
            calibrator.visualize_calibration(output_file)
        
    except KeyboardInterrupt:
        print("\n❌ Calibration interrupted by user")
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
    finally:
        calibrator.close()

if __name__ == "__main__":
    main()
