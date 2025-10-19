#!/usr/bin/env python3
"""
Raspberry Pi Main Controller for AI Tomato Sorter
Handles camera detection, coordinate mapping, and robotic arm control
"""

import cv2
import numpy as np
import yaml
import time
import serial
import threading
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from web_interface import app
    from models.tomato.tomato_inference import TomatoClassifier
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Running in standalone mode...")

class PiController:
    def __init__(self, config_file="pi_config.yaml"):
        """Initialize the Raspberry Pi controller"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Initialize components
        self.camera = None
        self.arduino = None
        self.classifier = None
        
        # State management
        self.running = False
        self.last_detection = 0
        self.detection_count = 0
        
        # Coordinate mapping
        self.camera_to_arm_matrix = None
        self.setup_coordinate_mapping()
        
        print("🍅 AI Tomato Sorter Pi Controller Initialized")
    
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'camera': {'index': 0, 'width': 640, 'height': 480, 'fps': 30},
            'arduino': {'port': '/dev/ttyUSB0', 'baudrate': 115200},
            'arm': {
                'home_position': [90, 90, 90],
                'bin_positions': {
                    'not_ready': [0, 45, 90],
                    'ready': [90, 45, 90],
                    'spoilt': [180, 45, 90]
                },
                'arm_length_1': 100.0,
                'arm_length_2': 80.0,
                'workspace_x': [-150, 150],
                'workspace_y': [50, 200]
            },
            'detection': {'confidence_threshold': 0.7, 'detection_interval': 2.0},
            'web_interface': {'host': '0.0.0.0', 'port': 5000, 'debug': False}
        }
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pi_controller.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_coordinate_mapping(self):
        """Setup coordinate mapping from camera to robotic arm"""
        # This creates a transformation matrix from camera pixels to arm coordinates
        # You'll need to calibrate this based on your physical setup
        
        # Camera resolution
        cam_width = self.config['camera']['width']
        cam_height = self.config['camera']['height']
        
        # Arm workspace (in mm)
        workspace_x = self.config['arm']['workspace_x']
        workspace_y = self.config['arm']['workspace_y']
        
        # Create transformation matrix (simplified - you may need more complex calibration)
        self.camera_to_arm_matrix = np.array([
            [(workspace_x[1] - workspace_x[0]) / cam_width, 0, workspace_x[0]],
            [0, (workspace_y[1] - workspace_y[0]) / cam_height, workspace_y[0]],
            [0, 0, 1]
        ])
        
        self.logger.info("Coordinate mapping initialized")
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.config['camera']['index'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            self.logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def initialize_arduino(self):
        """Initialize Arduino connection"""
        try:
            # Try different possible ports
            possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
            port = self.config['arduino']['port']
            
            if port not in possible_ports:
                # Auto-detect port
                for p in possible_ports:
                    if os.path.exists(p):
                        port = p
                        break
            
            self.arduino = serial.Serial(
                port=port,
                baudrate=self.config['arduino']['baudrate'],
                timeout=1
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Send status command to test connection
            self.arduino.write(b"STATUS\n")
            response = self.arduino.readline().decode().strip()
            
            if "STATUS" in response:
                self.logger.info(f"Arduino connected on {port}")
                return True
            else:
                raise Exception("Arduino not responding")
                
        except Exception as e:
            self.logger.error(f"Arduino initialization failed: {e}")
            return False
    
    def initialize_classifier(self):
        """Initialize AI classifier"""
        try:
            model_path = "models/tomato/best_model.pth"
            if os.path.exists(model_path):
                self.classifier = TomatoClassifier(model_path)
                self.logger.info("AI classifier initialized")
                return True
            else:
                self.logger.warning("Model file not found, using dummy classifier")
                self.classifier = DummyClassifier()
                return True
        except Exception as e:
            self.logger.error(f"Classifier initialization failed: {e}")
            self.classifier = DummyClassifier()
            return False
    
    def camera_to_arm_coordinates(self, pixel_x, pixel_y):
        """Convert camera pixel coordinates to arm coordinates"""
        # Convert pixel coordinates to homogeneous coordinates
        pixel_point = np.array([pixel_x, pixel_y, 1])
        
        # Transform to arm coordinates
        arm_point = self.camera_to_arm_matrix @ pixel_point
        
        return arm_point[0], arm_point[1]
    
    def calculate_pick_position(self, tomato_center, tomato_size):
        """Calculate optimal pick position for tomato"""
        x, y = tomato_center
        
        # Convert to arm coordinates
        arm_x, arm_y = self.camera_to_arm_coordinates(x, y)
        
        # Estimate depth based on tomato size (larger = closer)
        # This is a simplified approach - you might want to use stereo vision
        depth = self.estimate_depth_from_size(tomato_size)
        
        # Calculate final pick position
        pick_x = arm_x
        pick_y = arm_y
        pick_z = depth  # Height above workspace
        
        return pick_x, pick_y, pick_z
    
    def estimate_depth_from_size(self, tomato_size):
        """Estimate depth based on tomato size in pixels"""
        # This is a simplified depth estimation
        # In a real system, you might use stereo vision or depth sensors
        
        # Assume tomatoes are roughly 50mm in diameter
        # Larger in pixels = closer to camera
        expected_diameter_mm = 50
        expected_diameter_pixels = 100  # Calibrate this based on your setup
        
        depth_factor = expected_diameter_pixels / tomato_size
        depth = 50 + (depth_factor * 30)  # Adjust based on your setup
        
        return max(20, min(100, depth))  # Clamp between 20-100mm
    
    def detect_tomatoes(self, frame):
        """Detect tomatoes in frame using AI"""
        if not self.classifier:
            return []
        
        try:
            # Run inference
            detections = self.classifier.detect_tomatoes(frame)
            
            results = []
            for detection in detections:
                if detection['confidence'] > self.config['detection']['confidence_threshold']:
                    results.append({
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'center': detection['center']
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if not self.arduino:
            return False
        
        try:
            self.arduino.write(f"{command}\n".encode())
            time.sleep(0.1)  # Small delay for Arduino processing
            return True
        except Exception as e:
            self.logger.error(f"Arduino command failed: {e}")
            return False
    
    def pick_and_sort_tomato(self, tomato_detection):
        """Pick and sort a detected tomato"""
        try:
            # Extract tomato information
            center = tomato_detection['center']
            size = max(tomato_detection['bbox'][2], tomato_detection['bbox'][3])
            class_id = tomato_detection['class']
            
            # Calculate pick position
            pick_x, pick_y, pick_z = self.calculate_pick_position(center, size)
            
            self.logger.info(f"Picking tomato at ({pick_x:.1f}, {pick_y:.1f}, {pick_z:.1f}) - Class: {class_id}")
            
            # Move to pick position
            self.send_arduino_command(f"MOVE {pick_x:.1f} {pick_y:.1f} {class_id}")
            
            # Wait for movement to complete
            time.sleep(2)
            
            # Close gripper
            self.send_arduino_command("GRIP CLOSE")
            time.sleep(1)
            
            # Move to appropriate bin
            bin_positions = self.config['arm']['bin_positions']
            if class_id == 0:  # Not ready
                bin_pos = bin_positions['not_ready']
            elif class_id == 1:  # Ready
                bin_pos = bin_positions['ready']
            else:  # Spoilt
                bin_pos = bin_positions['spoilt']
            
            self.send_arduino_command(f"ANGLE {bin_pos[0]} {bin_pos[1]} {bin_pos[2]}")
            time.sleep(2)
            
            # Open gripper to drop tomato
            self.send_arduino_command("GRIP OPEN")
            time.sleep(1)
            
            # Return to home position
            self.send_arduino_command("HOME")
            time.sleep(2)
            
            self.logger.info(f"Tomato sorted successfully - Class: {class_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Pick and sort failed: {e}")
            return False
    
    def run_detection_loop(self):
        """Main detection and sorting loop"""
        self.logger.info("Starting detection loop...")
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Check if enough time has passed since last detection
                current_time = time.time()
                if current_time - self.last_detection < self.config['detection']['detection_interval']:
                    time.sleep(0.1)
                    continue
                
                # Detect tomatoes
                detections = self.detect_tomatoes(frame)
                
                if detections:
                    self.logger.info(f"Found {len(detections)} tomatoes")
                    
                    # Process each detection
                    for detection in detections:
                        if self.pick_and_sort_tomato(detection):
                            self.detection_count += 1
                            self.last_detection = current_time
                            break  # Process one tomato at a time
                else:
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                self.logger.info("Detection loop interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Detection loop error: {e}")
                time.sleep(1)
    
    def start_web_interface(self):
        """Start web interface in separate thread"""
        try:
            # Import and start Flask app
            from web_interface import app
            
            # Add Pi-specific routes
            @app.route('/pi/status')
            def pi_status():
                return {
                    'running': self.running,
                    'detection_count': self.detection_count,
                    'camera_connected': self.camera is not None,
                    'arduino_connected': self.arduino is not None,
                    'classifier_loaded': self.classifier is not None
                }
            
            @app.route('/pi/control/start')
            def start_detection():
                if not self.running:
                    self.running = True
                    threading.Thread(target=self.run_detection_loop, daemon=True).start()
                return {'status': 'started'}
            
            @app.route('/pi/control/stop')
            def stop_detection():
                self.running = False
                return {'status': 'stopped'}
            
            # Start Flask app
            app.run(
                host=self.config['web_interface']['host'],
                port=self.config['web_interface']['port'],
                debug=self.config['web_interface']['debug']
            )
            
        except Exception as e:
            self.logger.error(f"Web interface failed: {e}")
    
    def run(self):
        """Main run method"""
        self.logger.info("Starting AI Tomato Sorter...")
        
        # Initialize components
        if not self.initialize_camera():
            self.logger.error("Failed to initialize camera")
            return
        
        if not self.initialize_arduino():
            self.logger.error("Failed to initialize Arduino")
            return
        
        if not self.initialize_classifier():
            self.logger.error("Failed to initialize classifier")
            return
        
        # Start web interface in separate thread
        web_thread = threading.Thread(target=self.start_web_interface, daemon=True)
        web_thread.start()
        
        # Wait a moment for web interface to start
        time.sleep(2)
        
        # Start detection loop
        self.running = True
        self.run_detection_loop()
    
    def shutdown(self):
        """Shutdown the system"""
        self.logger.info("Shutting down...")
        self.running = False
        
        if self.arduino:
            self.send_arduino_command("HOME")
            self.arduino.close()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()

class DummyClassifier:
    """Dummy classifier for testing when model is not available"""
    def detect_tomatoes(self, frame):
        # Return dummy detection for testing
        return [{
            'class': 1,  # Ready
            'confidence': 0.9,
            'bbox': [100, 100, 50, 50],
            'center': [125, 125]
        }]

def main():
    """Main function"""
    controller = PiController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    main()
