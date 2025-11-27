#!/usr/bin/env python3
"""
Hardware Controller for AI Tomato Sorter
Manages connections to Arduino (Robotic Arm) and Camera
"""

import cv2
import numpy as np
import serial
import time
import threading
import os
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from models.tomato.tomato_inference import TomatoClassifier
except ImportError:
    print("Warning: Could not import TomatoClassifier. Using dummy.")
    TomatoClassifier = None

class HardwareController:
    def __init__(self):
        """Initialize Hardware Controller"""
        self.setup_logging()
        
        # Components
        self.arduino = None
        self.camera = None
        self.classifier = None
        self.camera_lock = threading.Lock()
        
        # State
        self.arduino_connected = False
        self.camera_connected = False
        self.auto_mode = False
        self.last_frame = None
        
        # Configuration
        self.arduino_port = '/dev/ttyUSB0'
        self.arduino_baud = 115200
        self.camera_index = 0
        self.detection_interval = 1.0 # Seconds between detections
        self.last_detection_time = 0
        
        # Initialize AI
        self.initialize_classifier()
        
        # Try to connect
        self.connect_hardware()
        
        # Start Auto Loop Thread
        threading.Thread(target=self._auto_loop, daemon=True).start()

    def initialize_classifier(self):
        """Initialize AI Model"""
        if TomatoClassifier:
            try:
                model_path = "models/tomato/best_model.pth"
                if os.path.exists(model_path):
                    self.classifier = TomatoClassifier(model_path)
                    self.logger.info("AI Model Loaded")
                else:
                    self.logger.warning("Model file not found. Using dummy.")
            except Exception as e:
                self.logger.error(f"Failed to load AI model: {e}")

    def _auto_loop(self):
        """Main Autonomous Loop"""
        self.logger.info("Auto Loop Started")
        while True:
            if self.auto_mode:
                try:
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.detection_interval:
                        self.process_auto_cycle()
                        self.last_detection_time = current_time
                except Exception as e:
                    self.logger.error(f"Auto Loop Error: {e}")
            
            time.sleep(0.1) # Prevent CPU hogging

    def process_auto_cycle(self):
        """Single cycle of autonomous logic"""
        frame = self.get_frame()
        if frame is None:
            return

        # 1. Detect
        detections = self.detect_tomatoes(frame)
        
        # 2. Decide
        if detections:
            # Pick the most confident detection
            target = max(detections, key=lambda x: x['confidence'])
            self.logger.info(f"[AUTO] Found {target['class']} tomato ({target['confidence']:.2f})")
            
            # 3. Act (if connected)
            if self.arduino_connected:
                # Calculate coordinates (Simplified mapping)
                # In real usage, map pixels to mm
                center_x, center_y = target['center']
                
                # Get Depth from VL53L0X sensor
                z_depth = self.get_distance_sensor()
                
                # Use default distance if sensor fails to prevent command parsing errors
                DEFAULT_DISTANCE_MM = 50  # Safe default distance
                if z_depth is None:
                    self.logger.warning(f"[AUTO] Distance sensor unavailable, using default {DEFAULT_DISTANCE_MM}mm")
                    z_depth = DEFAULT_DISTANCE_MM
                
                self.logger.info(f"[AUTO] Picking at {center_x}, {center_y}, {z_depth}mm")
                
                # Send Pick Command
                # Format: PICK <x> <y> <z> <class_id>
                # class_id: 0=Unripe, 1=Ripe (example)
                class_id = 1 if target['class'] == 'ripe' else 0
                self.send_command(f"PICK {center_x} {center_y} {z_depth} {class_id}")
                
                # Wait for operation to complete
                time.sleep(5) 
            else:
                self.logger.info("[AUTO] Simulation: Pick command sent")

    def detect_tomatoes(self, frame):
        """Run detection on frame"""
        if self.classifier:
            return self.classifier.detect_tomatoes(frame)
        else:
            # Dummy detection for simulation
            # Randomly find a tomato every few seconds
            if time.time() % 5 < 1: 
                return [{
                    'class': 'ripe',
                    'confidence': 0.95,
                    'bbox': [100, 100, 200, 200],
                    'center': [150, 150]
                }]
            return []

    def get_distance_sensor(self):
        """Read distance from VL53L0X via Arduino"""
        if self.arduino_connected and self.arduino:
            try:
                # Send DISTANCE command
                self.arduino.write(b"DISTANCE\n")
                time.sleep(0.1)  # Wait for response
                
                # Read response
                response = self.arduino.readline().decode().strip()
                
                if response.startswith("DISTANCE: "):
                    distance_str = response.replace("DISTANCE: ", "")
                    if distance_str == "OUT_OF_RANGE" or distance_str == "SENSOR_NOT_AVAILABLE":
                        self.logger.warning(f"Distance sensor: {distance_str}")
                        return None
                    try:
                        distance_mm = int(distance_str)
                        return distance_mm
                    except ValueError:
                        self.logger.error(f"Invalid distance reading: {distance_str}")
                        return None
                else:
                    self.logger.warning(f"Unexpected distance response: {response}")
                    return None
            except Exception as e:
                self.logger.error(f"Distance sensor read error: {e}")
                return None
        else:
            # Simulation mode
            return 50  # mm (dummy value)

    def start_auto_mode(self):
        self.auto_mode = True
        self.logger.info("Auto Mode ENABLED")
        return True

    def stop_auto_mode(self):
        self.auto_mode = False
        self.logger.info("Auto Mode DISABLED")
        return True
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HardwareController")

    def connect_hardware(self):
        """Attempt to connect to hardware"""
        # Connect Arduino
        try:
            # Try auto-detecting port if default doesn't exist
            port = self.arduino_port
            if not os.path.exists(port):
                for p in ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']:
                    if os.path.exists(p):
                        port = p
                        break
            
            if os.path.exists(port):
                self.arduino = serial.Serial(port, self.arduino_baud, timeout=1)
                time.sleep(2) # Wait for reset
                self.arduino_connected = True
                self.logger.info(f"Arduino connected on {port}")
            else:
                self.logger.warning("Arduino not found - Simulation Mode")
        except Exception as e:
            self.logger.error(f"Arduino connection failed: {e}")
            self.arduino_connected = False

        # Connect Camera
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                self.camera_connected = True
                self.logger.info("Camera connected")
                # Start frame reading thread
                threading.Thread(target=self._update_frame, daemon=True).start()
            else:
                self.logger.warning("Camera not found - Simulation Mode")
        except Exception as e:
            self.logger.error(f"Camera connection failed: {e}")
            self.camera_connected = False

    def _update_frame(self):
        """Background thread to keep reading frames"""
        while self.camera_connected:
            ret, frame = self.camera.read()
            if ret:
                with self.camera_lock:
                    self.last_frame = frame
            time.sleep(0.03) # ~30 FPS

    def get_frame(self):
        """Get the latest camera frame"""
        if self.camera_connected:
            with self.camera_lock:
                if self.last_frame is not None:
                    return self.last_frame.copy()
        
        # Return placeholder if no camera
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"NO CAMERA - {datetime.now().strftime('%H:%M:%S')}", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def send_command(self, command):
        """Send G-code style command to Arduino"""
        if self.arduino_connected and self.arduino:
            try:
                self.arduino.write(f"{command}\n".encode())
                self.logger.info(f"Sent to Arduino: {command}")
                return True
            except Exception as e:
                self.logger.error(f"Serial write error: {e}")
                return False
        else:
            self.logger.info(f"[SIMULATION] Command: {command}")
            return True

    def move_arm(self, x, y, z):
        """Move arm to coordinates"""
        return self.send_command(f"MOVE {x} {y} {z}")

    def home_arm(self):
        """Home the arm"""
        return self.send_command("HOME")

    def set_gripper(self, state):
        """Control gripper (OPEN/CLOSE)"""
        return self.send_command(f"GRIPPER {state}")
    
    # Conveyor belt not available in current setup
    # def set_conveyor(self, speed):
    #     """Control conveyor speed (0-100)"""
    #     return self.send_command(f"CONVEYOR {speed}")

    def get_status(self):
        """Get hardware status"""
        return {
            'arduino_connected': self.arduino_connected,
            'camera_connected': self.camera_connected,
            'auto_mode': self.auto_mode
        }
