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

# Suppress OpenCV warnings and errors globally
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Only show errors, suppress warnings
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'  # Disable video I/O debug messages
# Note: cv2.setLogLevel() may not be available in all OpenCV versions
try:
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(0)  # 0 = SILENT, 1 = ERROR, 2 = WARN, 3 = INFO, 4 = DEBUG
except:
    pass

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Try to import BLE libraries
try:
    import asyncio
    from bleak import BleakScanner, BleakClient
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    print("Warning: Bleak library not available. Bluetooth support disabled.")

try:
    # Try importing from the models directory
    from models.tomato.tomato_inference import TomatoClassifier
    TOMATO_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TomatoClassifier: {e}")
    print("Attempting alternative import path...")
    try:
        # Try adding the project root to path and importing
        import sys
        project_root = Path(__file__).parent.absolute()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from models.tomato.tomato_inference import TomatoClassifier
        TOMATO_CLASSIFIER_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Could not import TomatoClassifier (alternative path also failed): {e2}")
        TomatoClassifier = None
        TOMATO_CLASSIFIER_AVAILABLE = False

class BLEClient:
    """Bluetooth Low Energy client for Arduino R4 WiFi"""
    def __init__(self, service_uuid=None, char_uuid=None, target_name="Arduino", on_connect_callback=None):
        if not BLE_AVAILABLE:
            raise ImportError("Bleak library not available")
        
        self.service_uuid = service_uuid or '19B10000-E8F2-537E-4F6C-D104768A1214'
        self.char_uuid = char_uuid or '19B10001-E8F2-537E-4F6C-D104768A1214'
        self.target_name = target_name
        self.client = None
        self.loop = None
        self.thread = None
        self.connected = False
        self.command_queue = asyncio.Queue()
        self.device_address = None
        self.on_connect_callback = on_connect_callback  # Callback when connection state changes
        
    def _run_loop(self):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_loop())
    
    async def _main_loop(self):
        """Main BLE connection loop"""
        while True:
            if not self.connected:
                await self._connect()
            else:
                await asyncio.sleep(1)
    
    async def _connect(self):
        """Connect to BLE device"""
        try:
            self.logger.info(f"Scanning for BLE device: {self.target_name}...")
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: d.name and self.target_name in d.name
            )
            
            if device:
                self.device_address = device.address
                self.logger.info(f"Found device: {device.name} ({device.address})")
                self.client = BleakClient(device)
                try:
                    await self.client.connect()
                    self.connected = True
                    self.logger.info("BLE Connected!")
                    
                    # Notify callback if connection state changed
                    if self.on_connect_callback:
                        self.on_connect_callback(True)
                    
                    # Process commands
                    while self.connected:
                        try:
                            command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                            if command:
                                await self._send_command(command)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            self.logger.error(f"Command send error: {e}")
                except Exception as e:
                    self.logger.error(f"BLE Connection failed: {e}")
                    self.connected = False
                    if self.on_connect_callback:
                        self.on_connect_callback(False)
                    await asyncio.sleep(5)
            else:
                self.logger.warning("Device not found, retrying in 5 seconds...")
                await asyncio.sleep(5)
        except Exception as e:
            # Check for specific Bluetooth adapter errors
            error_str = str(e)
            if "No Bluetooth adapters found" in error_str or "NO_BLUETOOTH" in str(e):
                # Only log this error once per minute to avoid spam
                current_time = time.time()
                if not hasattr(self, '_last_bluetooth_error_time') or \
                   (current_time - getattr(self, '_last_bluetooth_error_time', 0)) > 60:
                    self.logger.error("=" * 60)
                    self.logger.error("❌ BLUETOOTH ADAPTER NOT FOUND")
                    self.logger.error("=" * 60)
                    self.logger.error("Your system does not have a Bluetooth adapter available.")
                    self.logger.error("")
                    self.logger.error("To fix this:")
                    self.logger.error("1. Check if your computer has Bluetooth hardware")
                    self.logger.error("2. Enable Bluetooth in system settings")
                    self.logger.error("3. Start Bluetooth service:")
                    self.logger.error("   sudo systemctl start bluetooth")
                    self.logger.error("   sudo systemctl enable bluetooth")
                    self.logger.error("4. Check Bluetooth status:")
                    self.logger.error("   bluetoothctl power on")
                    self.logger.error("   bluetoothctl show")
                    self.logger.error("")
                    self.logger.error("If you don't have Bluetooth, use Serial connection instead:")
                    self.logger.error("   Set connection_type='serial' in HardwareController")
                    self.logger.error("=" * 60)
                    self._last_bluetooth_error_time = current_time
            else:
                self.logger.error(f"BLE scan error: {e}")
                await asyncio.sleep(5)
    
    async def _send_command(self, command):
        """Send command via BLE"""
        try:
            # Check if client is still connected
            if not self.client.is_connected:
                self.logger.error("BLE client not connected, cannot send command")
                self.connected = False
                if self.on_connect_callback:
                    self.on_connect_callback(False)
                return
            
            # Send command with newline
            command_bytes = f"{command}\n".encode('utf-8')
            await self.client.write_gatt_char(self.char_uuid, command_bytes)
            self.logger.info(f"Sent BLE command: {command}")
        except Exception as e:
            self.logger.error(f"BLE write error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.connected = False
            if self.on_connect_callback:
                self.on_connect_callback(False)
    
    def start(self, logger=None):
        """Start BLE client in background thread"""
        self.logger = logger or logging.getLogger("BLEClient")
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def send_command(self, command):
        """Send command (thread-safe)"""
        if self.connected and self.loop:
            self.loop.call_soon_threadsafe(self.command_queue.put_nowait, command)
            return True
        return False
    
    def disconnect(self):
        """Disconnect from BLE device and clean up"""
        old_connected = self.connected
        self.connected = False
        
        if self.client and old_connected:
            try:
                if self.loop and self.loop.is_running():
                    # Try to schedule disconnect in the event loop
                    try:
                        future = asyncio.run_coroutine_threadsafe(self.client.disconnect(), self.loop)
                        # Wait up to 2 seconds for disconnect to complete
                        future.result(timeout=2)
                    except Exception as e:
                        self.logger.debug(f"Could not schedule async disconnect: {e}")
            except Exception as e:
                self.logger.warning(f"Error during BLE disconnect: {e}")
        
        # Reset client reference - the thread will exit naturally when connected is False
        # Give it a moment to clean up
        time.sleep(0.5)

class HardwareController:
    def __init__(self, connection_type='auto', ble_device_name="Arduino"):
        """Initialize Hardware Controller
        
        Args:
            connection_type: 'serial', 'bluetooth', or 'auto' (tries serial first, then bluetooth)
            ble_device_name: Name of BLE device to connect to
        """
        self.setup_logging()
        
        # Get project root directory (where this file is located)
        self.project_root = Path(__file__).parent.absolute()
        
        # Components
        self.arduino = None
        self.ble_client = None
        self.camera = None
        self.classifier = None
        self.yolo_detector = None
        self.camera_lock = threading.Lock()
        
        # State
        self.arduino_connected = False
        self.camera_connected = False
        self.auto_mode = False
        self.last_frame = None
        self.connection_type = connection_type
        self.ble_device_name = ble_device_name
        self.frame_update_thread = None
        self.frame_update_running = False
        
        # Servo availability configuration
        # Set to False for servos that are not available (manually fixed)
        self.servo_available = {
            'base': False,      # Base servo not available - manually fixed
            'shoulder': False,  # Shoulder servo not available - manually adjusted
            'forearm': True,   # Forearm servo available
            'elbow': True,     # Elbow servo available
            'pitch': True,     # Pitch servo available
            'claw': True       # Claw servo available
        }
        
        # Fixed positions for unavailable servos (manually set)
        # These should match your manual adjustments
        self.fixed_servo_angles = {
            'base': 90,        # Manually fixed base position
            'shoulder': 135,   # Manually adjusted shoulder (gives forearm clearance to reach floor)
        }
        
        # Track current servo angles (for front/back detection)
        # Servo indices: 0=base, 1=shoulder/arm, 2=forearm, 3=elbow/wrist_yaw, 4=pitch, 5=claw
        self.current_servo_angles = {
            'base': self.fixed_servo_angles['base'],
            'shoulder': self.fixed_servo_angles['shoulder'],
            'forearm': 90,
            'elbow': 90,  # Also called 'wrist_yaw' in backend
            'pitch': 90,  # Also called 'wrist_pitch' in backend
            'claw': 0
        }
        
        # Configuration
        self.arduino_port = '/dev/ttyUSB0'
        self.arduino_baud = 115200
        self.camera_index = 0
        self.detection_interval = 1.0 # Seconds between detections
        self.last_detection_time = 0
        
        # Calibration and coordinate mapping
        self.homography_matrix = None
        self.workspace_bounds = {
            'x_min': -150, 'x_max': 150,  # mm
            'y_min': 50, 'y_max': 250,    # mm
            'z_min': 5, 'z_max': 150      # mm (lowered to 5mm to allow surface-level tomatoes)
        }
        calibration_loaded = self.load_calibration_matrix()
        if calibration_loaded:
            self.logger.info("✅ Calibration loaded - coordinate mapping active")
        else:
            self.logger.warning("⚠️  No calibration found - using fallback coordinate mapping")
        
        # Pick tracking
        self.pending_picks = {}  # Track pick commands sent to Arduino
        self.pick_timeout = 10.0  # seconds to wait for pick completion
        self.max_pick_retries = 2
        
        # Initialize AI
        self.initialize_classifier()
        
        # Try to connect
        self.connect_hardware()
        
        # Start Auto Loop Thread
        threading.Thread(target=self._auto_loop, daemon=True).start()

    def initialize_classifier(self):
        """Initialize AI Model - tries YOLO first, falls back to ResNet"""
        # Try YOLO first
        try:
            from models.tomato.yolo_inference import load_yolo_model, YOLO_AVAILABLE
            if YOLO_AVAILABLE:
                # Try to find YOLO model
                possible_paths = [
                    self.project_root / "models" / "tomato" / "best.pt",
                    self.project_root / "models" / "tomato" / "yolov8_tomato.pt",
                    self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
                    self.project_root / "runs" / "detect" / "tomato_detector" / "weights" / "best.pt"
                ]
                
                for model_path in possible_paths:
                    if model_path.exists():
                        self.yolo_detector = load_yolo_model(str(model_path), confidence_threshold=0.5)
                        if self.yolo_detector and self.yolo_detector.is_available():
                            self.logger.info(f"✅ YOLO Model Loaded: {model_path}")
                            self.logger.info("   Using YOLO for detection. ResNet will not be loaded.")
                            return  # YOLO loaded successfully, skip ResNet initialization
                
                self.logger.info("⚠️  YOLO available but no model found. Falling back to ResNet.")
        except ImportError:
            self.logger.info("⚠️  YOLO not available. Using ResNet classifier.")
        except Exception as e:
            self.logger.warning(f"YOLO initialization error: {e}, falling back to ResNet")
        
        # Fallback to ResNet classifier
        if not TOMATO_CLASSIFIER_AVAILABLE or not TomatoClassifier:
            self.logger.warning("TomatoClassifier not available. Model detection disabled.")
            self.logger.warning("Make sure PyTorch and the model files are installed.")
            return
        
        try:
            # Use absolute path from project root
            model_path = self.project_root / "models" / "tomato" / "best_model.pth"
            model_path_str = str(model_path)
            
            self.logger.info(f"Looking for ResNet model at: {model_path_str}")
            
            if model_path.exists():
                self.logger.info(f"Model file found: {model_path_str}")
                try:
                    self.classifier = TomatoClassifier(model_path_str)
                    self.logger.info("✅ ResNet Model Loaded successfully")
                except Exception as load_error:
                    self.logger.error(f"Failed to load model (file exists but load failed): {load_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.warning(f"❌ Model file not found at: {model_path_str}")
                self.logger.warning("Using dummy classifier. Model detection will not work.")
                # List what's in the models directory for debugging
                models_dir = self.project_root / "models" / "tomato"
                if models_dir.exists():
                    files = list(models_dir.glob("*"))
                    self.logger.info(f"Files in models/tomato: {[f.name for f in files]}")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

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

    def is_arm_facing_front(self):
        """Determine if arm is facing front based on shoulder and forearm angles
        
        Returns:
            bool: True if arm is facing front (shoulder and forearm 90-180°), False otherwise
        """
        # If shoulder is fixed, use fixed angle; otherwise use tracked angle
        if not self.servo_available.get('shoulder', True):
            shoulder_angle = self.fixed_servo_angles.get('shoulder', 135)
        else:
            shoulder_angle = self.current_servo_angles.get('shoulder', 90)
        
        forearm_angle = self.current_servo_angles.get('forearm', 90)
        
        # Front: shoulder and forearm are 90-180 degrees
        # Back: shoulder and forearm are 90-0 degrees
        is_shoulder_front = 90 <= shoulder_angle <= 180
        is_forearm_front = 90 <= forearm_angle <= 180
        is_shoulder_back = 0 <= shoulder_angle <= 90
        is_forearm_back = 0 <= forearm_angle <= 90
        
        # If both are clearly front or back, use that
        if is_shoulder_front and is_forearm_front:
            return True
        elif is_shoulder_back and is_forearm_back:
            return False
        else:
            # Mixed orientation - calculate average "frontness"
            # Frontness: 0 (back) to 1 (front) based on how far from 90° toward 180°
            shoulder_frontness = (shoulder_angle - 90) / 90.0  # 0 to 1 for 90-180
            forearm_frontness = (forearm_angle - 90) / 90.0  # 0 to 1 for 90-180
            avg_frontness = (shoulder_frontness + forearm_frontness) / 2.0
            
            # Consider front if average frontness > 0 (more toward 180 than 0)
            return avg_frontness > 0

    def process_auto_cycle(self):
        """Single cycle of autonomous logic - Only picks 'ready' tomatoes"""
        # Check if arm is facing front before detecting tomatoes
        # Tomatoes are always placed in front of the arm
        if not self.is_arm_facing_front():
            self.logger.debug("[AUTO] Arm is facing back, skipping detection (tomatoes are in front)")
            return
        
        frame = self.get_frame()
        if frame is None:
            return

        # 1. Detect (only when arm is facing front)
        detections = self.detect_tomatoes(frame)
        
        # 2. Filter: Only pick "ready" tomatoes in automatic mode
        # Lower confidence threshold for Ugandan tomatoes (they may have different appearance)
        ready_tomatoes = [d for d in detections if d['class'].lower() in ['ready', 'ripe']]
        
        if not ready_tomatoes:
            # No ready tomatoes found - skip this cycle
            if detections:
                self.logger.info(f"[AUTO] Found {len(detections)} tomatoes, but none are ready. Classes: {[d['class'] for d in detections]}")
            else:
                self.logger.debug("[AUTO] No tomatoes detected in frame")
            return
        
        # 3. Sort ready tomatoes by confidence (highest first) for better picking order
        ready_tomatoes.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.logger.info(f"[AUTO] Found {len(ready_tomatoes)} ready tomato(s) to pick")
        
        # 4. Pick ALL ready tomatoes in sequence
        picked_count = 0
        skipped_count = 0
        
        for i, target in enumerate(ready_tomatoes):
            self.logger.info(f"[AUTO] Processing tomato {i+1}/{len(ready_tomatoes)} (confidence: {target['confidence']:.2f}, class: {target['class']})")
            
            # Act (if connected)
            if not self.arduino_connected:
                self.logger.info(f"[AUTO] Simulation: Pick command {i+1}/{len(ready_tomatoes)} (Arduino not connected)")
                continue
            
            try:
                # Get pixel coordinates from detection
                center_x, center_y = target['center']
                
                # Convert pixel coordinates to arm coordinates (mm)
                arm_coords = self.pixel_to_arm_coordinates(center_x, center_y)
                if arm_coords is None:
                    self.logger.error(f"[AUTO] Failed to convert coordinates for tomato {i+1}, skipping")
                    skipped_count += 1
                    continue
                
                arm_x, arm_y = arm_coords
                self.logger.debug(f"[AUTO] Pixel ({center_x:.0f}, {center_y:.0f}) -> Arm ({arm_x:.1f}, {arm_y:.1f}) mm {'[CALIBRATED]' if self.homography_matrix is not None else '[FALLBACK]'}")
                
                # Get distance from ToF sensor (on claw) to tomato
                # NOTE: ToF is on the claw, so it moves with the arm
                # When arm is at approach position, ToF reads distance from claw to tomato
                tof_distance = self.get_distance_sensor()
                
                # Calculate accurate depth for tomato
                # Since ToF is on claw, distance is from claw to tomato surface
                z_depth = self.calculate_tomato_depth(target, tof_distance)
                
                if tof_distance is None:
                    self.logger.warning(f"[AUTO] ToF sensor unavailable, using calculated depth: {z_depth:.1f}mm")
                else:
                    self.logger.debug(f"[AUTO] ToF distance (claw to tomato): {tof_distance:.1f}mm, Calculated depth: {z_depth:.1f}mm")
                
                # Validate position is reachable
                if not self.is_position_reachable(arm_x, arm_y, z_depth):
                    self.logger.warning(f"[AUTO] Position ({arm_x:.1f}, {arm_y:.1f}, {z_depth:.1f}) out of reach, skipping tomato {i+1}")
                    skipped_count += 1
                    continue
                
                self.logger.info(f"[AUTO] Picking tomato {i+1} at arm coordinates: ({arm_x:.1f}, {arm_y:.1f}, {z_depth:.1f}) mm")
                
                # Send Pick Command
                # Format: PICK <x> <y> <z> <class_id>
                # class_id: 1=Ready/Ripe (goes to right), 0=Other (goes to left)
                # Since we only pick ready tomatoes, always use class_id=1 (right bin)
                class_id = 1  # Ready tomatoes go to the right
                
                # Generate unique pick ID for tracking
                pick_id = f"auto_{int(time.time() * 1000)}_{i}"
                
                # Send pick command
                pick_command = f"PICK {int(arm_x)} {int(arm_y)} {int(z_depth)} {class_id}"
                self.send_command(pick_command)
                
                # Track pick command
                self.pending_picks[pick_id] = {
                    'target': target,
                    'arm_coords': (arm_x, arm_y, z_depth),
                    'timestamp': time.time(),
                    'retries': 0
                }
                
                # Wait for pick operation to complete
                # Arduino pick sequence typically takes 3-8 seconds
                wait_time = 8.0
                time.sleep(wait_time)
                
                # Check if pick completed (would need Arduino feedback for accurate tracking)
                # For now, assume success after wait time
                if pick_id in self.pending_picks:
                    # Pick should have completed by now
                    del self.pending_picks[pick_id]
                    picked_count += 1
                    self.logger.info(f"[AUTO] Pick {i+1} completed (assumed success)")
                
            except Exception as e:
                self.logger.error(f"[AUTO] Error picking tomato {i+1}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue
        
        # Summary
        if picked_count > 0:
            self.logger.info(f"[AUTO] ✅ Successfully picked {picked_count} tomato(s)")
        if skipped_count > 0:
            self.logger.warning(f"[AUTO] ⚠️  Skipped {skipped_count} tomato(s)")
        
        self.logger.info(f"[AUTO] Completed cycle: {picked_count} picked, {skipped_count} skipped out of {len(ready_tomatoes)} ready tomatoes")

    def detect_tomatoes(self, frame):
        """Run detection on frame - uses YOLO if available, otherwise ResNet + color detection"""
        # Try YOLO first if available
        if self.yolo_detector and self.yolo_detector.is_available():
            try:
                detections = self.yolo_detector.detect(frame, conf=0.5)
                if detections:
                    self.logger.debug(f"[DETECT] YOLO detected {len(detections)} tomatoes")
                    return detections
            except Exception as e:
                self.logger.warning(f"YOLO detection error: {e}, falling back to ResNet")
        
        # Fallback to ResNet classifier with color-based detection
        if self.classifier:
            # First, detect tomato bounding boxes using color-based detection
            import cv2
            import numpy as np
            
            # Use color-based detection to find tomato bounding boxes
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for tomatoes
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            lower_orange = np.array([10, 50, 50])
            upper_orange = np.array([25, 255, 255])
            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            
            combined_mask = red_mask + green_mask + orange_mask
            
            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours and extract bounding boxes
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.6 < aspect_ratio < 1.6:  # Circular requirement
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3 and w > 40 and h > 40:
                                # Add padding to bounding box for better classification
                                padding = 10
                                x = max(0, x - padding)
                                y = max(0, y - padding)
                                w = min(frame.shape[1] - x, w + 2 * padding)
                                h = min(frame.shape[0] - y, h + 2 * padding)
                                bboxes.append((x, y, w, h))
            
            # Now classify each detected tomato individually
            if bboxes:
                return self.classifier.detect_tomatoes(
                    frame,
                    confidence_threshold=0.3,
                    enhance_for_ugandan=True,
                    bboxes=bboxes
                )
            else:
                # No tomatoes detected, return empty list
                return []
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
        # Reset connection status
        self.arduino_connected = False
        
        # Clean up existing BLE client if reconnecting
        if hasattr(self, 'ble_client') and self.ble_client:
            try:
                self.logger.info("Stopping existing BLE client before reconnection...")
                self.ble_client.disconnect()
                # Give it a moment to clean up
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Error stopping existing BLE client: {e}")
            finally:
                self.ble_client = None
        
        # Connect Arduino via Serial or Bluetooth
        if self.connection_type == 'bluetooth' or (self.connection_type == 'auto' and BLE_AVAILABLE):
            # Try Bluetooth connection
            try:
                if BLE_AVAILABLE:
                    # Create callback to update connection status
                    def on_ble_connect(connected):
                        if connected:
                            self.arduino_connected = True
                            self.logger.info(f"BLE connection established - Arduino status updated")
                        else:
                            if self.connection_type == 'bluetooth' or (self.connection_type == 'auto' and not self.arduino):
                                self.arduino_connected = False
                            self.logger.warning("BLE connection lost")
                    
                    self.logger.info(f"Creating new BLE client for reconnection...")
                    self.ble_client = BLEClient(target_name=self.ble_device_name, on_connect_callback=on_ble_connect)
                    self.ble_client.start(logger=self.logger)
                    time.sleep(3)  # Give it time to scan and connect
                    if self.ble_client.connected:
                        self.arduino_connected = True
                        self.logger.info(f"Arduino connected via Bluetooth: {self.ble_client.device_address}")
                    else:
                        self.logger.warning("Bluetooth device not found, will keep scanning...")
                        if self.connection_type == 'bluetooth':
                            # If bluetooth is required, don't try serial
                            return
                else:
                    self.logger.warning("BLE library not available, trying serial...")
            except Exception as e:
                error_str = str(e)
                # Check for Bluetooth adapter not found error
                if "No Bluetooth adapters found" in error_str or "NO_BLUETOOTH" in error_str:
                    self.logger.error("=" * 60)
                    self.logger.error("❌ BLUETOOTH ADAPTER NOT AVAILABLE")
                    self.logger.error("=" * 60)
                    self.logger.error("Your system does not have a Bluetooth adapter.")
                    self.logger.error("")
                    if self.connection_type == 'bluetooth':
                        self.logger.error("Bluetooth is required but not available.")
                        self.logger.error("Options:")
                        self.logger.error("  1. Use Serial connection instead (set connection_type='serial')")
                        self.logger.error("  2. Install a USB Bluetooth adapter")
                        self.logger.error("  3. Enable Bluetooth in BIOS/UEFI if available")
                        return
                    else:
                        self.logger.error("Falling back to Serial connection...")
                        self.logger.error("(To use Serial, connect Arduino via USB cable)")
                else:
                    self.logger.error(f"Bluetooth connection failed: {e}")
                    if self.connection_type == 'bluetooth':
                        return
        
        # Try Serial connection (if bluetooth failed or not preferred)
        if not self.arduino_connected and self.connection_type != 'bluetooth':
            # Close existing serial connection if reconnecting
            if hasattr(self, 'arduino') and self.arduino:
                try:
                    self.logger.info("Closing existing serial connection before reconnection...")
                    self.arduino.close()
                    time.sleep(0.5)
                except Exception as e:
                    self.logger.warning(f"Error closing existing serial connection: {e}")
                finally:
                    self.arduino = None
            
            try:
                # Try auto-detecting port if default doesn't exist
                port = self.arduino_port
                if not os.path.exists(port):
                    for p in ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']:
                        if os.path.exists(p):
                            port = p
                            break
                
                if os.path.exists(port):
                    self.logger.info(f"Attempting serial connection on {port}...")
                    self.arduino = serial.Serial(port, self.arduino_baud, timeout=1)
                    time.sleep(2) # Wait for reset
                    self.arduino_connected = True
                    self.logger.info(f"Arduino connected on {port}")
                else:
                    self.logger.warning("Arduino not found - Simulation Mode")
            except Exception as e:
                self.logger.error(f"Arduino connection failed: {e}")
                self.arduino_connected = False

        # Connect Camera - detect all available cameras (built-in and USB)
        self.camera_connected = False
        self.available_cameras = self._detect_available_cameras()
        
        # Try to load saved camera preference
        saved_camera_index = self._load_camera_preference()
        
        # Try to connect to the first available camera (or use configured/saved index)
        if self.available_cameras:
            # Priority: saved preference > configured index > first USB camera > first available
            target_index = None
            
            # 1. Try saved preference
            if saved_camera_index is not None and saved_camera_index in self.available_cameras:
                target_index = saved_camera_index
                self.logger.info(f"Using saved camera preference: index {target_index}")
            # 2. Try configured index
            elif self.camera_index in self.available_cameras:
                target_index = self.camera_index
            # 3. Prefer USB cameras (index > 0) over built-in (index 0)
            else:
                usb_cameras = [idx for idx in self.available_cameras if idx > 0]
                if usb_cameras:
                    target_index = usb_cameras[0]
                    self.logger.info(f"Auto-selecting USB camera: index {target_index}")
                else:
                    target_index = self.available_cameras[0]
            
            if self._connect_camera(target_index):
                self.logger.info(f"Camera connected at index {target_index} ({len(self.available_cameras)} cameras available)")
            else:
                self.logger.warning("Failed to connect to any camera")
        else:
            self.logger.warning("No cameras detected - Simulation Mode")
            self.camera = None
    
    def _load_camera_preference(self):
        """Load saved camera preference from file"""
        try:
            import json
            camera_pref_file = Path(self.project_root) / 'camera_preference.json'
            if camera_pref_file.exists():
                with open(camera_pref_file, 'r') as f:
                    pref = json.load(f)
                    return pref.get('camera_index')
        except Exception as e:
            self.logger.debug(f"Could not load camera preference: {e}")
        return None

    def _detect_available_cameras(self):
        """Detect all available cameras (both built-in and USB)"""
        available = []
        self.logger.info("Scanning for available cameras (built-in and USB)...")
        
        # Suppress OpenCV warnings during camera detection
        import os
        import sys
        
        # First, check which /dev/video* devices exist to limit our search
        video_devices = []
        if os.path.exists('/dev'):
            for item in sorted(os.listdir('/dev')):
                if item.startswith('video'):
                    try:
                        # Extract index from /dev/videoN
                        idx = int(item.replace('video', ''))
                        video_devices.append(idx)
                    except ValueError:
                        pass
        
        # Determine indices to test
        if video_devices:
            # If we found video devices, ONLY test those
            # This avoids trying to open non-existent cameras which causes warnings and delays
            indices_to_test = sorted(list(set(video_devices)))
        else:
            # Fallback for non-Linux or if /dev/video* detection failed
            indices_to_test = list(range(10))
        
        # Save original stderr file descriptor
        original_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        
        try:
            # Open /dev/null
            with open(os.devnull, 'w') as devnull:
                # Redirect stderr to /dev/null at the OS level
                os.dup2(devnull.fileno(), original_stderr_fd)
                
                for idx in indices_to_test:
                    test_cap = None
                    try:
                        # Try to open camera with a short timeout
                        # Use V4L2 backend if we know it's a V4L2 device, otherwise ANY
                        backend = cv2.CAP_V4L2 if idx in video_devices else cv2.CAP_ANY
                        test_cap = cv2.VideoCapture(idx, backend)
                        
                        if test_cap is not None and test_cap.isOpened():
                            # Try to read a frame to verify it works
                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                # Get camera info if available
                                try:
                                    backend_name = test_cap.getBackendName()
                                except:
                                    backend_name = "Unknown"
                                
                                try:
                                    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                                    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                                except:
                                    width, height = 640, 480
                                
                                available.append(idx)
                                # We can't log here because stderr is redirected (and logging might use it)
                                # So we'll log after restoring stderr
                            
                            test_cap.release()
                    except Exception:
                        if test_cap:
                            try:
                                test_cap.release()
                            except:
                                pass
                        continue
        finally:
            # Restore stderr
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
        
        # Log results now that stderr is restored
        for idx in available:
            self.logger.info(f"Found camera at index {idx}: V4L2, 640x480") # Simplified log
            
        if available:
            self.logger.info(f"Detected {len(available)} camera(s): {available}")
        else:
            self.logger.warning("No cameras detected")
        
        return available
    
    def _connect_camera(self, camera_index):
        """Connect to a specific camera index"""
        try:
            # Release existing camera if any
            if self.camera:
                try:
                    self.camera.release()
                    # Give the camera a moment to fully release
                    time.sleep(0.2)
                except:
                    pass
                self.camera = None
                self.camera_connected = False
            
            # Suppress OpenCV warnings during connection
            import os
            import sys
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            
            # Check if this is a /dev/video* device
            video_devices = []
            if os.path.exists('/dev'):
                for item in sorted(os.listdir('/dev')):
                    if item.startswith('video'):
                        try:
                            idx = int(item.replace('video', ''))
                            video_devices.append(idx)
                        except ValueError:
                            pass
            
            # Use V4L2 backend if it's a /dev/video* device
            backend = cv2.CAP_V4L2 if camera_index in video_devices else cv2.CAP_ANY
            
            # Try multiple backends if V4L2 fails
            self.camera = None
            backends_to_try = []
            if camera_index in video_devices:
                backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
            else:
                backends_to_try = [cv2.CAP_ANY, cv2.CAP_V4L2]
            
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                sys.stdout = devnull
                for backend_to_try in backends_to_try:
                    try:
                        self.camera = cv2.VideoCapture(camera_index, backend_to_try)
                        if self.camera and self.camera.isOpened():
                            break
                        elif self.camera:
                            self.camera.release()
                            self.camera = None
                    except Exception as e:
                        self.logger.debug(f"Backend {backend_to_try} failed for camera {camera_index}: {e}")
                        if self.camera:
                            try:
                                self.camera.release()
                            except:
                                pass
                            self.camera = None
                        continue
                sys.stderr = original_stderr
                sys.stdout = original_stdout
                
                if self.camera and self.camera.isOpened():
                    # Test if we can actually read a frame (try multiple times)
                    ret = False
                    frame = None
                    for attempt in range(3):
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            break
                        time.sleep(0.1)  # Wait a bit between attempts
                    
                    if ret and frame is not None:
                        self.camera_index = camera_index
                        self.camera_connected = True
                        # Start frame reading thread
                        self.frame_update_running = True
                        self.frame_update_thread = threading.Thread(target=self._update_frame, daemon=True)
                        self.frame_update_thread.start()
                        self.logger.info(f"Camera {camera_index} connected successfully")
                        return True
                    else:
                        # Camera opened but can't read frames
                        self.logger.warning(f"Camera {camera_index} opened but cannot read frames")
                        if self.camera:
                            try:
                                self.camera.release()
                            except:
                                pass
                        self.camera = None
                        return False
                else:
                    # Camera didn't open
                    self.logger.warning(f"Camera {camera_index} failed to open")
                    if self.camera:
                        try:
                            self.camera.release()
                        except:
                            pass
                    self.camera = None
                    return False
        except Exception as e:
            self.logger.error(f"Failed to connect camera {camera_index}: {e}")
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            return False
    
    def switch_camera(self, camera_index):
        """Switch to a different camera"""
        # Convert to int if it's a string
        try:
            camera_index = int(camera_index)
        except (ValueError, TypeError):
            self.logger.error(f"Invalid camera index: {camera_index}")
            return False
        
        # Check if camera is in available list
        if not hasattr(self, 'available_cameras') or not self.available_cameras:
            self.logger.warning("Available cameras list not initialized, attempting to detect...")
            self.available_cameras = self._detect_available_cameras()
        
        if camera_index not in self.available_cameras:
            self.logger.error(f"Camera index {camera_index} not available. Available: {self.available_cameras}")
            # Try to detect again in case a new camera was connected
            self.logger.info("Re-detecting cameras...")
            self.available_cameras = self._detect_available_cameras()
            if camera_index not in self.available_cameras:
                self.logger.error(f"Camera index {camera_index} still not available after re-detection. Available: {self.available_cameras}")
                return False
            else:
                self.logger.info(f"Camera {camera_index} found after re-detection")
        
        # Don't switch if we're already on this camera
        if self.camera_connected and self.camera_index == camera_index:
            self.logger.debug(f"Already on camera {camera_index}, skipping switch")
            return True
        
        old_index = self.camera_index
        self.logger.info(f"Attempting to switch from camera {old_index} to camera {camera_index}")
        
        if self._connect_camera(camera_index):
            self.logger.info(f"Successfully switched from camera {old_index} to camera {camera_index}")
            return True
        else:
            self.logger.error(f"Failed to switch to camera {camera_index} - connection failed")
            return False
    
    def get_available_cameras(self):
        """Get list of available cameras with details"""
        cameras = []
        import os
        import sys
        original_stderr = sys.stderr
        
        for idx in self.available_cameras:
            try:
                # Suppress OpenCV warnings
                with open(os.devnull, 'w') as devnull:
                    sys.stderr = devnull
                    test_cap = cv2.VideoCapture(idx)
                    sys.stderr = original_stderr
                    
                    if test_cap.isOpened():
                        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        backend = test_cap.getBackendName()
                        
                        # Try to identify camera type
                        camera_name = f"Camera {idx}"
                        camera_type = "Unknown"
                        
                        # Check if it's likely a USB camera (index > 0 usually means USB)
                        if idx == 0:
                            camera_name = "Built-in Camera"
                            camera_type = "Built-in"
                        else:
                            # Try to get more info about USB cameras
                            # Check if there's a /dev/video* device
                            video_devices = []
                            if os.path.exists('/dev'):
                                for item in os.listdir('/dev'):
                                    if item.startswith('video'):
                                        try:
                                            video_idx = int(item.replace('video', ''))
                                            if video_idx == idx:
                                                camera_name = f"USB Camera ({item})"
                                                camera_type = "USB"
                                                break
                                        except:
                                            pass
                            
                            if camera_type == "Unknown":
                                camera_name = f"USB Camera {idx}"
                                camera_type = "USB"
                        
                        cameras.append({
                            'index': idx,
                            'name': camera_name,
                            'type': camera_type,
                            'backend': backend,
                            'resolution': f"{width}x{height}",
                            'current': idx == self.camera_index
                        })
                    test_cap.release()
            except:
                sys.stderr = original_stderr
                pass
        
        sys.stderr = original_stderr
        return cameras

    def _update_frame(self):
        """Background thread to keep reading frames"""
        while self.camera_connected and self.frame_update_running:
            if not self.camera or not self.camera.isOpened():
                break
            try:
                ret, frame = self.camera.read()
                if ret:
                    with self.camera_lock:
                        self.last_frame = frame
            except Exception as e:
                self.logger.error(f"Error reading frame: {e}")
                break
            time.sleep(0.03) # ~30 FPS
        self.logger.debug("Frame update thread stopped")
    
    def stop_camera_feed(self):
        """Stop camera feed and frame update thread"""
        self.logger.info("Stopping camera feed...")
        self.frame_update_running = False
        self.camera_connected = False
        
        # Wait a moment for thread to stop
        if self.frame_update_thread and self.frame_update_thread.is_alive():
            time.sleep(0.1)
        
        # Note: We don't release the camera here because it might be needed by other components
        # The camera will be released when switching cameras or on shutdown

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

    def update_servo_angle(self, servo_name, angle):
        """Update tracked servo angle"""
        # Map backend servo names to our tracking names
        servo_map = {
            'base': 'base',
            'arm': 'shoulder',  # Backend 'arm' is frontend 'shoulder'
            'shoulder': 'shoulder',
            'forearm': 'forearm',
            'wrist_yaw': 'elbow',  # Backend 'wrist_yaw' is frontend 'elbow'
            'elbow': 'elbow',
            'wrist_pitch': 'pitch',  # Backend 'wrist_pitch' is frontend 'pitch'
            'pitch': 'pitch',
            'claw': 'claw'
        }
        
        tracked_name = servo_map.get(servo_name.lower(), servo_name.lower())
        if tracked_name in self.current_servo_angles:
            self.current_servo_angles[tracked_name] = int(angle)
            self.logger.debug(f"Updated {tracked_name} angle to {angle}°")

    def filter_servo_command(self, command):
        """Filter servo commands to skip unavailable servos
        
        Args:
            command: Command string (e.g., "ANGLE 90 90 90 90 90 0")
        
        Returns:
            str: Filtered command with unavailable servos set to -1 (no change)
        """
        if not command.startswith("ANGLE"):
            return command  # Not a servo command, return as-is
        
        try:
            # Parse ANGLE command: "ANGLE base shoulder forearm elbow pitch claw"
            parts = command.split()
            if len(parts) != 7:  # "ANGLE" + 6 servo values
                return command
            
            angles = [int(parts[i]) if parts[i] != '-1' else -1 for i in range(1, 7)]
            
            # Map servo indices to names
            servo_map = ['base', 'shoulder', 'forearm', 'elbow', 'pitch', 'claw']
            
            # Replace unavailable servos with -1 (no change) or use fixed angle
            for i, servo_name in enumerate(servo_map):
                if not self.servo_available.get(servo_name, True):
                    # Use -1 to keep current position (servo is manually fixed)
                    # The fixed angle is just for tracking, not sent to Arduino
                    angles[i] = -1
                    # Update our tracking with fixed angle
                    if servo_name in self.fixed_servo_angles:
                        self.current_servo_angles[servo_name] = self.fixed_servo_angles[servo_name]
            
            # Reconstruct command
            filtered_command = f"ANGLE {' '.join(str(a) for a in angles)}"
            return filtered_command
            
        except Exception as e:
            self.logger.warning(f"Error filtering servo command: {e}, using original")
            return command

    def send_command(self, command):
        """Send G-code style command to Arduino (via Serial or Bluetooth)"""
        # Filter servo commands to skip unavailable servos
        if command.startswith("ANGLE"):
            command = self.filter_servo_command(command)
        
        # Parse ANGLE commands to track servo angles
        if command.startswith("ANGLE"):
            try:
                parts = command.split()
                if len(parts) >= 7:  # ANGLE base shoulder forearm elbow pitch claw
                    # Update tracked angles (skip -1 values which mean "keep current")
                    servo_names = ['base', 'shoulder', 'forearm', 'elbow', 'pitch', 'claw']
                    for i, angle_str in enumerate(parts[1:7]):
                        if angle_str != '-1' and i < len(servo_names):
                            try:
                                angle = int(angle_str)
                                self.update_servo_angle(servo_names[i], angle)
                            except ValueError:
                                pass
            except Exception as e:
                self.logger.debug(f"Could not parse ANGLE command for tracking: {e}")
        
        # Check BLE connection status dynamically
        ble_connected = False
        if self.ble_client:
            ble_connected = self.ble_client.connected
            # Update arduino_connected if BLE just connected
            if ble_connected and not self.arduino_connected:
                self.arduino_connected = True
                self.logger.info("BLE connection detected during command send")
        
        if self.arduino_connected or ble_connected:
            # Try Bluetooth first
            if self.ble_client and ble_connected:
                if self.ble_client.send_command(command):
                    self.logger.info(f"Sent to Arduino (BLE): {command}")
                    return True
                else:
                    self.logger.warning("BLE send failed - command not queued")
                    return False
            
            # Try Serial
            if self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
                try:
                    self.arduino.write(f"{command}\n".encode())
                    self.logger.info(f"Sent to Arduino (Serial): {command}")
                    return True
                except Exception as e:
                    self.logger.error(f"Serial write error: {e}")
                    return False
            else:
                self.logger.warning(f"Arduino not connected - cannot send command: {command}")
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
    
    def handle_pick_result(self, pick_id, status, result, duration_ms=0):
        """Handle pick result from Arduino
        
        Args:
            pick_id: Unique pick identifier
            status: 'SUCCESS', 'FAILED', 'ABORTED', etc.
            result: Result details (e.g., 'ripe', 'unripe', 'none')
            duration_ms: Pick operation duration in milliseconds
        """
        if pick_id in self.pending_picks:
            pick_info = self.pending_picks[pick_id]
            if status == 'SUCCESS':
                self.logger.info(f"[AUTO] ✅ Pick {pick_id} succeeded: {result} ({duration_ms}ms)")
                del self.pending_picks[pick_id]
            elif status == 'FAILED' or status == 'ABORTED':
                pick_info['retries'] += 1
                if pick_info['retries'] < self.max_pick_retries:
                    self.logger.warning(f"[AUTO] ⚠️  Pick {pick_id} failed ({status}), will retry (attempt {pick_info['retries']}/{self.max_pick_retries})")
                    # Could retry here if needed
                else:
                    self.logger.error(f"[AUTO] ❌ Pick {pick_id} failed after {pick_info['retries']} attempts: {status}")
                    del self.pending_picks[pick_id]
            else:
                self.logger.warning(f"[AUTO] Unknown pick status: {status} for {pick_id}")
        else:
            self.logger.debug(f"[AUTO] Received pick result for unknown pick_id: {pick_id}")
    
    def cleanup_old_picks(self, timeout_seconds=30):
        """Remove old pending picks that haven't completed
        
        Args:
            timeout_seconds: Time after which a pick is considered stale
        """
        current_time = time.time()
        stale_picks = []
        
        for pick_id, pick_info in self.pending_picks.items():
            age = current_time - pick_info['timestamp']
            if age > timeout_seconds:
                stale_picks.append(pick_id)
        
        for pick_id in stale_picks:
            self.logger.warning(f"[AUTO] Removing stale pick {pick_id} (age: {current_time - self.pending_picks[pick_id]['timestamp']:.1f}s)")
            del self.pending_picks[pick_id]
    
    # Conveyor belt not available in current setup
    # def set_conveyor(self, speed):
    #     """Control conveyor speed (0-100)"""
    #     return self.send_command(f"CONVEYOR {speed}")

    def load_calibration_matrix(self):
        """Load homography matrix from calibration file
        
        Tries multiple locations:
        1. calibration.npz (from update_calibration)
        2. calibration_data.json (from web interface)
        3. homography.npy (from calibrate_homography.py)
        
        Returns:
            bool: True if calibration loaded successfully
        """
        try:
            # Try calibration.npz first
            calib_file = self.project_root / 'calibration.npz'
            if calib_file.exists():
                data = np.load(str(calib_file))
                if 'homography' in data:
                    self.homography_matrix = data['homography']
                    self.logger.info("✅ Loaded calibration from calibration.npz")
                    return True
            
            # Try calibration_data.json (from web interface)
            calib_json = self.project_root / 'calibration_data.json'
            if calib_json.exists():
                import json
                with open(calib_json, 'r') as f:
                    data = json.load(f)
                    # Check for homography directly or in calibration.matrix
                    if 'homography' in data:
                        self.homography_matrix = np.array(data['homography'])
                        self.logger.info("✅ Loaded calibration from calibration_data.json")
                        return True
                    elif 'calibration' in data and isinstance(data['calibration'], dict):
                        if 'matrix' in data['calibration']:
                            self.homography_matrix = np.array(data['calibration']['matrix'])
                            self.logger.info("✅ Loaded calibration from calibration_data.json (matrix)")
                            return True
            
            # Try homography.npy (from calibrate_homography.py)
            homography_file = self.project_root / 'homography.npy'
            if homography_file.exists():
                self.homography_matrix = np.load(str(homography_file))
                self.logger.info("✅ Loaded calibration from homography.npy")
                return True
            
            self.logger.warning("⚠️  No calibration file found. Using fallback coordinate mapping.")
            self.logger.warning("   Please calibrate the system for accurate automatic picking.")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
    
    def pixel_to_arm_coordinates(self, pixel_x, pixel_y):
        """Convert pixel coordinates to arm coordinates (millimeters)
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
        
        Returns:
            tuple: (arm_x, arm_y) in millimeters, or None if conversion failed
        """
        try:
            if self.homography_matrix is not None:
                # Use homography transformation
                pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
                arm_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
                arm_x = float(arm_point[0][0][0])
                arm_y = float(arm_point[0][0][1])
                return arm_x, arm_y
            else:
                # Fallback: simple scaling (needs calibration!)
                # Assume camera is 640x480, workspace is roughly 300x200mm
                # This is a rough estimate and should be replaced with proper calibration
                scale_x = 300.0 / 640.0  # mm per pixel
                scale_y = 200.0 / 480.0  # mm per pixel
                arm_x = pixel_x * scale_x - 150  # Center at 0
                arm_y = pixel_y * scale_y + 50   # Offset from base
                self.logger.debug(f"Using fallback scaling: ({pixel_x}, {pixel_y}) -> ({arm_x:.1f}, {arm_y:.1f})")
                return arm_x, arm_y
                
        except Exception as e:
            self.logger.error(f"Coordinate conversion failed: {e}")
            return None
    
    def is_position_reachable(self, arm_x, arm_y, arm_z):
        """Check if position is within arm workspace bounds
        
        Args:
            arm_x: X coordinate in mm
            arm_y: Y coordinate in mm
            arm_z: Z coordinate (depth) in mm
        
        Returns:
            bool: True if position is reachable
        """
        bounds = self.workspace_bounds
        in_bounds = (
            bounds['x_min'] <= arm_x <= bounds['x_max'] and
            bounds['y_min'] <= arm_y <= bounds['y_max'] and
            bounds['z_min'] <= arm_z <= bounds['z_max']
        )
        
        if not in_bounds:
            self.logger.warning(
                f"Position ({arm_x:.1f}, {arm_y:.1f}, {arm_z:.1f}) out of bounds: "
                f"X[{bounds['x_min']}, {bounds['x_max']}], "
                f"Y[{bounds['y_min']}, {bounds['y_max']}], "
                f"Z[{bounds['z_min']}, {bounds['z_max']}]"
            )
        
        return in_bounds
    
    def calculate_tomato_depth(self, target, tof_distance):
        """Calculate accurate depth for tomato picking
        
        IMPORTANT: ToF sensor is on the claw, so it moves with the arm!
        When arm is positioned above tomato, ToF reads distance from claw to tomato.
        
        Args:
            target: Detection dict with 'bbox' and 'center'
            tof_distance: Distance from ToF sensor (on claw) to tomato surface (mm)
                          This is read when arm is at approach position above tomato
        
        Returns:
            float: Calculated depth in mm (distance from claw to tomato center)
        """
        try:
            # Since ToF is on the claw and moves with the arm:
            # - When arm approaches tomato, ToF reads distance to tomato surface
            # - We need to account for tomato radius to get to center
            # - Or use ToF distance directly if we want to pick from surface
            
            # Get bounding box to estimate tomato size
            bbox = target.get('bbox', [0, 0, 50, 50])  # [x, y, w, h]
            tomato_width = max(bbox[2], bbox[3])  # Use larger dimension
            
            # Estimate tomato radius from pixel size
            # Typical tomato: 40-60mm diameter (20-30mm radius)
            if tomato_width > 0:
                # Rough estimate: 50mm tomato = ~100 pixels at reference distance
                pixel_to_mm_ratio = 50.0 / 100.0
                estimated_tomato_diameter = tomato_width * pixel_to_mm_ratio
                estimated_tomato_radius = estimated_tomato_diameter / 2.0
                # Clamp to reasonable range
                estimated_tomato_radius = max(15.0, min(35.0, estimated_tomato_radius))
            else:
                # Default tomato radius
                estimated_tomato_radius = 25.0  # 25mm radius = 50mm diameter
            
            # Depth calculation for ToF on claw:
            # Option 1: Use ToF distance directly (pick from surface)
            # Option 2: Subtract tomato radius (pick from center) - more accurate
            if tof_distance is not None and tof_distance > 0:
                # ToF reads distance to tomato surface
                # Subtract radius to get to center (more accurate picking)
                depth = tof_distance - estimated_tomato_radius
                # But ensure we don't go negative (ToF might already be close)
                depth = max(10.0, depth)  # Minimum 10mm clearance
            else:
                # Fallback: use default depth
                depth = 50.0  # Default 50mm above surface
            
            # Clamp to reasonable range
            depth = max(10.0, min(150.0, depth))
            
            self.logger.debug(f"ToF on claw: distance={tof_distance}mm, tomato_radius={estimated_tomato_radius:.1f}mm, depth={depth:.1f}mm")
            
            return depth
            
        except Exception as e:
            self.logger.warning(f"Depth calculation error: {e}, using default")
            return 50.0  # Safe default

    def update_calibration(self, points):
        """Update calibration data for coordinate mapping
        
        Args:
            points: List of calibration points with pixel and world coordinates
                    [{'pixel': [x, y], 'world': [x, y]}, ...]
        
        Returns:
            bool: True if calibration successful
        """
        try:
            if len(points) < 4:
                self.logger.error("Need at least 4 calibration points")
                return False
            
            # Extract pixel and world coordinates
            pixel_coords = np.array([p['pixel'] for p in points], dtype=np.float32)
            world_coords = np.array([p['world'] for p in points], dtype=np.float32)
            
            # Compute homography matrix
            homography_matrix, status = cv2.findHomography(pixel_coords, world_coords)
            
            if homography_matrix is None:
                self.logger.error("Failed to compute homography")
                return False
            
            # Save calibration
            self.homography_matrix = homography_matrix
            calibration_file = self.project_root / 'calibration.npz'
            np.savez(str(calibration_file), 
                     homography=homography_matrix,
                     pixel_coords=pixel_coords,
                     world_coords=world_coords)
            
            self.logger.info(f"✅ Calibration saved with {len(points)} points")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration update failed: {e}")
            return False

    def get_status(self):
        """Get hardware status"""
        # Check current BLE connection status dynamically
        ble_connected = False
        if self.ble_client:
            ble_connected = self.ble_client.connected
            # Update arduino_connected based on actual BLE status
            if ble_connected and not self.arduino_connected:
                self.arduino_connected = True
                self.logger.info("BLE connection detected - updating status")
            elif not ble_connected and self.arduino_connected and not self.arduino:
                # Only set to False if we're using BLE and it's disconnected
                if self.connection_type == 'bluetooth' or (self.connection_type == 'auto' and self.ble_client):
                    self.arduino_connected = False
        
        # Also check if BLE client exists but connection might be in progress
        # If we have a BLE client, it means we're trying to use Bluetooth
        if self.ble_client and not ble_connected:
            # Check if client is still trying to connect (thread is alive)
            if self.ble_client.thread and self.ble_client.thread.is_alive():
                self.logger.debug("BLE client thread is alive, connection may be in progress")
        
        # Determine connection type based on current state
        if ble_connected:
            connection_type = 'bluetooth'
        elif self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
            connection_type = 'serial'
        else:
            connection_type = 'none'
        
        # If we have a BLE client but it's not connected yet, still report connection_type as bluetooth
        # if we're configured for Bluetooth
        if self.ble_client and connection_type == 'none' and (self.connection_type == 'bluetooth' or self.connection_type == 'auto'):
            connection_type = 'bluetooth'  # Indicate we're trying Bluetooth
        
        # Calculate arduino_connected - ensure it's always a boolean
        arduino_conn = False
        if ble_connected:
            arduino_conn = True
        elif self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
            arduino_conn = True
        elif self.arduino_connected:
            arduino_conn = True
        
        # Determine arm orientation
        is_facing_front = self.is_arm_facing_front()
        
        # Check if any AI model is loaded (YOLO or ResNet classifier)
        model_loaded = False
        if self.yolo_detector and hasattr(self.yolo_detector, 'is_available') and self.yolo_detector.is_available():
            model_loaded = True
        elif self.classifier is not None:
            model_loaded = True
        
        status = {
            'arduino_connected': arduino_conn,
            'camera_connected': self.camera_connected,
            'camera_index': self.camera_index if self.camera_connected else None,
            'available_cameras': len(self.available_cameras) if hasattr(self, 'available_cameras') else 0,
            'auto_mode': self.auto_mode,
            'connection_type': connection_type,
            'classifier_loaded': model_loaded,
            'arm_orientation': 'front' if is_facing_front else 'back',
            'servo_angles': self.current_servo_angles.copy()
        }
        
        if self.ble_client:
            status['ble_connected'] = ble_connected
            status['ble_address'] = self.ble_client.device_address
            status['ble_client_active'] = self.ble_client.thread.is_alive() if self.ble_client.thread else False
            # Log connection attempt status for debugging
            if not ble_connected and self.ble_client.thread and self.ble_client.thread.is_alive():
                self.logger.debug(f"BLE client active but not connected. Device: {self.ble_client.device_address}")
        
        return status
    
    def scan_bluetooth_devices(self, timeout=10):
        """Scan for Bluetooth devices (async helper)"""
        if not BLE_AVAILABLE:
            return []
        
        async def _scan():
            devices = []
            try:
                scanner = BleakScanner()
                found_devices = await asyncio.wait_for(scanner.discover(timeout=timeout), timeout=timeout+2)
                for device in found_devices:
                    devices.append({
                        'name': device.name or 'Unknown',
                        'address': device.address,
                        'rssi': getattr(device, 'rssi', None)
                    })
            except Exception as e:
                self.logger.error(f"BLE scan error: {e}")
            return devices
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            devices = loop.run_until_complete(_scan())
        finally:
            loop.close()
        return devices
    
    def connect_bluetooth_device(self, address, name=None):
        """Connect to a specific Bluetooth device by address"""
        if not BLE_AVAILABLE:
            return False
        
        # Disconnect existing connection
        if self.ble_client:
            self.ble_client.disconnect()
        
        try:
            # Create callback to update connection status
            def on_ble_connect(connected):
                if connected:
                    self.arduino_connected = True
                    self.logger.info(f"BLE connection established via connect_bluetooth_device")
                else:
                    if self.connection_type == 'bluetooth' or (self.connection_type == 'auto' and not self.arduino):
                        self.arduino_connected = False
            
            self.ble_client = BLEClient(target_name=name or "Arduino", on_connect_callback=on_ble_connect)
            self.ble_client.device_address = address
            self.ble_client.start(logger=self.logger)
            time.sleep(3)
            if self.ble_client.connected:
                self.arduino_connected = True
                return True
        except Exception as e:
            self.logger.error(f"Bluetooth connection failed: {e}")
        return False
