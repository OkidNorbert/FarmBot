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

# Try to import BLE libraries
try:
    import asyncio
    from bleak import BleakScanner, BleakClient
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    print("Warning: Bleak library not available. Bluetooth support disabled.")

try:
    from models.tomato.tomato_inference import TomatoClassifier
except ImportError:
    print("Warning: Could not import TomatoClassifier. Using dummy.")
    TomatoClassifier = None

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
        """Disconnect from BLE device"""
        self.connected = False
        if self.client:
            try:
                self.loop.call_soon_threadsafe(self.client.disconnect)
            except:
                pass

class HardwareController:
    def __init__(self, connection_type='auto', ble_device_name="Arduino"):
        """Initialize Hardware Controller
        
        Args:
            connection_type: 'serial', 'bluetooth', or 'auto' (tries serial first, then bluetooth)
            ble_device_name: Name of BLE device to connect to
        """
        self.setup_logging()
        
        # Components
        self.arduino = None
        self.ble_client = None
        self.camera = None
        self.classifier = None
        self.camera_lock = threading.Lock()
        
        # State
        self.arduino_connected = False
        self.camera_connected = False
        self.auto_mode = False
        self.last_frame = None
        self.connection_type = connection_type
        self.ble_device_name = ble_device_name
        
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
                self.logger.error(f"Bluetooth connection failed: {e}")
                if self.connection_type == 'bluetooth':
                    return
        
        # Try Serial connection (if bluetooth failed or not preferred)
        if not self.arduino_connected and self.connection_type != 'bluetooth':
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

        # Connect Camera - try multiple indices
        self.camera_connected = False
        for camera_idx in range(5):  # Try indices 0-4
            try:
                self.camera = cv2.VideoCapture(camera_idx)
                if self.camera.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.camera_index = camera_idx
                        self.camera_connected = True
                        self.logger.info(f"Camera connected at index {camera_idx}")
                        # Start frame reading thread
                        threading.Thread(target=self._update_frame, daemon=True).start()
                        break
                    else:
                        self.camera.release()
                        self.logger.debug(f"Camera {camera_idx} opened but cannot read frames")
                else:
                    self.logger.debug(f"Camera {camera_idx} not available")
            except Exception as e:
                self.logger.debug(f"Camera {camera_idx} error: {e}")
                if self.camera:
                    try:
                        self.camera.release()
                    except:
                        pass
        
        if not self.camera_connected:
            self.logger.warning("No working camera found - Simulation Mode")
            self.camera = None

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
        """Send G-code style command to Arduino (via Serial or Bluetooth)"""
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
    
    # Conveyor belt not available in current setup
    # def set_conveyor(self, speed):
    #     """Control conveyor speed (0-100)"""
    #     return self.send_command(f"CONVEYOR {speed}")

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
            calibration_file = 'calibration.npz'
            np.savez(calibration_file, 
                     homography=homography_matrix,
                     pixel_coords=pixel_coords,
                     world_coords=world_coords)
            
            self.logger.info(f"Calibration saved with {len(points)} points")
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
        
        status = {
            'arduino_connected': arduino_conn,
            'camera_connected': self.camera_connected,
            'auto_mode': self.auto_mode,
            'connection_type': connection_type,
            'classifier_loaded': self.classifier is not None
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
