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

# Suppress OpenCV warnings globally
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# Note: cv2.setLogLevel() may not be available in all OpenCV versions
try:
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(3)  # 3 = LOG_LEVEL_SILENT in newer versions
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
        
        # Get project root directory (where this file is located)
        self.project_root = Path(__file__).parent.absolute()
        
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
        self.frame_update_thread = None
        self.frame_update_running = False
        
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
        if not TOMATO_CLASSIFIER_AVAILABLE or not TomatoClassifier:
            self.logger.warning("TomatoClassifier not available. Model detection disabled.")
            self.logger.warning("Make sure PyTorch and the model files are installed.")
            return
        
        try:
            # Use absolute path from project root
            model_path = self.project_root / "models" / "tomato" / "best_model.pth"
            model_path_str = str(model_path)
            
            self.logger.info(f"Looking for model at: {model_path_str}")
            
            if model_path.exists():
                self.logger.info(f"Model file found: {model_path_str}")
                try:
                    self.classifier = TomatoClassifier(model_path_str)
                    self.logger.info("✅ AI Model Loaded successfully")
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

    def process_auto_cycle(self):
        """Single cycle of autonomous logic - Only picks 'ready' tomatoes"""
        frame = self.get_frame()
        if frame is None:
            return

        # 1. Detect
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
        
        # 3. Decide: Pick the most confident ready tomato
        target = max(ready_tomatoes, key=lambda x: x['confidence'])
        self.logger.info(f"[AUTO] Picking ready tomato (confidence: {target['confidence']:.2f}, class: {target['class']})")
        
        # 4. Act (if connected)
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
            
            self.logger.info(f"[AUTO] Picking ready tomato at {center_x}, {center_y}, {z_depth}mm")
            
            # Send Pick Command
            # Format: PICK <x> <y> <z> <class_id>
            # class_id: 1=Ready/Ripe (goes to right), 0=Other (goes to left)
            # Since we only pick ready tomatoes, always use class_id=1 (right bin)
            class_id = 1  # Ready tomatoes go to the right
            self.send_command(f"PICK {center_x} {center_y} {z_depth} {class_id}")
            
            # Wait for operation to complete
            time.sleep(5) 
        else:
            self.logger.info("[AUTO] Simulation: Pick command sent")

    def detect_tomatoes(self, frame):
        """Run detection on frame with enhanced support for Ugandan tomatoes"""
        if self.classifier:
            # Use lower confidence threshold for Ugandan tomatoes (0.3 instead of default 0.5)
            # Enable image enhancement optimized for Ugandan tomato varieties
            return self.classifier.detect_tomatoes(
                frame, 
                confidence_threshold=0.3,  # Lower threshold for better detection
                enhance_for_ugandan=True   # Apply color/brightness enhancement
            )
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
            'camera_index': self.camera_index if self.camera_connected else None,
            'available_cameras': len(self.available_cameras) if hasattr(self, 'available_cameras') else 0,
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
