#!/usr/bin/env python3
"""
Hardware Controller for AI Tomato Sorter
Manages connections to Arduino (Robotic Arm) and Camera
"""

import cv2
import numpy as np
import serial
import time
import math
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
    def __init__(self, service_uuid=None, char_uuid=None, telemetry_uuid=None, target_name="Arduino", on_connect_callback=None, on_data_callback=None):
        if not BLE_AVAILABLE:
            raise ImportError("Bleak library not available")
        
        self.service_uuid = service_uuid or '19B10000-E8F2-537E-4F6C-D104768A1214'
        self.char_uuid = char_uuid or '19B10001-E8F2-537E-4F6C-D104768A1214'
        self.telemetry_uuid = telemetry_uuid or '19B10002-E8F2-537E-4F6C-D104768A1214'
        self.target_name = target_name
        self.client = None
        self.loop = None
        self.thread = None
        self.connected = False
        self._ble_running = False  # Controls the _main_loop; set True in start(), False in disconnect()
        self.command_queue = asyncio.Queue()
        self.device_address = None
        self.on_connect_callback = on_connect_callback  # Callback when connection state changes
        self.on_data_callback = on_data_callback        # Callback for incoming telemetry
        self.logger = logging.getLogger("BLEClient")
        
    def _run_loop(self):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_loop())
    
    async def _main_loop(self):
        """Main BLE connection loop with exponential backoff to avoid log spam"""
        retry_count = 0
        while self._ble_running:
            if not self.connected:
                await self._connect()
                if not self.connected:
                    # Exponential backoff: 5s, 10s, 20s, cap at 60s
                    wait = min(5 * (2 ** min(retry_count, 3)), 60)
                    retry_count += 1
                    await asyncio.sleep(wait)
                else:
                    retry_count = 0  # reset on successful connect
            else:
                retry_count = 0
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
                
                def disconnected_callback(client):
                    self.logger.warning("BLE Client disconnected callback triggered")
                    self.connected = False
                    if self.on_connect_callback:
                        self.on_connect_callback(False)

                self.client = BleakClient(device, disconnected_callback=disconnected_callback)
                try:
                    await self.client.connect()
                    
                    # Start notification for telemetry characteristic
                    try:
                        await self.client.start_notify(self.telemetry_uuid, self._on_telemetry_notification)
                        self.logger.info(f"Subscribed to BLE telemetry: {self.telemetry_uuid}")
                    except Exception as e:
                        self.logger.warning(f"Failed to subscribe to telemetry: {e}")

                    self.connected = True
                    self.logger.info("BLE Connected!")
                    
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
    
    def _on_telemetry_notification(self, characteristic, data):
        """Handle incoming telemetry notification"""
        try:
            payload = data.decode('utf-8')
            if self.on_data_callback:
                self.on_data_callback(payload)
        except Exception as e:
            self.logger.error(f"Error decoding BLE telemetry: {e}")

    async def _send_command(self, command):
        """Send command via BLE"""
        try:
            # Send command with newline
            command_bytes = f"{command}\n".encode('utf-8')
            await self.client.write_gatt_char(self.char_uuid, command_bytes)
            self.logger.info(f"Sent BLE command: {command}")
        except Exception as e:
            self.logger.error(f"BLE write error: {e}")
            self.connected = False
            try:
                if self.client:
                    await self.client.disconnect()
            except:
                pass
            if self.on_connect_callback:
                self.on_connect_callback(False)
    
    def start(self, logger=None):
        """Start BLE client in background thread"""
        self.logger = logger or logging.getLogger("BLEClient")
        self._ble_running = True
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
        self.serial_lock = threading.Lock()
        
        # State
        self.arduino_connected = False
        self.camera_connected = False
        self.auto_mode = False
        self.movement_active = False # Track if a pre-defined movement is playing
        self.last_frame = None
        self.connection_type = connection_type
        self.ble_device_name = ble_device_name
        self.frame_update_thread = None
        self.frame_update_running = False
        self.tof_initialized = False # Track if ToF sensor is ready on Arduino
        self.camera_on_arm = False   # Set to True if camera moves with the arm
        
        # Servo availability configuration
        # Set to False for servos that are not available (manually fixed)
        self.servo_available = {
            'waist': True,       # Enabled for diagnostic and control
            'shoulder': True,   # Shoulder servo now available
            'elbow': True,      # Elbow servo available
            'wrist_roll': True, # Wrist roll available
            'wrist_pitch': True,# Wrist pitch available
            'claw': True        # Claw servo available
        }
        # Strict safe endpoints matching physical robot body limits
        self.servo_limits = {
            'waist': {'min': 0, 'max': 180},
            'shoulder': {'min': 15, 'max': 165},
            'elbow': {'min': 10, 'max': 165},
            'wrist_roll': {'min': 15, 'max': 165},
            'wrist_pitch': {'min': 20, 'max': 160},
            'claw': {'min': 30, 'max': 110}
        }
        
        # Fixed positions for unavailable servos (manually set)
        # These should match your manual adjustments
        self.fixed_servo_angles = {
            'waist': 90        # Manually fixed waist position
        }
        
        # Track current servo angles (for front/back detection)
        # Servo indices: 0=base, 1=shoulder/arm, 2=forearm, 3=elbow/wrist_yaw, 4=pitch, 5=claw
        self.current_servo_angles = {
            'waist': self.fixed_servo_angles['waist'],
            'shoulder': 90,
            'elbow': 90,
            'wrist_roll': 90,
            'wrist_pitch': 90,
            'claw': 110  # home position is fully closed
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
            'x_min': -40, 'x_max': 40,  # mm
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

        # Fixed Pickup Zone State Machine parameters
        self.auto_state = 'IDLE' # IDLE, DETECTING, TARGET_CANDIDATE, TARGET_LOCKED, HARVESTING, COOLDOWN, ERROR
        self.target_candidate = None
        self.candidate_frames = 0
        self.candidate_required_frames = 3
        self.cooldown_end_time = 0
        self.cooldown_duration_s = 5.0
        
        # Pickup Zone Config (relative bounds — center of frame aligned to physical pickup point)
        # x: covers centre 50% of frame width (0.25-0.75) so slight off-centre tomatoes aren't rejected
        # y: lower 55% of frame height where reachable workspace sits
        self.pickup_zone = {
            'x_min': 0.25, 'x_max': 0.75,  # ← widened from 0.35-0.65 (was only 30% wide)
            'y_min': 0.35, 'y_max': 0.90   # ← slightly expanded top to catch taller placements
        }
        
        # Size mapping: derived from camera height (~700mm above ground) and webcam FOV (~60°)
        # Formula: pixel_to_mm = (2 * height * tan(FOV/2)) / frame_width_px
        #          = (2 * 700 * tan30°) / 640 = 808 / 640 ≈ 1.26 mm/px
        # Tune this if the claw consistently over/under-grips:
        #   - Claw too wide (drops tomato) → reduce toward 1.0
        #   - Claw too narrow (crushes)   → increase toward 1.5
        self.pixel_to_mm_ratio = 1.26
        
        # Obsolete visual servoing params (kept for backward compatibility logic if used)
        self.alignment_threshold_x = 0.20
        self.alignment_threshold_y = 0.25
        self.alignment_min_tof_mm  = 30
        self.alignment_max_tof_mm  = 400
        self.alignment_cooldown    = 5.0
        self._last_auto_pick_time  = 0.0
        # Auto-searching & Nudging logic
        self.consecutive_no_detections = 0
        self.search_direction = 1 # 1 = up/forward, -1 = down/back
        self.last_adjustment_time = 0
        
        # Movement collection
        self.movements = {}
        self.load_movements()
        
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
        _last_health_log = 0
        while True:
            now_loop = time.time()

            # FIX (Cause 3): Periodic 30s health log — makes silent failures visible in server output
            if now_loop - _last_health_log >= 30.0:
                classifier_name = (
                    'YOLO' if (self.yolo_detector and self.yolo_detector.is_available())
                    else ('ResNet' if self.classifier else 'NONE')
                )
                cam_idx = getattr(self, 'camera_index', 'N/A')
                cam_type = 'USB' if (isinstance(cam_idx, int) and cam_idx > 0) else ('built-in' if cam_idx == 0 else 'N/A')
                self.logger.info(
                    f"[AUTO_LOOP] Status: auto_mode={self.auto_mode} | "
                    f"camera_connected={self.camera_connected} | camera_index={cam_idx}({cam_type}) | "
                    f"arduino_connected={self.arduino_connected} | "
                    f"auto_state={getattr(self, 'auto_state', 'N/A')} | "
                    f"classifier={classifier_name}"
                )
                _last_health_log = now_loop

            if self.auto_mode:
                try:
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.detection_interval:
                        self.process_auto_cycle()
                        self.last_detection_time = current_time
                except Exception as e:
                    self.logger.error(f"Auto Loop Error: {e}", exc_info=True)
            else:
                # Camera resource management: when auto mode is OFF the camera is only needed
                # by active browser feed clients. The _update_frame thread keeps running so the
                # feed stays available, but we slow our own polling to avoid wasting CPU.
                # The camera LED will remain on as long as a browser tab has /video_feed open.
                time.sleep(0.9)  # long idle sleep when not in auto mode

            # Poll distance logic
            poll_interval = 0.5 if self.tof_initialized else 10.0 # Poll much slower if offline
            if self.arduino_connected and (time.time() - getattr(self, 'last_dist_poll', 0) > poll_interval):
                self.get_distance_sensor(force_poll=(not self.tof_initialized))
                self.last_dist_poll = time.time()
            time.sleep(0.1) # Prevent CPU hogging

    def get_arm_body_status(self):
        """Interpret the physical state of the robot based on servo angles
        
        Returns:
            dict: Semantic status of each arm component
        """
        angles = self.current_servo_angles
        
        # Mapping according to user-defined body logic
        status = {
            'tof_orientation': 'downward' if angles['wrist_pitch'] < 90 else 'upside_down_flipped',
            'wrist_roll_pos': 'left' if angles['wrist_roll'] < 90 else ('right' if angles['wrist_roll'] > 90 else 'center'),
            'elbow_pos': 'front' if angles['elbow'] >= 90 else 'back_reversed',
            'shoulder_pos': 'front' if angles['shoulder'] <= 90 else 'back_reversed',
            'claw_state': 'closed' if abs(angles['claw'] - self.servo_limits['claw']['min']) < 5 else 'open_or_active'
        }
        
        # Calculate overall arm 'frontness'
        status['is_facing_front'] = (status['elbow_pos'] == 'front' and status['shoulder_pos'] == 'front')
        
        # Determine if detection is viable (ToF must be facing down)
        status['can_detect_ground'] = (status['tof_orientation'] == 'downward')
        
        return status

    def is_arm_facing_front(self):
        """Verify if arm is oriented toward the front workspace"""
        body = self.get_arm_body_status()
        return body['is_facing_front']

    def process_auto_cycle(self):
        """Auto-pick cycle: Use Fixed Pickup Zone with State Machine."""
        now = time.time()
        
        # --- IDLE & EXT STATE HANDLING ---
        if not self.auto_mode:
            self.auto_state = 'IDLE'
            return

        # FIX (Cause 1): Camera must be connected internally before running detection.
        # get_frame() returns a black 640x480 placeholder (NOT None) when camera_connected=False,
        # so the existing 'if frame is None' check below never fires. We catch it here explicitly.
        if not self.camera_connected:
            self.logger.warning(
                "[AUTO_PICK] BLOCKED: hw_controller.camera_connected=False — "
                "detection cannot run. Browser feed may use a separate video route. "
                "Check /pi/status to confirm camera state."
            )
            return

        if self.auto_state == 'IDLE':
            self.auto_state = 'DETECTING'
            self.logger.info("[AUTO_PICK] Auto-harvest enabled, entering DETECTING state")

        if self.auto_state == 'COOLDOWN':
            if now >= self.cooldown_end_time:
                self.logger.info("[AUTO_PICK] Cooldown finished. Returning to DETECTING state.")
                self.auto_state = 'DETECTING'
            else:
                return

        if self.auto_state == 'HARVESTING':
            # Ignore detections while busy
            return

        # Error state recovery
        if self.auto_state == 'ERROR':
            if now >= getattr(self, 'error_recovery_time', 0):
                self.logger.info("[AUTO_PICK] Error timeout finished, returning to DETECTING.")
                self.auto_state = 'DETECTING'
            return

        frame = self.get_frame()
        if frame is None:
            self.logger.warning("[AUTO_PICK] BLOCKED: get_frame() returned None unexpectedly")
            return

        # --- DETECTION ---
        detections = self.detect_tomatoes(frame)
        h, w = frame.shape[:2]

        # FIX (Cause 4): Log raw count so we can tell 'nothing detected' from 'all filtered out'
        self.logger.info(f"[AUTO_PICK] Frame {w}x{h} | Raw detections: {len(detections)}")

        # Valid targets in zone
        valid_targets = []
        for d in detections:
            cx, cy = d['center']
            class_label = str(d.get('class', 'unknown')).lower()
            conf = d.get('confidence', 0.0)

            # FIX (Cause 4): Log each detection before any filter so we see exactly what was found
            self.logger.info(
                f"[AUTO_PICK] >> Evaluating: class={class_label} conf={conf:.2f} center=({cx},{cy})"
            )

            # Rejection 1: Class and Confidence
            # Accept every label that means the tomato IS ready/ripe:
            #   - ResNet model  → 'ready'
            #   - YOLO 3-class  → 'ripe', 'ready'
            #   - YOLO 1-class  → 'tomato' (any detected tomato counts)
            # Reject only explicit negatives.
            NEGATIVE_CLASSES = {'not_ready', 'unripe', 'spoilt', 'spoiled', 'bad', 'unknown'}
            if class_label in NEGATIVE_CLASSES:
                self.logger.info(f"[AUTO_PICK]    REJECT: class '{class_label}' is a negative class")
                continue
            # Any non-negative label (ready/ripe/tomato/etc.) is accepted — log it for traceability
            self.logger.info(f"[AUTO_PICK]    CLASS PASS: '{class_label}' accepted as harvestable")
            if conf < 0.5:
                self.logger.info(f"[AUTO_PICK]    REJECT: confidence {conf:.2f} < 0.50 threshold")
                continue

            # Rejection 2: Outside Pickup Zone
            rel_x = cx / float(w)
            rel_y = cy / float(h)
            zone = self.pickup_zone
            in_zone = (
                zone['x_min'] <= rel_x <= zone['x_max'] and
                zone['y_min'] <= rel_y <= zone['y_max']
            )
            self.logger.info(
                f"[AUTO_PICK]    Zone check: rel=({rel_x:.2f},{rel_y:.2f}) "
                f"bounds=x[{zone['x_min']:.2f}-{zone['x_max']:.2f}] "
                f"y[{zone['y_min']:.2f}-{zone['y_max']:.2f}] PASS={in_zone}"
            )
            if not in_zone:
                self.logger.info(f"[AUTO_PICK]    REJECT: outside pickup zone")
                continue

            valid_targets.append(d)

        # Decide candidate
        if not valid_targets:
            if self.auto_state in ['TARGET_CANDIDATE', 'TARGET_LOCKED']:
                self.logger.info("[AUTO_PICK] Target lost. Returning to DETECTING.")
            self.auto_state = 'DETECTING'
            self.target_candidate = None
            self.candidate_frames = 0
            return

        # Sort valid targets by closeness to center x implicitly
        valid_targets.sort(key=lambda d: abs(d['center'][0] - float(w)/2))
        best_target = valid_targets[0]
        
        # --- STABILITY LOGIC ---
        if self.auto_state == 'DETECTING':
            self.auto_state = 'TARGET_CANDIDATE'
            self.target_candidate = best_target
            self.candidate_frames = 1
            self.logger.info(f"[AUTO_PICK] Candidate detected: ripe=true conf={best_target['confidence']:.2f} center={best_target['center']}")
            return
            
        if self.auto_state in ['TARGET_CANDIDATE', 'TARGET_LOCKED']:
            # Check if this best target matches the existing candidate (roughly)
            prev_cx, prev_cy = self.target_candidate['center']
            cx, cy = best_target['center']
            
            if abs(cx - prev_cx) < 50 and abs(cy - prev_cy) < 50:
                self.candidate_frames += 1
                self.target_candidate = best_target  # update with latest

                # FIX (Cause 5): Log frame-count progress so we can see if lock is building or stuck
                self.logger.info(
                    f"[AUTO_PICK] Stability: {self.candidate_frames}/{self.candidate_required_frames} frames confirmed"
                )

                if self.candidate_frames == self.candidate_required_frames and self.auto_state != 'TARGET_LOCKED':
                    self.logger.info(f"[AUTO_PICK] Candidate stable for {self.candidate_required_frames} frames")
                    self.logger.info(f"[AUTO_PICK] >>> TARGET LOCKED — ready to harvest <<<")
                    self.auto_state = 'TARGET_LOCKED'
            else:
                self.logger.info("[AUTO_PICK] Candidate moved too much, resetting lock")
                self.target_candidate = best_target
                self.candidate_frames = 1
                self.auto_state = 'TARGET_CANDIDATE'
                return

        if self.auto_state == 'TARGET_LOCKED':
            # Compute width and height parameters
            # Use bbox width, convert by pixel_to_mm
            bbox = self.target_candidate.get('bbox')
            bbox_width_px = 60  # fallback
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # Common format across this repo is [x, y, w, h].
                # Some detectors may instead emit [x_min, y_min, x_max, y_max].
                x = float(bbox[0])
                y = float(bbox[1])
                c = float(bbox[2])
                d = float(bbox[3])
                # Heuristic: if [x + w, y + h] stays inside the frame, treat c/d as width/height.
                if (x + c) <= float(w) and (y + d) <= float(h):
                    bbox_width_px = c
                else:
                    # Interpret as x_max (c) instead of width (w).
                    bbox_width_px = c - x
            else:
                # fallback for different bbox format
                bbox_width_px = self.target_candidate.get('w', 60)
            bbox_width_px = max(0.0, bbox_width_px)
                
            est_width_mm = max(10, min(80, int(bbox_width_px * self.pixel_to_mm_ratio)))
            self.logger.info(f"[AUTO_PICK] Estimated width={est_width_mm} mm")
            
            # Height estimate:
            # We use ToF simply to verify if something is there later or we just pass default height.
            # "Option C: small number of height bands / preset offsets based on apparent size"
            base_height = 50
            if bbox_width_px > 150: 
                base_height = 80 # Big tomato, approach higher
            elif bbox_width_px < 80:
                base_height = 40 # Small tomato
                
            self.logger.info(f"[AUTO_PICK] Estimated height offset={base_height} mm")
            
            # ToF reading (Optional: to verify object presence locally)
            tof_dist = self.get_distance_sensor()
            if tof_dist is not None:
                self.logger.info(f"[AUTO_PICK] ToF sensor reading: {tof_dist} mm")

            # FIX (Cause 2): Block harvest if Arduino is not physically connected.
            # execute_simulated_pick() returns True even with no Arduino (simulation path),
            # which sets state=HARVESTING while the arm never actually moves.
            if not self.arduino_connected:
                self.logger.error(
                    "[AUTO_PICK] BLOCKED: arduino_connected=False — "
                    "PICK_FB command cannot be sent. The arm WILL NOT physically move. "
                    "Connect Arduino via USB serial or BLE before using Auto Mode."
                )
                self.auto_state = 'ERROR'
                self.error_recovery_time = now + 5.0
                self.target_candidate = None
                self.candidate_frames = 0
                return

            self.logger.info(
                f"[AUTO_PICK] Triggering harvest — width={est_width_mm}mm height={base_height}mm"
            )
            success = self.execute_simulated_pick(est_width_mm, base_height)
            
            if success:
                self.logger.info(f"[AUTO_PICK] Harvesting busy, ignoring detections")
                self.auto_state = 'HARVESTING'
            else:
                self.logger.error(f"[AUTO_PICK] ERROR: Failed to trigger harvest")
                self.auto_state = 'ERROR'
                self.error_recovery_time = time.time() + 5.0
            
            # Reset properties for next cycle
            self.target_candidate = None
            self.candidate_frames = 0

    def execute_simulated_pick(self, width_mm=35, object_height_mm=50):
        """Perform a robust front-to-back pick and place sequence.
        
        Args:
            width_mm: Diameter/width of the object to grip (mm). Default 35mm.
            object_height_mm: Height of the object (mm). Taller objects require the
                              arm to pick higher and place higher. Default 50mm.
        
        This uses the firmware-side state machine:
        1. Approach Front -> Down -> Grip -> Lift
        2. Transit via center height
        3. Approach Back -> Place -> Release -> Retreat -> Home
        """
        # Clamp to safe physical ranges
        width_mm         = max(10, min(int(width_mm), 80))
        object_height_mm = max(5,  min(int(object_height_mm), 200))
        
        self.logger.info(
            f"[PICK] Starting front-to-back sequence "
            f"(Width: {width_mm}mm, Height: {object_height_mm}mm)"
        )
        
        # Send both width and height to firmware
        pick_command = f"PICK_FB {width_mm} {object_height_mm}"
        pick_id = f"auto_{int(time.time() * 1000)}"
        
        if self.arduino_connected:
            sent = self.send_command(pick_command)
            if not sent:
                self.logger.error("[PICK] Command send failed while Arduino marked connected")
                return False
            # Store structured pick metadata so handle_pick_result can safely update retries.
            self.pending_picks[str(pick_id)] = {
                'timestamp': float(time.time()),
                'retries': 0,
                'width_mm': width_mm,
                'object_height_mm': object_height_mm,
            }
            return True
        else:
            # FIX (Cause 2 secondary): Upgrade to ERROR so it's unmissable in logs
            self.logger.error(
                "[PICK] *** NO ARDUINO *** PICK_FB command NOT sent — arm WILL NOT move. "
                "Running in simulation mode only."
            )
            # In simulation we mock success and delay fake processing
            self.pending_picks[str(pick_id)] = {
                'timestamp': float(time.time()),
                'retries': 0,
                'width_mm': width_mm,
                'object_height_mm': object_height_mm,
            }
            
            # For pure testing simulation, we spawn a thread to resolve it after 3s
            def simulated_resolve():
                time.sleep(3.0)
                if hasattr(self, 'handle_pick_result'):
                    self.handle_pick_result(pick_id, 'SUCCESS', 'ripe', 3000)
                    
            threading.Thread(target=simulated_resolve, daemon=True).start()
            return True


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
            # Produce a "ripe" tomato inside the pickup zone.
            h_frame, w_frame = frame.shape[:2]
            if time.time() % 5 < 1 and h_frame > 0 and w_frame > 0:
                cx = int(0.5 * w_frame)
                cy = int(0.625 * h_frame)  # matches the dashboard crosshair guidance
                bw = int(0.18 * w_frame)
                bh = int(0.25 * h_frame)
                x = max(0, cx - bw // 2)
                y = max(0, cy - bh // 2)
                bw = max(1, min(bw, w_frame - x))
                bh = max(1, min(bh, h_frame - y))
                return [{
                    'class': 'ripe',
                    'confidence': 0.95,
                    # Use [x, y, w, h] bbox format (repo convention).
                    'bbox': [x, y, bw, bh],
                    'center': [cx, cy]
                }]
            return []

    def _on_ble_data(self, payload):
        """Handle telemetry data received via BLE"""
        try:
            import json
            data = json.loads(payload)
            
            # Distance reading
            if 'distance_mm' in data:
                dist = data['distance_mm']
                if dist < 0:
                    self.last_distance_reading = None
                else:
                    self.last_distance_reading = dist
            
            # Full telemetry
            if 'tof_status' in data:
                self.tof_initialized = data['tof_status']
                if not self.tof_initialized:
                    self.last_distance_reading = None
            
            # Pick result
            if 'status' in data and 'id' in data:
                pick_id = data['id']
                status = data['status']
                result = data.get('result', 'none')
                duration = data.get('duration_ms', 0)
                self.handle_pick_result(pick_id, status, result, duration)
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error parsing BLE telemetry: {e}")

    def get_distance_sensor(self, force_poll=False):
        """Read distance from VL53L0X via Arduino"""
        # If we know it's offline and we aren't forcing a poll, skip immediately
        if not self.tof_initialized and not force_poll:
            return None

        # If using BLE, distance is updated via passive telemetry (notifications)
        if hasattr(self, 'ble_client') and self.ble_client and self.ble_client.connected:
            if self.last_distance_reading is not None:
                return self.last_distance_reading
            else:
                # Optionally send a manual poll if cache is empty
                self.ble_client.send_command("DISTANCE")
                return None

        if self.arduino_connected and self.arduino:
            # Use lock for thread-safe serial access
            with self.serial_lock:
                try:
                    # Clear input buffer to ensure we get fresh data
                    self.arduino.reset_input_buffer()
                    # Send DISTANCE command
                    self.arduino.write(b"DISTANCE\n")
                    time.sleep(0.05)  # Shorter wait
                    
                    # Read response (with a short timeout)
                    response = self.arduino.readline().decode().strip()
                    
                    if response.startswith("DISTANCE: "):
                        distance_str = response.replace("DISTANCE: ", "")
                        if distance_str == "SENSOR_NOT_AVAILABLE":
                            self.tof_initialized = False
                            self.last_distance_reading = None
                            return None
                        if distance_str == "OUT_OF_RANGE":
                            self.tof_initialized = True
                            self.last_distance_reading = None
                            return None
                        try:
                            distance_mm = int(distance_str)
                            self.tof_initialized = True
                            self.last_distance_reading = distance_mm
                            return distance_mm
                        except ValueError:
                            # ...
                            self.logger.error(f"Invalid distance reading: {distance_str}")
                            return None
                    else:
                        return None
                except Exception as e:
                    self.logger.error(f"Distance sensor read error: {e}")
                    self.last_distance_reading = None
                    return None
        else:
            # Simulation mode - return None to show sensor is not active
            self.last_distance_reading = None
            return None

    def start_auto_mode(self):
        self.auto_mode = True
        # Reset state machine so it always starts clean from DETECTING
        self.auto_state = 'DETECTING'
        self.target_candidate = None
        self.candidate_frames = 0
        self.logger.info("Auto Mode ENABLED — state reset to DETECTING")
        return True

    def stop_auto_mode(self):
        self.auto_mode = False
        # Fully reset state machine so next enable starts clean
        self.auto_state = 'IDLE'
        self.target_candidate = None
        self.candidate_frames = 0
        self.logger.info("Auto Mode DISABLED — state reset to IDLE")
        return True
        
    def setup_logging(self):
        """Configure logging for the controller"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
                    self.ble_client = BLEClient(
                        target_name=self.ble_device_name, 
                        on_connect_callback=on_ble_connect,
                        on_data_callback=self._on_ble_data
                    )
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
            # Priority: 1. saved preference  2. USB camera (index>0)  3. built-in (index 0) last resort
            target_index = None

            # 1. Saved preference always wins
            if saved_camera_index is not None and saved_camera_index in self.available_cameras:
                target_index = saved_camera_index
                self.logger.info(f"Using saved camera preference: index {target_index}")
            else:
                # 2. Prefer USB cameras (index > 0) — skip built-in even if camera_index=0 configured
                usb_cameras = [idx for idx in sorted(self.available_cameras) if idx > 0]
                if usb_cameras:
                    target_index = usb_cameras[0]
                    self.logger.info(
                        f"Auto-selecting USB camera (index {target_index}). "
                        f"Built-in camera ignored — USB is preferred for this project."
                    )
                else:
                    # 3. No USB found — only built-in available
                    target_index = self.available_cameras[0]
                    self.logger.warning(
                        f"⚠ Only built-in camera (index {target_index}) found. "
                        "Connect the USB external camera for best results with auto mode."
                    )

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
            camera_pref_file = Path(self.project_root) / 'config' / 'camera_preference.json'
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
            'waist': 'waist',
            'shoulder': 'shoulder',
            'elbow': 'elbow',
            'wrist_roll': 'wrist_roll',
            'wrist_pitch': 'wrist_pitch',
            'claw': 'claw'
        }
        
        tracked_name = servo_map.get(servo_name.lower(), servo_name.lower())
        if tracked_name in self.current_servo_angles:
            try:
                angle = self.clamp_servo(tracked_name, int(angle))
            except Exception:
                pass
            self.current_servo_angles[tracked_name] = int(angle)
            self.logger.debug(f"Updated {tracked_name} angle to {angle}°")

    def clamp_servo(self, servo_name, angle):
        """Ensure servo angle stays within calibrated endpoints."""
        try:
            limits = self.servo_limits.get(servo_name)
            if limits:
                return max(limits['min'], min(limits['max'], int(angle)))
            return int(angle)
        except Exception:
            return angle

    def filter_servo_command(self, command):
        """Filter servo commands to skip unavailable servos
        
        Args:
            command: Command string (e.g., "ANGLE 90 90 90 90 90 110")
        
        Returns:
            str: Filtered command with unavailable servos set to -1 (no change)
        """
        if not command.startswith("ANGLE"):
            return command  # Not a servo command, return as-is
        
        try:
            # Parse ANGLE command: "ANGLE waist shoulder elbow wrist_roll wrist_pitch claw"
            parts = command.split()
            if len(parts) != 7:  # "ANGLE" + 6 servo values
                return command
            
            angles = [int(parts[i]) if parts[i] != '-1' else -1 for i in range(1, 7)]
            
            # Map servo indices to names
            servo_map = ['waist', 'shoulder', 'elbow', 'wrist_roll', 'wrist_pitch', 'claw']
            
            # Replace unavailable servos with -1 (no change) or use fixed angle
            for i, servo_name in enumerate(servo_map):
                if not self.servo_available.get(servo_name, True):
                    # Use -1 to keep current position (servo is manually fixed)
                    angles[i] = -1
                # clamp all servo angles if present
                if angles[i] != -1:
                    angles[i] = self.clamp_servo(servo_name, angles[i])
            
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
            
        # Handle legacy GRIPPER commands
        if command.startswith("GRIPPER"):
            parts = command.split()
            if len(parts) >= 2:
                return self.set_gripper(parts[1])
        
        # Parse ANGLE commands to track servo angles
        if command.startswith("ANGLE"):
            try:
                parts = command.split()
                if len(parts) >= 7:  # ANGLE waist shoulder elbow wrist_roll wrist_pitch claw
                    # Update tracked angles (skip -1 values which mean "keep current")
                    servo_names = ['waist', 'shoulder', 'elbow', 'wrist_roll', 'wrist_pitch', 'claw']
                    for i, angle_str in enumerate(parts[1:7]):
                        if angle_str != '-1' and i < len(servo_names):
                            try:
                                angle = int(angle_str)
                                # clamp value before tracking
                                angle = self.clamp_servo(servo_names[i], angle)
                                # Map 'waist' directly to update_servo_angle
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
                with self.serial_lock:
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
        """Control gripper (OPEN/CLOSE) using ANGLE command"""
        # OPEN: 10, CLOSE: 110 (based on config.h and servoConfig)
        angle = self.claw_min if state.upper() == "OPEN" else self.claw_max
        # Format: ANGLE waist shoulder elbow wrist_roll wrist_pitch claw
        # We only want to change the claw (Index 5)
        return self.send_command(f"ANGLE -1 -1 -1 -1 -1 {angle}")
    
    def handle_pick_result(self, pick_id, status, result, duration_ms=0):
        """Handle pick result from Arduino
        
        Args:
            pick_id: Unique pick identifier
            status: 'SUCCESS', 'FAILED', 'ABORTED', etc.
            result: Result details (e.g., 'ripe', 'unripe', 'none')
            duration_ms: Pick operation duration in milliseconds
        """
        matched_pick_id = pick_id
        if matched_pick_id not in self.pending_picks and self.pending_picks:
            # Firmware-side PICK_FB generates ids like "sim_<millis>" while controller may
            # have generated a local key (e.g. "auto_<timestamp>"). If there is exactly one
            # active pending pick and we are harvesting, treat this result as that pick.
            if getattr(self, 'auto_state', None) == 'HARVESTING' and len(self.pending_picks) == 1:
                matched_pick_id = next(iter(self.pending_picks.keys()))
                self.logger.info(
                    f"[AUTO_PICK] Remapped result id {pick_id} -> pending id {matched_pick_id}"
                )
        
        if matched_pick_id in self.pending_picks:
            pick_info = self.pending_picks[matched_pick_id]
            # Backward compatibility for older in-memory values stored as float timestamps.
            if not isinstance(pick_info, dict):
                pick_info = {
                    'timestamp': float(pick_info),
                    'retries': 0,
                }
                self.pending_picks[matched_pick_id] = pick_info
            if status == 'SUCCESS':
                self.logger.info(f"[AUTO_PICK] Harvest complete (Pick {pick_id} succeeded in {duration_ms}ms)")
                self.logger.info(f"[AUTO_PICK] Cooldown started")
                self.pending_picks.pop(matched_pick_id, None)
                if getattr(self, 'auto_state', None) == 'HARVESTING':
                    self.auto_state = 'COOLDOWN'
                    self.cooldown_end_time = time.time() + self.cooldown_duration_s
            elif status == 'FAILED' or status == 'ABORTED':
                pick_info['retries'] = int(pick_info.get('retries', 0)) + 1
                if pick_info['retries'] < self.max_pick_retries:
                    self.logger.warning(f"[AUTO_PICK] ⚠️ Pick {pick_id} failed ({status}), will retry (attempt {pick_info['retries']}/{self.max_pick_retries})")
                else:
                    self.logger.error(f"[AUTO_PICK] ❌ Pick {pick_id} failed after {pick_info['retries']} attempts: {status}")
                    self.pending_picks.pop(matched_pick_id, None)
                    if getattr(self, 'auto_state', None) == 'HARVESTING':
                        self.auto_state = 'ERROR'
                        self.error_recovery_time = time.time() + 5.0
            else:
                self.logger.warning(f"[AUTO_PICK] Unknown pick status: {status} for {pick_id}")
                self.pending_picks.pop(matched_pick_id, None)
                if getattr(self, 'auto_state', None) == 'HARVESTING':
                    self.auto_state = 'ERROR'
                    self.error_recovery_time = time.time() + 5.0
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
            ts = float(pick_info.get('timestamp', current_time))
            age = current_time - ts
            if age > timeout_seconds:
                stale_picks.append(pick_id)
        
        for pick_id in stale_picks:
            ts = float(self.pending_picks[pick_id].get('timestamp', current_time))
            self.logger.warning(f"[AUTO] Removing stale pick {pick_id} (age: {current_time - ts:.1f}s)")
            self.pending_picks.pop(pick_id, None)
    
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
            calib_json = self.project_root / 'config' / 'calibration_data.json'
            if calib_json.exists():
                import json
                with open(calib_json, 'r') as f:
                    data = json.load(f)
                    # load claw endpoints if present
                    if 'claw_open_angle' in data and 'claw_closed_angle' in data:
                        try:
                            self.claw_min = int(data['claw_open_angle'])
                            self.claw_max = int(data['claw_closed_angle'])
                            self.logger.info(f"✅ Loaded claw endpoints: open={self.claw_min}, closed={self.claw_max}")
                        except Exception:
                            pass
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
                # Ugandan setup: camera height ~350mm, viewport covers roughly 250x180mm
                scale_x = 250.0 / 640.0  # mm per pixel
                scale_y = 180.0 / 480.0  # mm per pixel
                arm_x = pixel_x * scale_x - 125  # Center at 0
                arm_y = pixel_y * scale_y + 80   # Offset from base
                self.logger.debug(f"Using fallback scaling: ({pixel_x}, {pixel_y}) -> ({arm_x:.1f}, {arm_y:.1f})")
                return arm_x, arm_y
                
        except Exception as e:
            self.logger.error(f"Coordinate conversion failed: {e}")
            return None

    def get_current_arm_xyz(self):
        """Calculate current arm end-effector position (X, Y, Z) from servo angles
        
        Returns:
            dict: {'x': x, 'y': y, 'z': z} in mm, or default if unavailable
        """
        try:
            # Get current angles
            waist = self.current_servo_angles.get('waist', 90)
            shoulder = self.current_servo_angles.get('shoulder', 90)
            elbow = self.current_servo_angles.get('elbow', 90)
            
            # Simple Forward Kinematics (Approximation)
            # Real-world measurements from STL files (Fabri Creator v2 Print-in-place)
            l1 = 112.7 # mm (Brazo)
            l2 = 161.9 # mm (Antebrazo)
            l3 = 137.5 # mm (Gripper)
            
            # Convert to radians and adjust for zero-position offsets
            rad_waist = math.radians(waist - 90)
            rad_shoulder = math.radians(shoulder - 90)
            rad_elbow = math.radians(elbow - 90)
            
            # Planar projection for shoulder/elbow
            # distal is the horizontal reach from the central axis
            distal = l1 * math.cos(rad_shoulder) + l2 * math.cos(rad_shoulder + rad_elbow) + l3
            z = l1 * math.sin(rad_shoulder) + l2 * math.sin(rad_shoulder + rad_elbow) + 172.9 # +173mm base height
            
            # Project distal into X, Y based on waist rotation
            x = distal * math.sin(rad_waist)
            y = distal * math.cos(rad_waist)
            
            return {
                'x': round(float(x), 1),
                'y': round(float(y), 1),
                'z': round(float(z), 1)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate current arm XYZ: {e}")
            return {'x': 0, 'y': 150, 'z': 50} # Default safe position

    
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
        
        # Check if any AI model is loaded
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
            'auto_state': getattr(self, 'auto_state', 'IDLE'),
            'movement_active': self.movement_active,
            'connection_type': connection_type,
            'is_facing_front': is_facing_front,
            'tof_initialized': self.tof_initialized,
            'tof_distance': getattr(self, 'last_distance_reading', None),
            'classifier_loaded': model_loaded,
            'body_status': self.get_arm_body_status(),
            'servo_angles': self.current_servo_angles.copy()
        }
        
        if self.ble_client:
            status['ble_connected'] = ble_connected
            status['ble_address'] = self.ble_client.device_address
            status['ble_client_active'] = self.ble_client.thread.is_alive() if self.ble_client.thread else False
            
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
    
    def load_movements(self):
        """Load movement definitions from config/movements.json"""
        import json
        config_path = self.project_root / "config" / "movements.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    self.movements = data.get("movements", {})
                self.logger.info(f"✅ Loaded {len(self.movements)} movements from config")
            except Exception as e:
                self.logger.error(f"❌ Failed to load movements.json: {e}")
        else:
            self.logger.warning("⚠️  movements.json not found")

    def play_movement(self, movement_id):
        """Execute a named movement sequence from the config"""
        self.logger.info(f"Executing movement: {movement_id} 🤖")
        
        # Check if movement exists in config
        if movement_id in self.movements:
            movement = self.movements[movement_id]
            steps = movement.get("steps", [])
            return self.play_sequence(steps, name=movement_id)
        
        # Fallback for legacy hardcoded movements or complex ones
        if movement_id == "home":
            self.home_arm()
            return True
            
        self.logger.warning(f"Movement '{movement_id}' not found in configuration")
        return False

    def play_sequence(self, steps, name="Sequence"):
        """Play an arbitrary list of steps (with angles, speed, delay_ms)"""
        # Save current mode
        previous_auto_mode = self.auto_mode
        self.auto_mode = False
        self.movement_active = True
        
        last_timestamp_ms = 0
        try:
            for step in steps:
                angles = step.get("angles", [])
                speed = step.get("speed")
                delay_ms = step.get("delay_ms", 500)
                comment = step.get("comment", "")
                
                # If a step has all 6 angles
                if "angles" in step and len(step["angles"]) == 6:
                    if speed is not None:
                        self.send_command(f"SPEED {speed}")
                        self.logger.debug(f"  Speed set to {speed}")
                        
                    cmd = f"ANGLE {' '.join(str(a) for a in step['angles'])}"
                    self.send_command(cmd)
                    if comment:
                        self.logger.debug(f"  Step: {comment}")
                    time.sleep(delay_ms / 1000.0)
                else:
                    # Support for individual recorded servo steps
                    servo = step.get("servo")
                    angle = step.get("angle")
                    interval_ms = step.get("interval_ms")
                    timestamp_ms = step.get("timestamp")
                    
                    if servo and angle is not None:
                        # Build ANGLE command keeping others at -1 (ignore)
                        idx_map = {"waist": 0, "shoulder": 1, "elbow": 2, "wrist_roll": 3, "wrist_pitch": 4, "claw": 5}
                        if servo in idx_map:
                            cmd_angles = [-1]*6
                            cmd_angles[idx_map[servo]] = angle
                            
                            # Handle timing
                            if timestamp_ms is not None:
                                delay = (timestamp_ms - last_timestamp_ms) / 1000.0
                                if delay > 0:
                                    time.sleep(delay)
                                last_timestamp_ms = timestamp_ms
                            elif interval_ms is not None:
                                time.sleep(interval_ms / 1000.0)
                            else:
                                time.sleep(0.05)
                                
                            # Apply dynamic speed if recorded
                            if speed is not None:
                                self.send_command(f"SPEED {speed}")
                                
                            self.send_command(f"ANGLE {' '.join(str(a) for a in cmd_angles)}")
            
            self.logger.info(f"Playback '{name}' complete!")
            return True
        except Exception as e:
            self.logger.error(f"Error playing '{name}': {e}")
            return False
        finally:
            self.movement_active = False
            self.auto_mode = previous_auto_mode

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
