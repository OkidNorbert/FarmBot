#!/usr/bin/env python3
"""
Unified Web Interface for FarmBOT
==================================

A comprehensive web interface for training AI models, controlling hardware,
and monitoring the FarmBOT system. Works on both regular systems and Raspberry Pi.
"""

import os
import sys
import json
import shutil
import subprocess
import csv
import io
from pathlib import Path
from datetime import datetime
import threading
import time

# Suppress OpenCV warnings and errors globally
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Only show errors, suppress warnings
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'  # Disable video I/O debug messages

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, Response
from werkzeug.utils import secure_filename
import yaml
import cv2
import numpy as np

# Set OpenCV log level to suppress warnings
try:
    cv2.setLogLevel(0)  # 0 = SILENT, 1 = ERROR, 2 = WARN, 3 = INFO, 4 = DEBUG
except:
    pass  # Older OpenCV versions might not have this

# Try to import flask_socketio (required for WebSocket support)
# Note: These imports are optional - if not installed, fallback classes are used
# The linter warning about unresolved imports is expected and can be safely ignored
try:
    import warnings
    # Suppress eventlet deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import eventlet  # noqa: F401  # pyright: ignore
    from flask_socketio import SocketIO, emit  # noqa: F401  # pyright: ignore
    SOCKETIO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: flask_socketio not available: {e}")
    print("Install with: pip install flask-socketio eventlet")
    # Create dummy classes to prevent errors
    class SocketIO:
        def __init__(self, *args, **kwargs):
            pass
        def on(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def emit(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            pass
    def emit(*args, **kwargs):
        pass
    SOCKETIO_AVAILABLE = False

# Try to import hardware controller (optional for non-Pi systems)
try:
    from hardware_controller import HardwareController
    # Initialize with Bluetooth support - will try Bluetooth if serial not available
    # Arduino advertises as "FarmBot" (from firmware)
    hw_controller = HardwareController(connection_type='auto', ble_device_name="FarmBot")
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Warning: Hardware controller not available. Running in software-only mode.")
    hw_controller = None
    HARDWARE_AVAILABLE = False
except Exception as e:
    print(f"Warning: Could not initialize hardware controller: {e}")
    hw_controller = None
    HARDWARE_AVAILABLE = False

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Disable caching for static files in development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable browser caching

# Add context processor to inject timestamp into all templates for cache-busting
@app.context_processor
def inject_timestamp():
    return dict(timestamp=int(time.time()))

# Disable caching for all responses
@app.after_request
def add_no_cache_headers(response):
    """Add no-cache headers to prevent browser caching"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Initialize SocketIO if available
if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
else:
    socketio = SocketIO(app)  # Dummy instance

# WebSocket client tracking
arduino_clients = set()  # Track connected Arduino WebSocket clients

# Configuration
UPLOAD_FOLDER = 'datasets'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
LOG_FILE = 'detection_log.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for training status
training_status = {
    'is_training': False,
    'current_crop': None,
    'progress': 0,
    'status_message': '',
    'logs': []
}

# Global variables for system state
system_state = {
    'running': False,
    'detection_count': 0,
    'ripe_count': 0,
    'unripe_count': 0,
    'spoilt_count': 0,
    'camera_connected': False,
    'arduino_connected': False,
    'classifier_loaded': False,
    'last_detection': None,
    'current_frame': None,
    'session_start': datetime.now().isoformat()
}

# Global variable for camera index (fallback when hardware controller is not available)
CURRENT_CAMERA_INDEX = 0

# Camera list cache
CAMERA_LIST_CACHE = None
LAST_CAMERA_SCAN_TIME = 0
CAMERA_CACHE_TTL = 60  # Cache for 60 seconds

# Statistics storage file
STATS_FILE = 'monitoring_stats.json'

# Initialize stats on module load
_initialized_stats = False

def initialize_stats():
    """Initialize stats file on startup"""
    global _initialized_stats
    if not _initialized_stats:
        stats = load_stats()  # This will create the file if it doesn't exist
        # Add current session to session history
        if 'session_history' not in stats:
            stats['session_history'] = []
        
        # Add current session
        current_session = {
            'start_time': datetime.now().isoformat(),
            'session_id': f"session_{int(time.time())}"
        }
        stats['session_history'].append(current_session)
        
        # Keep only last 50 sessions
        if len(stats['session_history']) > 50:
            stats['session_history'] = stats['session_history'][-50:]
        
        save_stats(stats)
        _initialized_stats = True
        print(f"📊 Monitoring stats initialized: {STATS_FILE}")

def load_stats():
    """Load statistics from file"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
                # Ensure all required fields exist
                if 'detection_history' not in stats:
                    stats['detection_history'] = []
                if 'session_history' not in stats:
                    stats['session_history'] = []
                return stats
        except Exception as e:
            print(f"Error loading stats: {e}")
            # Return default and save it
            default_stats = {
                'total_sorted': 0,
                'ripe_count': 0,
                'unripe_count': 0,
                'spoilt_count': 0,
                'detection_history': [],
                'session_history': []
            }
            save_stats(default_stats)
            return default_stats
    
    # File doesn't exist, create it with defaults
    default_stats = {
        'total_sorted': 0,
        'ripe_count': 0,
        'unripe_count': 0,
        'spoilt_count': 0,
        'detection_history': [],
        'session_history': []
    }
    save_stats(default_stats)
    return default_stats

def save_stats(stats):
    """Save statistics to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(STATS_FILE) if os.path.dirname(STATS_FILE) else '.', exist_ok=True)
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving stats: {e}")
        import traceback
        traceback.print_exc()

def update_sorting_stats(class_name):
    """Update sorting statistics when a tomato is sorted"""
    try:
        stats = load_stats()
        stats['total_sorted'] += 1
        
        # Normalize class name for consistency
        class_name_lower = str(class_name).lower()
        
        if class_name_lower in ['ready', 'ripe']:
            stats['ripe_count'] += 1
            system_state['ripe_count'] = stats['ripe_count']
        elif class_name_lower in ['not_ready', 'unripe']:
            stats['unripe_count'] += 1
            system_state['unripe_count'] = stats['unripe_count']
        elif class_name_lower in ['spoilt', 'spoiled']:
            stats['spoilt_count'] += 1
            system_state['spoilt_count'] = stats['spoilt_count']
        
        # Add to detection history (keep last 1000 for better retention)
        detection_entry = {
            'timestamp': datetime.now().isoformat(),
            'class': class_name_lower
        }
        stats['detection_history'].append(detection_entry)
        if len(stats['detection_history']) > 1000:
            stats['detection_history'] = stats['detection_history'][-1000:]
        
        save_stats(stats)
        print(f"📊 Stats updated: {class_name_lower} - Total: {stats['total_sorted']}")
    except Exception as e:
        print(f"Error updating sorting stats: {e}")
        import traceback
        traceback.print_exc()

def rotate_log_if_needed():
    """Rotate log file if it exceeds 10MB"""
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 10 * 1024 * 1024:  # 10MB
            # Create logs directory if it doesn't exist
            logs_dir = 'logs'
            os.makedirs(logs_dir, exist_ok=True)
            
            # Archive old log with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_log = os.path.join(logs_dir, f"detection_log_{timestamp}.csv")
            shutil.move(LOG_FILE, archived_log)
            print(f"📝 Log rotated: {archived_log}")
            
            # Keep only last 10 archived logs
            archived_logs = sorted([f for f in os.listdir(logs_dir) if f.startswith('detection_log_') and f.endswith('.csv')])
            if len(archived_logs) > 10:
                for old_log in archived_logs[:-10]:
                    try:
                        os.remove(os.path.join(logs_dir, old_log))
                    except:
                        pass
    except Exception as e:
        print(f"Error rotating log: {e}")

def log_detection(detection_data):
    """Log detection event to CSV and update statistics"""
    # Update sorting statistics
    if 'class' in detection_data:
        update_sorting_stats(detection_data['class'])
    
    # Rotate log if needed
    rotate_log_if_needed()
    
    # Use file locking to prevent race conditions
    import fcntl
    
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            # Acquire exclusive lock to prevent race conditions
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.writer(f)
                
                # Check file size to determine if header exists (thread-safe)
                # If file is empty or just created, write header
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Reset to beginning
                
                if file_size == 0:
                    writer.writerow(['timestamp', 'detection_id', 'class', 'confidence'])
                
                # Write data
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    detection_data.get('id', 'unknown'),
                    detection_data.get('class', 'unknown'),
                    detection_data.get('confidence', 0.0)
                ])
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        # Fallback for systems without fcntl (Windows)
        try:
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(LOG_FILE) == 0:
                    writer.writerow(['timestamp', 'detection_id', 'class', 'confidence'])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    detection_data.get('id', 'unknown'),
                    detection_data.get('class', 'unknown'),
                    detection_data.get('confidence', 0.0)
                ])
        except Exception as e:
            print(f"Error logging detection: {e}")
    except Exception as e:
        print(f"Error logging detection: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_datasets():
    """Get list of available datasets"""
    datasets = []
    if os.path.exists(UPLOAD_FOLDER):
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            if os.path.isdir(item_path):
                # Count images in dataset
                image_count = 0
                class_folders = set()
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                            image_count += 1
                    # Look for class folders in train/val/test subdirectories
                    if root != item_path and os.path.basename(os.path.dirname(root)) in ['train', 'val', 'test']:
                        class_name = os.path.basename(root)
                        if class_name not in ['train', 'val', 'test']:
                            class_folders.add(class_name)
                
                datasets.append({
                    'name': item,
                    'path': item_path,
                    'image_count': image_count,
                    'classes': list(class_folders),
                    'has_model': os.path.exists(os.path.join(MODELS_FOLDER, item))
                })
    return datasets

def get_models():
    """Get list of trained models"""
    models = []
    if os.path.exists(MODELS_FOLDER):
        for item in os.listdir(MODELS_FOLDER):
            item_path = os.path.join(MODELS_FOLDER, item)
            if os.path.isdir(item_path):
                # Check for model files
                model_files = []
                metadata = None
                
                for file in os.listdir(item_path):
                    if file.endswith('.pth'):
                        model_files.append(file)
                    elif file == 'training_metadata.json':
                        try:
                            with open(os.path.join(item_path, file), 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                
                if model_files:
                    models.append({
                        'name': item,
                        'path': item_path,
                        'model_files': model_files,
                        'metadata': metadata,
                        'inference_script': f"{item}_inference.py" if f"{item}_inference.py" in os.listdir(item_path) else None
                    })
    return models

# ==========================================
# Raspberry Pi Hardware Control Routes
# ==========================================

@app.route('/pi/status')
def pi_status():
    """Get system status"""
    if HARDWARE_AVAILABLE and hw_controller:
        hw_status = hw_controller.get_status()
        system_state['camera_connected'] = hw_status['camera_connected']
        system_state['arduino_connected'] = hw_status['arduino_connected']
        system_state['auto_mode'] = hw_status.get('auto_mode', False)
        system_state['connection_type'] = hw_status.get('connection_type', 'none')
        system_state['classifier_loaded'] = hw_status.get('classifier_loaded', False)
    else:
        system_state['camera_connected'] = False
        system_state['arduino_connected'] = False
        system_state['auto_mode'] = False
        system_state['connection_type'] = 'none'
        system_state['classifier_loaded'] = False
    
    # Load persistent stats
    stats = load_stats()
    system_state['ripe_count'] = stats.get('ripe_count', 0)
    system_state['unripe_count'] = stats.get('unripe_count', 0)
    system_state['spoilt_count'] = stats.get('spoilt_count', 0)
    
    return jsonify(system_state)

@app.route('/api/monitor/debug')
def api_monitor_debug():
    """Debug endpoint to check stats file status"""
    try:
        stats_file_exists = os.path.exists(STATS_FILE)
        log_file_exists = os.path.exists(LOG_FILE)
        
        stats = None
        if stats_file_exists:
            try:
                stats = load_stats()
            except Exception as e:
                stats = {'error': str(e)}
        
        log_size = 0
        if log_file_exists:
            log_size = os.path.getsize(LOG_FILE)
        
        return jsonify({
            'stats_file_exists': stats_file_exists,
            'stats_file_path': os.path.abspath(STATS_FILE),
            'log_file_exists': log_file_exists,
            'log_file_size': log_size,
            'stats': stats,
            'system_state': system_state
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/stats')
def api_monitor_stats():
    """Get detailed monitoring statistics"""
    try:
        # Ensure stats are initialized
        initialize_stats()
        stats = load_stats()
        
        # Sync stats with system_state for consistency
        system_state['ripe_count'] = stats.get('ripe_count', 0)
        system_state['unripe_count'] = stats.get('unripe_count', 0)
        system_state['spoilt_count'] = stats.get('spoilt_count', 0)
        
        # Update system state from hardware controller if available
        if HARDWARE_AVAILABLE and hw_controller:
            try:
                hw_status = hw_controller.get_status()
                system_state['camera_connected'] = hw_status.get('camera_connected', False)
                system_state['arduino_connected'] = hw_status.get('arduino_connected', False)
                system_state['auto_mode'] = hw_status.get('auto_mode', False)
                system_state['connection_type'] = hw_status.get('connection_type', 'none')
            except Exception as e:
                print(f"Error getting hardware status: {e}")
        
        status = system_state.copy()
        
        # Ensure session_start is set
        if not status.get('session_start'):
            status['session_start'] = datetime.now().isoformat()
            system_state['session_start'] = status['session_start']
        
        # Get detection rate (per minute)
        detection_history = stats.get('detection_history', [])
        if len(detection_history) > 0:
            # Calculate detections in last minute
            now = datetime.now()
            try:
                recent_detections = [
                    d for d in detection_history 
                    if isinstance(d, dict) and 'timestamp' in d and
                    (now - datetime.fromisoformat(d['timestamp'])).total_seconds() < 60
                ]
                detection_rate = len(recent_detections)
            except Exception as e:
                print(f"Error calculating detection rate: {e}")
                detection_rate = 0
        else:
            detection_rate = 0
        
        # Get tomato detection status
        tomato_status = {'detected': False, 'tomato_count': 0}
        try:
            # Try to get detection status from camera if available
            if HARDWARE_AVAILABLE and hw_controller:
                if hasattr(hw_controller, 'camera_connected') and hw_controller.camera_connected:
                    if hasattr(hw_controller, 'camera') and hw_controller.camera is not None:
                        ret, frame = hw_controller.camera.read()
                        if ret and frame is not None:
                            tomato_count = count_tomatoes_in_frame(frame)
                            tomato_status = {
                                'detected': tomato_count > 0,
                                'tomato_count': tomato_count
                            }
        except Exception as e:
            # If detection fails, just use defaults
            print(f"Error getting tomato detection: {e}")
            pass
        
        # Ensure we return the latest stats from file
        response_data = {
            'total_sorted': stats.get('total_sorted', 0),
            'ripe_count': stats.get('ripe_count', 0),
            'unripe_count': stats.get('unripe_count', 0),
            'spoilt_count': stats.get('spoilt_count', 0),
            'detection_count': status.get('detection_count', 0),
            'detection_rate': detection_rate,
            'detection_history': detection_history[-20:] if detection_history else [],  # Last 20 detections
            'session_history': stats.get('session_history', [])[-10:] if stats.get('session_history') else [],  # Last 10 sessions
            'tomato_detected': tomato_status.get('detected', False),
            'tomato_count': tomato_status.get('tomato_count', 0),
            'camera_connected': status.get('camera_connected', False),
            'arduino_connected': status.get('arduino_connected', False),
            'auto_mode': status.get('auto_mode', False),
            'session_start': status.get('session_start', datetime.now().isoformat())
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in api_monitor_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'total_sorted': 0,
            'ripe_count': 0,
            'unripe_count': 0,
            'spoilt_count': 0,
            'detection_count': 0,
            'detection_rate': 0,
            'detection_history': [],
            'tomato_detected': False,
            'tomato_count': 0,
            'camera_connected': False,
            'arduino_connected': False,
            'auto_mode': False,
            'session_start': datetime.now().isoformat()
        }), 500

@app.route('/api/auto/start', methods=['POST'])
def api_start_auto():
    """Enable Auto Mode"""
    system_state['auto_mode'] = True
    system_state['system_mode'] = 'AUTO'
    
    # Send mode change to Arduino via BLE/Serial
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.auto_mode = True
        if hw_controller.arduino_connected:
            import json
            mode_command = json.dumps({"cmd": "set_mode", "mode": "AUTO"})
            if hw_controller.ble_client and hw_controller.ble_client.connected:
                hw_controller.ble_client.send_command(mode_command)
            elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                hw_controller.arduino.write(f"{mode_command}\n".encode())
        hw_controller.start_auto_mode()
    
    # Also send via WebSocket if Arduino clients connected
    if arduino_clients:
        socketio.emit('command', {'cmd': 'set_mode', 'mode': 'AUTO'}, namespace='/arduino')
    
    return jsonify({'success': True, 'message': 'Auto Mode Started (Speed: 45 deg/s)', 'mode': 'AUTO'})

@app.route('/api/auto/stop', methods=['POST'])
def api_stop_auto():
    """Disable Auto Mode"""
    system_state['auto_mode'] = False
    system_state['system_mode'] = 'MANUAL'
    
    # Send mode change to Arduino via BLE/Serial
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.auto_mode = False
        if hw_controller.arduino_connected:
            import json
            mode_command = json.dumps({"cmd": "set_mode", "mode": "MANUAL"})
            if hw_controller.ble_client and hw_controller.ble_client.connected:
                hw_controller.ble_client.send_command(mode_command)
            elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                hw_controller.arduino.write(f"{mode_command}\n".encode())
        hw_controller.stop_auto_mode()
    
    # Also send via WebSocket if Arduino clients connected
    if arduino_clients:
        socketio.emit('command', {'cmd': 'set_mode', 'mode': 'MANUAL'}, namespace='/arduino')
    
    return jsonify({'success': True, 'message': 'Auto Mode Stopped (Manual mode - Max speed: 120 deg/s)', 'mode': 'MANUAL'})

@app.route('/api/control/mode', methods=['POST'])
def api_set_mode():
    """Set system mode (AUTO/MANUAL)"""
    data = request.get_json() or {}
    mode = data.get('mode', 'MANUAL').upper()
    
    if mode == 'AUTO':
        system_state['auto_mode'] = True
        system_state['system_mode'] = 'AUTO'
    else:
        system_state['auto_mode'] = False
        system_state['system_mode'] = 'MANUAL'
    
    # Send mode change to Arduino via BLE/Serial
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.auto_mode = (mode == 'AUTO')
        if hw_controller.arduino_connected:
            import json
            mode_command = json.dumps({"cmd": "set_mode", "mode": mode})
            if hw_controller.ble_client and hw_controller.ble_client.connected:
                hw_controller.ble_client.send_command(mode_command)
            elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                hw_controller.arduino.write(f"{mode_command}\n".encode())
    
    # Also send via WebSocket if Arduino clients connected
    if arduino_clients:
        socketio.emit('command', {'cmd': 'set_mode', 'mode': mode}, namespace='/arduino')
    
    speed_info = "45 deg/s" if mode == 'AUTO' else "max 120 deg/s"
    return jsonify({'success': True, 'mode': mode, 'message': f'Mode set to {mode} (Speed: {speed_info})'})

@app.route('/api/control/emergency_stop', methods=['POST'])
def api_emergency_stop():
    """Trigger emergency stop"""
    if arduino_clients:
        socketio.emit('command', {'cmd': 'stop'}, namespace='/arduino')
    
    system_state['auto_mode'] = False
    system_state['system_mode'] = 'MANUAL'
    
    return jsonify({'success': True, 'message': 'Emergency stop activated'})

@app.route('/api/manual/move', methods=['POST'])
def api_manual_move():
    """Send manual joint angles to Arduino"""
    data = request.get_json()
    
    if not arduino_clients:
        return jsonify({'error': 'Arduino not connected'}), 400
    
    command = {
        'cmd': 'move_joints',
        'base': data.get('base', 90),
        'shoulder': data.get('shoulder', 90),
        'forearm': data.get('forearm', 90),
        'elbow': data.get('elbow', 90),
        'pitch': data.get('pitch', 90),
        'claw': data.get('claw', 90)
    }
    
    socketio.emit('command', command, namespace='/arduino')
    return jsonify({'success': True, 'message': 'Manual move command sent'})

@app.route('/pi/control/start')
def start_detection():
    """Start detection and sorting"""
    system_state['running'] = True
    return jsonify({'status': 'started', 'message': 'Detection started'})

@app.route('/pi/control/stop')
def stop_detection():
    """Stop detection and sorting"""
    system_state['running'] = False
    return jsonify({'status': 'stopped', 'message': 'Detection stopped'})

@app.route('/pi/control/home')
def home_arm():
    """Move arm to home position"""
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.home_arm()
        return jsonify({'status': 'home', 'message': 'Arm moved to home position'})
    return jsonify({'status': 'error', 'message': 'Hardware not available'})

@app.route('/pi/control/calibrate')
def start_calibration():
    """Start coordinate calibration"""
    return jsonify({'status': 'calibration', 'message': 'Calibration started'})

@app.route('/api/system/start', methods=['POST'])
def api_start_system():
    """API endpoint to start system"""
    system_state['running'] = True
    return jsonify({'success': True, 'message': 'System started'})

@app.route('/api/system/stop', methods=['POST'])
def api_stop_system():
    """API endpoint to stop system"""
    system_state['running'] = False
    return jsonify({'success': True, 'message': 'System stopped'})

@app.route('/api/arm/move', methods=['POST'])
def api_move_arm():
    """API endpoint to move arm - supports both absolute and relative movement"""
    if not HARDWARE_AVAILABLE or not hw_controller:
        return jsonify({'success': False, 'message': 'Hardware not available'})
    
    # Check if request has JSON data
    if not request.is_json or request.json is None:
        return jsonify({'success': False, 'message': 'Invalid request: JSON data required'}), 400
    
    data = request.json
    
    # Check if this is a relative movement (axis + value format)
    if 'axis' in data and 'value' in data:
        # Relative movement - need to track current position
        # For now, use a simple approach: send relative movement as absolute
        # This is a simplified implementation - in production, track actual position
        axis = data.get('axis', 'x').lower()
        value = data.get('value', 0)
        relative = data.get('relative', True)
        
        # Initialize position tracking if not exists
        if not hasattr(api_move_arm, 'current_pos'):
            api_move_arm.current_pos = {'x': 0, 'y': 0, 'z': 0}
        
        # Update position based on axis
        if relative:
            api_move_arm.current_pos[axis] += value
        else:
            api_move_arm.current_pos[axis] = value
        
        x = api_move_arm.current_pos['x']
        y = api_move_arm.current_pos['y']
        z = api_move_arm.current_pos['z']
        
        hw_controller.move_arm(x, y, z)
        return jsonify({'success': True, 'message': f'Arm moved {axis} by {value} to ({x}, {y}, {z})'})
    else:
        # Absolute movement (x, y, z format)
        x = data.get('x', 0)
        y = data.get('y', 0)
        z = data.get('z', 0)
        
        # Update tracked position
        if not hasattr(api_move_arm, 'current_pos'):
            api_move_arm.current_pos = {'x': 0, 'y': 0, 'z': 0}
        api_move_arm.current_pos['x'] = x
        api_move_arm.current_pos['y'] = y
        api_move_arm.current_pos['z'] = z
        
        hw_controller.move_arm(x, y, z)
        return jsonify({'success': True, 'message': f'Arm moved to ({x}, {y}, {z})'})

@app.route('/api/arm/home', methods=['POST'])
def api_home_arm():
    """API endpoint to home arm"""
    if HARDWARE_AVAILABLE and hw_controller:
        # Reset tracked position
        if hasattr(api_move_arm, 'current_pos'):
            api_move_arm.current_pos = {'x': 0, 'y': 0, 'z': 0}
        hw_controller.home_arm()
        return jsonify({'success': True, 'message': 'Arm homed'})
    return jsonify({'success': False, 'message': 'Hardware not available'})

@app.route('/api/servo/set', methods=['POST'])
def api_set_servo():
    """API endpoint to set individual servo angle"""
    if not HARDWARE_AVAILABLE or not hw_controller:
        return jsonify({'success': False, 'message': 'Hardware not available'})
    
    if not request.is_json or request.json is None:
        return jsonify({'success': False, 'message': 'Invalid request: JSON data required'}), 400
    
    data = request.json
    servo = data.get('servo', '').lower()
    angle = data.get('angle', 90)
    
    # Map servo names to indices: 0=base, 1=shoulder, 2=forearm, 3=elbow, 4=pitch, 5=claw
    servo_map = {
        'base': 0,
        'shoulder': 1,
        'forearm': 2,
        'elbow': 3,
        'pitch': 4,
        'claw': 5
    }
    
    if servo not in servo_map:
        return jsonify({'success': False, 'message': f'Invalid servo name: {servo}'}), 400
    
    # Send ANGLE command with -1 for servos we don't want to change
    # Format: ANGLE base shoulder forearm elbow pitch claw
    # Use -1 to keep current angle for other servos
    angles = [-1, -1, -1, -1, -1, -1]
    angles[servo_map[servo]] = int(angle)
    
    # Build ANGLE command
    angle_cmd = f"ANGLE {' '.join(map(str, angles))}"
    hw_controller.send_command(angle_cmd)
    
    return jsonify({'success': True, 'message': f'Servo {servo} set to {angle}°'})

@app.route('/api/bluetooth/scan', methods=['POST'])
def api_bluetooth_scan():
    """API endpoint to scan for Bluetooth devices"""
    try:
        if not HARDWARE_AVAILABLE or not hw_controller:
            return jsonify({'success': False, 'message': 'Hardware controller not available'}), 500
        
        # Scan for Bluetooth devices
        devices = hw_controller.scan_bluetooth_devices(timeout=10)
        return jsonify({'success': True, 'devices': devices})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bluetooth/connect', methods=['POST'])
def api_bluetooth_connect():
    """API endpoint to connect to Bluetooth device"""
    try:
        if not HARDWARE_AVAILABLE or not hw_controller:
            return jsonify({'success': False, 'message': 'Hardware controller not available'}), 500
        
        data = request.json
        address = data.get('address')
        name = data.get('name', 'FarmBot')
        
        if not address:
            return jsonify({'success': False, 'message': 'No device address provided'}), 400
        
        # Connect to Bluetooth device
        if hw_controller.connect_bluetooth_device(address, name):
            # Wait a moment and check status
            import time
            time.sleep(2)
            status = hw_controller.get_status()
            return jsonify({
                'success': True, 
                'message': f'Connected to {name} ({address})',
                'address': address,
                'connected': status.get('arduino_connected', False),
                'ble_connected': status.get('ble_connected', False)
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to connect to device. It may already be connected by the OS.'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bluetooth/disconnect', methods=['POST'])
def api_bluetooth_disconnect():
    """API endpoint to disconnect from Bluetooth device"""
    try:
        if not HARDWARE_AVAILABLE or not hw_controller:
            return jsonify({'success': False, 'message': 'Hardware controller not available'}), 500
        
        if hw_controller.ble_client:
            hw_controller.ble_client.disconnect()
            hw_controller.arduino_connected = False
            return jsonify({'success': True, 'message': 'Disconnected from Bluetooth device'})
        else:
            return jsonify({'success': False, 'message': 'No Bluetooth connection active'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bluetooth/status', methods=['GET'])
def api_bluetooth_status():
    """API endpoint to get Bluetooth connection status"""
    try:
        if not HARDWARE_AVAILABLE or not hw_controller:
            return jsonify({'success': False, 'message': 'Hardware controller not available'}), 500
        
        status = hw_controller.get_status()
        return jsonify({
            'success': True,
            'connected': status.get('arduino_connected', False),
            'connection_type': status.get('connection_type', 'none'),
            'ble_connected': status.get('ble_connected', False),
            'ble_address': status.get('ble_address', None)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/list', methods=['GET'])
def api_list_cameras():
    """API endpoint to list all available cameras (built-in and USB)"""
    global CAMERA_LIST_CACHE, LAST_CAMERA_SCAN_TIME
    
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Use cache only if not forcing refresh
        if not force_refresh and CAMERA_LIST_CACHE is not None:
            # Update current status in cached list
            for cam in CAMERA_LIST_CACHE:
                cam['current'] = (cam['index'] == CURRENT_CAMERA_INDEX)
                if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
                     cam['current'] = (cam['index'] == hw_controller.camera_index)
            
            return jsonify({
                'success': True, 
                'cameras': CAMERA_LIST_CACHE, 
                'current': hw_controller.camera_index if (HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected) else CURRENT_CAMERA_INDEX
            })

        # If no cache, build from hardware controller's available_cameras (fast, no camera opening)
        if HARDWARE_AVAILABLE and hw_controller:
            # Use available_cameras list from hardware controller (already detected, no need to open cameras)
            if hasattr(hw_controller, 'available_cameras') and hw_controller.available_cameras:
                # Build camera list from available_cameras (already detected, no camera opening)
                cameras = []
                for idx in hw_controller.available_cameras:
                    try:
                        camera_info = {
                            'index': idx,
                            'name': f"Camera {idx}",
                            'type': 'Built-in' if idx == 0 else 'USB',
                            'backend': 'V4L2',
                            'resolution': '640x480',
                            'current': idx == (hw_controller.camera_index if hw_controller.camera_connected else None)
                        }
                        cameras.append(camera_info)
                    except:
                        pass
                current = hw_controller.camera_index if hw_controller.camera_connected else None
                
                # Cache the result
                CAMERA_LIST_CACHE = cameras
                LAST_CAMERA_SCAN_TIME = time.time()
                
                return jsonify({
                    'success': True, 
                    'cameras': cameras, 
                    'current': current
                })
            else:
                # No available_cameras list, fall through to /dev/video* fallback
                pass
        
        # Fallback: Actually test cameras to see which ones work
        cameras = []
        current = None
        
        # First, get list of /dev/video* devices
        video_devices = []
        if os.path.exists('/dev'):
            for item in sorted(os.listdir('/dev')):
                if item.startswith('video'):
                    try:
                        idx = int(item.replace('video', ''))
                        video_devices.append(idx)
                    except ValueError:
                        pass
        
        # If force refresh or no cache, actually test cameras
        if force_refresh or CAMERA_LIST_CACHE is None:
            # Test each camera index to see if it actually works
            # Only test indices that exist as /dev/video* devices to avoid unnecessary warnings
            indices_to_test = sorted(video_devices) if video_devices else list(range(10))
            
            # Suppress OpenCV warnings during camera testing
            import os
            import sys
            import warnings
            
            # Save original stderr
            original_stderr_fd = sys.stderr.fileno()
            saved_stderr_fd = os.dup(original_stderr_fd)
            
            try:
                # Redirect stderr to suppress OpenCV warnings
                with open(os.devnull, 'w') as devnull:
                    os.dup2(devnull.fileno(), original_stderr_fd)
                    
                    for idx in indices_to_test:
                        test_cap = None
                        try:
                            # Try to open camera with V4L2 backend
                            backend = cv2.CAP_V4L2 if idx in video_devices else cv2.CAP_ANY
                            
                            # Suppress OpenCV warnings for this specific call
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                test_cap = cv2.VideoCapture(idx, backend)
                            
                            if test_cap is not None and test_cap.isOpened():
                                # Try to read a frame to verify it works
                                ret, frame = test_cap.read()
                                if ret and frame is not None:
                                    # Get camera properties
                                    try:
                                        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                                        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                                    except:
                                        width, height = 640, 480
                                    
                                    # Try to determine camera type and name
                                    camera_type = "USB"
                                    camera_name = f"Camera {idx}"
                                    
                                    # Try to get more info about the camera device
                                    try:
                                        import subprocess
                                        # Try to get device info using v4l2-ctl if available
                                        result = subprocess.run(['v4l2-ctl', '--device=/dev/video{}'.format(idx), '--info'], 
                                                              capture_output=True, text=True, timeout=1, stderr=subprocess.DEVNULL)
                                        if result.returncode == 0:
                                            # Parse device name from v4l2-ctl output
                                            for line in result.stdout.split('\n'):
                                                if 'Card type' in line or 'Driver name' in line:
                                                    # Extract camera name
                                                    parts = line.split(':')
                                                    if len(parts) > 1:
                                                        device_name = parts[1].strip()
                                                        if device_name:
                                                            camera_name = f"{device_name} ({idx})"
                                                    break
                                    except:
                                        pass
                                    
                                    # Label camera type - index 0 is often built-in, but not always
                                    # For now, we'll just label it as the index number
                                    if idx == 0:
                                        camera_type = "Camera 0"  # Could be built-in or USB
                                    else:
                                        camera_type = f"USB Camera {idx}"
                                    
                                    # Determine current camera
                                    is_current = False
                                    if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
                                        is_current = (idx == hw_controller.camera_index)
                                    else:
                                        is_current = (idx == CURRENT_CAMERA_INDEX)
                                    
                                    cameras.append({
                                        'index': idx,
                                        'name': camera_name,
                                        'type': camera_type,
                                        'backend': 'V4L2',
                                        'resolution': f'{width}x{height}',
                                        'current': is_current
                                    })
                                
                                test_cap.release()
                        except Exception as e:
                            # Camera doesn't work or can't be opened - silently skip
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
        else:
            # Use cached list but update current status
            cameras = CAMERA_LIST_CACHE or []
            for cam in cameras:
                if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
                    cam['current'] = (cam['index'] == hw_controller.camera_index)
                else:
                    cam['current'] = (cam['index'] == CURRENT_CAMERA_INDEX)
        
        # Determine current camera index
        if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
            current = hw_controller.camera_index
        else:
            current = CURRENT_CAMERA_INDEX
        
        # Cache the result
        CAMERA_LIST_CACHE = cameras
        LAST_CAMERA_SCAN_TIME = time.time()
        
        return jsonify({
            'success': True, 
            'cameras': cameras, 
            'current': current
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def api_stop_camera():
    """API endpoint to stop camera feed"""
    try:
        if HARDWARE_AVAILABLE and hw_controller:
            hw_controller.stop_camera_feed()
            return jsonify({'success': True, 'message': 'Camera feed stopped'})
        else:
            return jsonify({'success': True, 'message': 'Camera feed stopped (no hardware controller)'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/switch', methods=['POST'])
def api_switch_camera():
    """API endpoint to switch to a different camera"""
    global CURRENT_CAMERA_INDEX
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        camera_index = data.get('index')
        
        if camera_index is None:
            return jsonify({'success': False, 'message': 'Camera index required'}), 400
        
        # Ensure camera_index is an integer
        try:
            camera_index = int(camera_index)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': f'Invalid camera index: {camera_index}'}), 400
            
        success = False
        error_message = None
        
        print(f"[CAMERA_SWITCH] Attempting to switch to camera {camera_index}")
        
        if HARDWARE_AVAILABLE and hw_controller:
            try:
                if hasattr(hw_controller, 'switch_camera'):
                    if hw_controller.switch_camera(camera_index):
                        success = True
                        print(f"[CAMERA_SWITCH] Hardware controller switched to camera {camera_index}")
                    else:
                        error_message = f"Hardware controller failed to switch to camera {camera_index}"
                        print(f"[CAMERA_SWITCH] Hardware controller switch returned False")
                else:
                    print(f"[CAMERA_SWITCH] Hardware controller does not have switch_camera method")
                    # Fall through to fallback method
                    raise AttributeError("switch_camera method not available")
            except AttributeError:
                # Fall through to fallback method
                pass
            except Exception as e:
                error_message = f"Error switching camera: {str(e)}"
                print(f"[CAMERA_SWITCH] Hardware controller error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: Just update the index for the generator
        if not success:
            try:
                print(f"[CAMERA_SWITCH] Using fallback method to verify camera {camera_index}")
                import cv2
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        CURRENT_CAMERA_INDEX = camera_index
                        success = True
                        print(f"[CAMERA_SWITCH] Successfully switched to camera {camera_index} (fallback)")
                    else:
                        error_message = f"Camera {camera_index} opened but could not read frame"
                        print(f"[CAMERA_SWITCH] Camera {camera_index} opened but read failed")
                    cap.release()
                else:
                    error_message = f"Could not open camera {camera_index}"
                    print(f"[CAMERA_SWITCH] Could not open camera {camera_index}")
            except Exception as e:
                error_message = f"Error in fallback camera switch: {str(e)}"
                print(f"[CAMERA_SWITCH] Fallback error: {e}")
                import traceback
                traceback.print_exc()
        
        if success:
            # Save camera preference to file
            try:
                camera_pref_file = 'camera_preference.json'
                with open(camera_pref_file, 'w') as f:
                    json.dump({'camera_index': camera_index, 'timestamp': datetime.now().isoformat()}, f)
            except Exception as e:
                print(f"Warning: Could not save camera preference: {e}")
            
            return jsonify({
                'success': True, 
                'message': f'Switched to camera {camera_index}', 
                'current': camera_index
            })
        else:
            error_msg = error_message or f'Failed to switch to camera {camera_index}'
            print(f"[CAMERA_SWITCH] Switch failed: {error_msg}")
            return jsonify({'success': False, 'message': error_msg}), 500
    except Exception as e:
        print(f"[CAMERA_SWITCH] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/capture', methods=['POST'])
def api_capture_image():
    """API endpoint to capture image"""
    if HARDWARE_AVAILABLE and hw_controller:
        img = hw_controller.get_frame()
        if img is not None:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join('temp', filename)
            os.makedirs('temp', exist_ok=True)
            # Check if image write succeeded
            if cv2.imwrite(filepath, img):
                return jsonify({'success': True, 'message': f'Image captured: {filename}', 'filepath': filepath})
            else:
                return jsonify({'success': False, 'message': 'Failed to save image'}), 500
    return jsonify({'success': False, 'message': 'Camera not available'})

@app.route('/api/detection/run', methods=['POST'])
def api_run_detection():
    """API endpoint to run detection on current frame"""
    system_state['detection_count'] += 1
    system_state['last_detection'] = datetime.now().strftime("%H:%M:%S")
    
    # Log the detection
    detection_data = {
        'id': system_state['detection_count'],
        'class': 'ripe' if system_state['detection_count'] % 2 == 0 else 'unripe',
        'confidence': 0.95
    }
    log_detection(detection_data)
    
    return jsonify({'success': True, 'message': 'Detection completed', 'data': detection_data})

@app.route('/api/system/info')
def api_system_info():
    """Get system information"""
    try:
        # Get IP address
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
            
        # Get system stats
        uptime = datetime.now().strftime("%H:%M:%S")
        
        info = {
            'ip': ip,
            'uptime': uptime,
            'hardware_available': HARDWARE_AVAILABLE
        }
        
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            info.update({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory': f"{memory.percent}%"
            })
            
            # Try to get temperature (Raspberry Pi)
            temp = "N/A"
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_c = int(f.read()) / 1000.0
                    temp = f"{temp_c:.1f}°C"
            except:
                pass
            info['temperature'] = temp
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/pi/logs')
def get_logs():
    """Get system logs"""
    try:
        with open('pi_controller.log', 'r') as f:
            logs = f.readlines()[-50:]  # Last 50 lines
        return jsonify({'logs': logs})
    except FileNotFoundError:
        return jsonify({'logs': ['No logs available']})

@app.route('/pi/config')
def get_config():
    """Get system configuration"""
    try:
        with open('pi_config.yaml', 'r') as f:
            config = f.read()
        return jsonify({'config': config})
    except FileNotFoundError:
        return jsonify({'config': 'No config file found'})

@app.route('/pi/config', methods=['POST'])
def update_config():
    """Update system configuration"""
    try:
        config_data = request.json
        # Update configuration file
        return jsonify({'status': 'updated', 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==========================================
# WebSocket Events
# ==========================================

@socketio.on('connect', namespace='/arduino')
def handle_arduino_connect():
    print('🔌 Arduino connected to WebSocket')
    arduino_clients.add(request.sid)
    system_state['arduino_connected'] = True
    emit('status', {'msg': 'Connected to Server'})

@socketio.on('disconnect', namespace='/arduino')
def handle_arduino_disconnect():
    print('🔌 Arduino disconnected')
    arduino_clients.discard(request.sid)
    if not arduino_clients:
        system_state['arduino_connected'] = False

@socketio.on('telemetry', namespace='/arduino')
def handle_telemetry(data):
    """Handle telemetry from Arduino"""
    # Broadcast to frontend
    socketio.emit('telemetry_update', data)
    
    # Update system state
    if 'status' in data:
        system_state['arduino_status'] = data['status']
    if 'battery_voltage' in data:
        system_state['battery'] = data['battery_voltage']

@socketio.on('pick_result', namespace='/arduino')
def handle_pick_result(data):
    """Handle pick result from Arduino"""
    print(f"✅ Pick Result: {data}")
    socketio.emit('pick_complete', data, namespace='/arduino')
    log_detection(data) # Log result

# Modern Controller WebSocket Handlers (no namespace - default)
# Note: These handlers work alongside the /arduino namespace handlers
@socketio.on('connect')
def handle_controller_connect(auth=None):
    """Handle client connection for modern controller"""
    print(f'✅ Client connected to modern controller (SID: {request.sid})')
    emit('status', {'message': 'Connected to server', 'connected': True})

@socketio.on('disconnect')
def handle_controller_disconnect():
    """Handle client disconnection"""
    print(f'❌ Client disconnected from modern controller (SID: {request.sid})')

@socketio.on('servo_command')
def handle_servo_command(data):
    """Handle servo commands from modern controller"""
    print(f'📨 Received servo command: {data}')
    try:
        cmd = data.get('cmd', '').lower()
        
        if cmd == 'connect':
            # Try to re-initialize hardware controller if it's missing
            global hw_controller, HARDWARE_AVAILABLE
            if not HARDWARE_AVAILABLE or hw_controller is None:
                try:
                    from hardware_controller import HardwareController
                    print("Attempting to re-initialize hardware controller...")
                    hw_controller = HardwareController(connection_type='auto', ble_device_name="FarmBot")
                    HARDWARE_AVAILABLE = True
                    print("Hardware controller re-initialized successfully")
                except Exception as e:
                    print(f"Failed to re-initialize hardware controller: {e}")
            
            # Initialize connection and check hardware status
            if HARDWARE_AVAILABLE and hw_controller:
                # Try to connect if not connected
                if not hw_controller.arduino_connected:
                    hw_controller.connect_hardware()
                    
                # Get actual connection status
                status = hw_controller.get_status()
                arduino_connected = status.get('arduino_connected', False)
                connection_type = status.get('connection_type', 'none')
                
                if arduino_connected:
                    emit('status', {
                        'message': f'Connected via {connection_type}',
                        'connected': True
                    })
                    emit('telemetry', {
                        'status': 'connected',
                        'mode': 'auto' if hw_controller.auto_mode else 'manual',
                        'connection_type': connection_type
                    })
                else:
                    emit('status', {
                        'message': 'Hardware controller available but Arduino not connected',
                        'connected': False
                    })
                    emit('telemetry', {
                        'status': 'disconnected',
                        'mode': 'auto' if hw_controller.auto_mode else 'manual'
                    })
            else:
                emit('status', {
                    'message': 'Hardware controller not available',
                    'connected': False
                })
                emit('telemetry', {
                    'status': 'unavailable',
                    'mode': 'manual'
                })
        
        elif cmd == 'disconnect':
            emit('status', {'message': 'Disconnected', 'connected': False})
        
        elif cmd == 'move':
            # Move individual servo with optional dynamic speed
            servo_name = data.get('servo', '').lower()
            angle = data.get('angle', 90)
            speed = data.get('speed', None)  # Optional dynamic speed (0-100%)
            
            # Map servo names to hardware controller methods
            servo_map = {
                'base': 'base',
                'forearm': 'forearm',
                'arm': 'shoulder',  # Arm is shoulder in hardware
                'wrist_yaw': 'elbow',  # Wrist yaw is elbow
                'wrist_pitch': 'pitch',
                'claw': 'claw'
            }
            
            if servo_name in servo_map and HARDWARE_AVAILABLE and hw_controller:
                # Send ANGLE command to Arduino
                servo_index_map = {
                    'base': 0,
                    'shoulder': 1,
                    'forearm': 2,
                    'elbow': 3,
                    'pitch': 4,
                    'claw': 5
                }
                
                hw_servo = servo_map[servo_name]
                if hw_servo in servo_index_map:
                    # Only send SPEED command if speed changed (optimization to reduce lag)
                    # Speed is sent separately and cached to avoid redundant commands
                    # Convert 0-100% to degrees per second: Manual mode max 120, Auto mode 45
                    if speed is not None:
                        # Check if we need to update speed (only if different from last)
                        if not hasattr(hw_controller, '_last_speed') or hw_controller._last_speed != speed:
                            # Determine current mode (default to manual if not set)
                            is_auto_mode = getattr(hw_controller, 'auto_mode', False)
                            
                            # Convert percentage to deg/s based on mode
                            if is_auto_mode:
                                # Auto mode: 0-100% maps to 10-45 deg/s (smooth, controlled)
                                speed_deg_per_sec = int(10 + (speed / 100.0) * 35)  # 10 to 45 deg/s
                            else:
                                # Manual mode: 0-100% maps to 10-120 deg/s (responsive)
                                speed_deg_per_sec = int(10 + (speed / 100.0) * 110)  # 10 to 120 deg/s
                            
                            speed_command = f"SPEED {speed_deg_per_sec}"
                            if hw_controller.arduino_connected:
                                if hw_controller.ble_client and hw_controller.ble_client.connected:
                                    hw_controller.ble_client.send_command(speed_command)
                                elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                                    hw_controller.arduino.write(f"{speed_command}\n".encode())
                            hw_controller._last_speed = speed
                    
                    # Update tracked servo angle in hardware controller
                    hw_controller.update_servo_angle(hw_servo, angle)
                    
                    # Build ANGLE command: ANGLE base shoulder forearm elbow pitch claw
                    # Use -1 for servos we don't want to change
                    angles = [-1, -1, -1, -1, -1, -1]
                    angles[servo_index_map[hw_servo]] = int(angle)
                    
                    command = f"ANGLE {' '.join(str(a) for a in angles)}"
                    if hw_controller.arduino_connected:
                        if hw_controller.ble_client and hw_controller.ble_client.connected:
                            hw_controller.ble_client.send_command(command)
                        elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                            hw_controller.arduino.write(f"{command}\n".encode())
                    
                    # Don't emit status for every command to reduce overhead
                    # emit('status', {'message': f'{servo_name} moved to {angle}°'})
        
        elif cmd == 'start':
            # Start automatic mode
            if HARDWARE_AVAILABLE and hw_controller:
                hw_controller.auto_mode = True
                # Send set_mode command to Arduino to switch to AUTO mode (sets speed to 45 deg/s)
                if hw_controller.arduino_connected:
                    # Send JSON command via BLE/Serial
                    import json
                    mode_command = json.dumps({"cmd": "set_mode", "mode": "AUTO"})
                    if hw_controller.ble_client and hw_controller.ble_client.connected:
                        hw_controller.ble_client.send_command(mode_command)
                    elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                        hw_controller.arduino.write(f"{mode_command}\n".encode())
                
                emit('status', {'message': 'Automatic mode started (Speed set to 45 deg/s)'})
                emit('telemetry', {'mode': 'auto', 'status': 'running'})
        
        elif cmd == 'set_mode':
            # Set manual/auto mode
            mode = data.get('mode', 'manual')
            mode_upper = mode.upper()
            if HARDWARE_AVAILABLE and hw_controller:
                hw_controller.auto_mode = (mode_upper == 'AUTO' or mode_upper == 'AUTOMATIC')
                
                # Send set_mode command to Arduino via BLE/Serial
                if hw_controller.arduino_connected:
                    import json
                    mode_command = json.dumps({"cmd": "set_mode", "mode": mode_upper})
                    if hw_controller.ble_client and hw_controller.ble_client.connected:
                        hw_controller.ble_client.send_command(mode_command)
                    elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                        hw_controller.arduino.write(f"{mode_command}\n".encode())
                
                emit('telemetry', {'mode': mode, 'status': 'idle' if mode_upper == 'MANUAL' else 'running'})
                emit('status', {'message': f'Mode set to {mode_upper}'})
        
        elif cmd == 'set_speed':
            # Set movement speed (0-100%)
            speed = data.get('speed', 50)
            if HARDWARE_AVAILABLE and hw_controller:
                # Store speed percentage for reference
                if hasattr(hw_controller, 'motion_speed'):
                    hw_controller.motion_speed = speed
                
                # Determine current mode (default to manual if not set)
                is_auto_mode = getattr(hw_controller, 'auto_mode', False)
                
                # Convert percentage to deg/s based on mode
                if is_auto_mode:
                    # Auto mode: 0-100% maps to 10-45 deg/s (smooth, controlled)
                    speed_deg_per_sec = int(10 + (speed / 100.0) * 35)  # 10 to 45 deg/s
                else:
                    # Manual mode: 0-100% maps to 10-120 deg/s (responsive)
                    speed_deg_per_sec = int(10 + (speed / 100.0) * 110)  # 10 to 120 deg/s
                
                # Send SPEED command to Arduino if connected
                if hw_controller.arduino_connected:
                    command = f"SPEED {speed_deg_per_sec}"
                    if hw_controller.ble_client and hw_controller.ble_client.connected:
                        hw_controller.ble_client.send_command(command)
                    elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                        hw_controller.arduino.write(f"{command}\n".encode())
                
                emit('status', {'message': f'Speed set to {speed}% ({speed_deg_per_sec} deg/s)'})
        
        elif cmd == 'save':
            # Save current pose to file
            pose = data.get('pose', {})
            try:
                import json
                poses_dir = Path('saved_poses')
                poses_dir.mkdir(exist_ok=True)
                
                # Create pose filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pose_file = poses_dir / f'pose_{timestamp}.json'
                
                pose_data = {
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    'servo_values': pose,
                    'name': data.get('name', f'Pose_{timestamp}')
                }
                
                with open(pose_file, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                emit('status', {
                    'message': f'Pose saved as {pose_file.name}',
                    'pose_file': str(pose_file)
                })
            except Exception as e:
                print(f"Error saving pose: {e}")
                emit('error', {'message': f'Failed to save pose: {str(e)}'})
        
        elif cmd == 'reset':
            # Reset to home position (90° all joints, claw closed)
            if HARDWARE_AVAILABLE and hw_controller:
                if hw_controller.arduino_connected:
                    # Send HOME command to reset all servos
                    if hw_controller.ble_client and hw_controller.ble_client.connected:
                        hw_controller.ble_client.send_command("HOME")
                    elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                        hw_controller.arduino.write(b"HOME\n")
                    
                    # Also send explicit ANGLE command to ensure all servos reset
                    # Format: ANGLE base shoulder forearm elbow pitch claw
                    home_command = "ANGLE 90 90 90 90 90 0"  # All 90° except claw at 0°
                    if hw_controller.ble_client and hw_controller.ble_client.connected:
                        hw_controller.ble_client.send_command(home_command)
                    elif hasattr(hw_controller, 'arduino') and hw_controller.arduino:
                        hw_controller.arduino.write(f"{home_command}\n".encode())
                    
                    emit('status', {'message': 'Arm reset to home position'})
                    emit('telemetry', {'status': 'idle'})
                else:
                    emit('status', {'message': 'Arduino not connected, cannot reset'})
                    emit('error', {'message': 'Arduino not connected'})
            else:
                emit('status', {'message': 'Hardware controller not available'})
                emit('error', {'message': 'Hardware controller not available'})
        
    except Exception as e:
        print(f"Error handling servo command: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})
        emit('status', {'message': f'Error: {str(e)}', 'connected': False})

# Telemetry update thread for modern controller
def controller_telemetry_thread():
    """Background thread to send telemetry updates to modern controller"""
    import time
    while True:
        try:
            if HARDWARE_AVAILABLE and hw_controller:
                # Get hardware status
                hw_status = hw_controller.get_status()
                
                # Get ToF distance if available
                distance = None
                try:
                    distance = hw_controller.get_distance_sensor()
                except:
                    pass
                
                # Determine status
                status = 'idle'
                if hw_controller.auto_mode:
                    status = 'running'
                if not hw_status.get('arduino_connected', False):
                    status = 'disconnected'
                
                # Emit telemetry to all connected clients (no namespace = default)
                telemetry_data = {
                    'distance_mm': distance if distance is not None else None,
                    'status': status,
                    'mode': 'auto' if hw_controller.auto_mode else 'manual',
                    'arduino_connected': hw_status.get('arduino_connected', False),
                    'camera_connected': hw_status.get('camera_connected', False),
                    'connection_type': hw_status.get('connection_type', 'none'),
                    'arm_orientation': hw_status.get('arm_orientation', 'unknown'),
                    'servo_angles': hw_status.get('servo_angles', {})
                }
                # Use app context for emit in background thread
                # Emit to all clients in default namespace
                with app.app_context():
                    socketio.emit('telemetry', telemetry_data)
            else:
                # Emit default telemetry when hardware not available
                default_telemetry = {
                    'distance_mm': None,
                    'status': 'unavailable',
                    'mode': 'manual',
                    'arduino_connected': False,
                    'camera_connected': False,
                    'connection_type': 'none'
                }
                # Use app context for emit in background thread
                # Emit to all clients in default namespace
                with app.app_context():
                    socketio.emit('telemetry', default_telemetry)
            
            time.sleep(0.5)  # Update every 500ms
        except Exception as e:
            print(f"Controller telemetry thread error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# Start telemetry thread for modern controller
controller_telemetry_thread_instance = threading.Thread(target=controller_telemetry_thread, daemon=True)
controller_telemetry_thread_instance.start()

@socketio.on('yolo_detection', namespace='/arduino')
def handle_yolo_detection(data):
    """Handle YOLO detection from detection service"""
    print(f"🎯 YOLO Detection: {data}")
    
    # Convert detection to pick command
    bbox = data.get('bbox', {})
    pixel_x = bbox.get('x', 320)
    pixel_y = bbox.get('y', 240)
    class_type = data.get('class', 'ripe')
    confidence = data.get('confidence', 0.0)
    detection_id = data.get('id', f"det_{int(time.time() * 1000)}")
    
    # Log detection
    log_detection({
        'id': detection_id,
        'class': class_type,
        'confidence': confidence
    })
    
    # Check if system is in AUTO mode and ready
    if system_state.get('auto_mode', False) and system_state.get('arduino_connected', False):
        # Send pick command to Arduino
        pick_command = {
            "cmd": "pick",
            "id": detection_id,
            "x": pixel_x,
            "y": pixel_y,
            "class": class_type,
            "confidence": confidence
        }
        
        # Emit to Arduino via WebSocket (Socket.IO format)
        socketio.emit('command', pick_command, namespace='/arduino')
        print(f"📤 Pick command sent to Arduino: {detection_id}")
    else:
        mode_status = "MANUAL" if not system_state.get('auto_mode', False) else "AUTO"
        conn_status = "disconnected" if not system_state.get('arduino_connected', False) else "connected"
        print(f"⏸️  Detection received but system in {mode_status} mode or Arduino {conn_status}")

@app.route('/api/vision/detection', methods=['POST'])
def api_vision_detection():
    """REST endpoint for YOLO detection service (HTTP fallback)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Emit via WebSocket to handle it
        socketio.emit('yolo_detection', data, namespace='/arduino')
        
        return jsonify({'success': True, 'message': 'Detection received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# Main Dashboard Routes
# ==========================================

@app.route('/')
def index():
    """Main dashboard - unified for both training and Pi control"""
    datasets = get_datasets()
    models = get_models()
    
    # Check if we should show Pi dashboard or training dashboard
    # For now, show training dashboard as default
    return render_template('index.html', 
                         datasets=datasets, 
                         models=models,
                         training_status=training_status,
                         hardware_available=HARDWARE_AVAILABLE,
                         system_state=system_state)

@app.route('/pi/dashboard')
def pi_dashboard():
    """Raspberry Pi specific dashboard"""
    return render_template('pi_dashboard.html')

@app.route('/control')
def control():
    """Robotic arm control interface with 3D visualization"""
    return render_template('arm_control.html')

@app.route('/arm-control')
def arm_control():
    """Robotic arm control interface with 3D visualization (backward compatibility)"""
    return render_template('arm_control.html')

# Removed /controller route - consolidated to /control and /arm-control using arm_control.html


@app.route('/monitor')
def monitor():
    """Monitoring panel"""
    return render_template('pi_monitor.html')

@app.route('/calibrate')
def calibrate():
    """Calibration panel"""
    return render_template('pi_calibrate.html')

@app.route('/api/socketio/status', methods=['GET'])
def api_socketio_status():
    """Check SocketIO server status"""
    return jsonify({
        'available': SOCKETIO_AVAILABLE,
        'message': 'SocketIO is available' if SOCKETIO_AVAILABLE else 'SocketIO is not installed. Install with: pip install flask-socketio eventlet'
    })

# ==========================================
# Calibration API Routes
# ==========================================

@app.route('/api/calibration/calculate', methods=['POST'])
def api_calibration_calculate():
    """Calculate calibration transformation matrix from points"""
    try:
        data = request.get_json()
        if not data or 'points' not in data:
            return jsonify({'success': False, 'message': 'No points provided'}), 400
        
        points = data['points']
        if len(points) < 4:
            return jsonify({'success': False, 'message': 'Need at least 4 calibration points'}), 400
        
        # Extract pixel and world coordinates
        pixel_coords = np.array([p['pixel'] for p in points], dtype=np.float32)
        world_coords = np.array([p['world'] for p in points], dtype=np.float32)
        
        # Calculate homography matrix using OpenCV
        homography_matrix, mask = cv2.findHomography(pixel_coords, world_coords, 
                                                      cv2.RANSAC, 5.0)
        
        if homography_matrix is None:
            return jsonify({'success': False, 'message': 'Failed to calculate transformation matrix'}), 500
        
        # Calculate accuracy (reprojection error)
        errors = []
        for i, pixel in enumerate(pixel_coords):
            # Transform pixel to world
            pixel_homogeneous = np.array([pixel[0], pixel[1], 1.0])
            predicted_world = homography_matrix @ pixel_homogeneous
            predicted_world = predicted_world[:2] / predicted_world[2]
            
            # Calculate error
            error = np.linalg.norm(predicted_world - world_coords[i])
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        calibration_data = {
            'matrix': homography_matrix.tolist(),
            'point_count': len(points),
            'accuracy': float(avg_error),
            'max_error': float(max_error),
            'errors': [float(e) for e in errors],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': calibration_data,
            'message': f'Calibration calculated with {len(points)} points (avg error: {avg_error:.2f}mm)'
        })
        
    except Exception as e:
        print(f"Error calculating calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/calibration/save', methods=['POST'])
def api_calibration_save():
    """Save calibration data to file"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Save using hardware controller if available
        if HARDWARE_AVAILABLE and hw_controller:
            points = data.get('points', [])
            if points:
                success = hw_controller.update_calibration(points)
                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Calibration saved successfully'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to save calibration via hardware controller'
                    }), 500
        
        # Fallback: Save to JSON file
        calibration_file = 'calibration_data.json'
        calibration_data = {
            'points': data.get('points', []),
            'calibration': data.get('calibration', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Also save as numpy format if calibration matrix exists
        if 'calibration' in data and data['calibration'] and 'matrix' in data['calibration']:
            try:
                matrix = np.array(data['calibration']['matrix'])
                pixel_coords = np.array([p['pixel'] for p in data.get('points', [])], dtype=np.float32)
                world_coords = np.array([p['world'] for p in data.get('points', [])], dtype=np.float32)
                
                np.savez('calibration.npz',
                        homography=matrix,
                        pixel_coords=pixel_coords,
                        world_coords=world_coords)
            except Exception as e:
                print(f"Warning: Could not save numpy format: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Calibration saved successfully'
        })
        
    except Exception as e:
        print(f"Error saving calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/calibration/load', methods=['GET'])
def api_calibration_load():
    """Load calibration data from file"""
    try:
        # Try to load from JSON file first
        calibration_file = 'calibration_data.json'
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                return jsonify({
                    'success': True,
                    'data': data,
                    'message': 'Calibration loaded successfully'
                })
        
        # Try to load from numpy format
        if os.path.exists('calibration.npz'):
            calib_data = np.load('calibration.npz')
            matrix = calib_data['homography']
            pixel_coords = calib_data['pixel_coords']
            world_coords = calib_data['world_coords']
            
            # Reconstruct points
            points = []
            for i in range(len(pixel_coords)):
                points.append({
                    'pixel': pixel_coords[i].tolist(),
                    'world': world_coords[i].tolist()
                })
            
            return jsonify({
                'success': True,
                'data': {
                    'points': points,
                    'calibration': {
                        'matrix': matrix.tolist(),
                        'point_count': len(points)
                    }
                },
                'message': 'Calibration loaded from numpy format'
            })
        
        return jsonify({
            'success': False,
            'message': 'No calibration file found'
        })
        
    except Exception as e:
        print(f"Error loading calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/calibration/test', methods=['POST'])
def api_calibration_test():
    """Test calibration by converting pixel to world coordinates"""
    try:
        data = request.get_json()
        if not data or 'pixel' not in data:
            return jsonify({'success': False, 'message': 'No pixel coordinates provided'}), 400
        
        pixel = np.array(data['pixel'], dtype=np.float32)
        
        # Load calibration
        calibration_file = 'calibration_data.json'
        if not os.path.exists(calibration_file):
            if os.path.exists('calibration.npz'):
                calib_data = np.load('calibration.npz')
                matrix = calib_data['homography']
            else:
                return jsonify({'success': False, 'message': 'No calibration found'}), 404
        else:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
                if 'calibration' not in calib_data or 'matrix' not in calib_data['calibration']:
                    return jsonify({'success': False, 'message': 'Invalid calibration data'}), 400
                matrix = np.array(calib_data['calibration']['matrix'])
        
        # Transform pixel to world coordinates
        pixel_homogeneous = np.array([pixel[0], pixel[1], 1.0])
        predicted_world = matrix @ pixel_homogeneous
        predicted_world = predicted_world[:2] / predicted_world[2]
        
        # Calculate error if actual position provided
        error = None
        actual = None
        if 'actual' in data:
            actual = np.array(data['actual'], dtype=np.float32)
            error = float(np.linalg.norm(predicted_world - actual))
        
        return jsonify({
            'success': True,
            'pixel': pixel.tolist(),
            'predicted': predicted_world.tolist(),
            'actual': actual.tolist() if actual is not None else None,
            'error': error if error is not None else 0.0
        })
        
    except Exception as e:
        print(f"Error testing calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/training')
def training_dashboard():
    """Training dashboard"""
    datasets = get_datasets()
    models = get_models()
    return render_template('training_dashboard.html', 
                         datasets=datasets, 
                         models=models,
                         training_status=training_status)

@app.route('/create_dataset', methods=['GET', 'POST'])
def create_dataset():
    """Create a new dataset"""
    if request.method == 'POST':
        crop_name = request.form.get('crop_name')
        classes = request.form.getlist('classes')
        
        if not crop_name or not classes:
            flash('Please provide crop name and at least one class', 'error')
            return redirect(url_for('create_dataset'))
        
        # Create dataset structure
        dataset_path = os.path.join(UPLOAD_FOLDER, crop_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            # Create README
            readme_content = f"""# {class_name.title()} {crop_name.title()}

Place your {class_name} {crop_name} images in this folder.

Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff
Recommended: At least 50-100 images per class for good results
"""
            with open(os.path.join(class_path, 'README.md'), 'w') as f:
                f.write(readme_content)
        
        flash(f'Dataset "{crop_name}" created successfully!', 'success')
        return redirect(url_for('manage_dataset', dataset_name=crop_name))
    
    return render_template('create_dataset.html')

@app.route('/manage_dataset/<dataset_name>')
def manage_dataset(dataset_name):
    """Manage a specific dataset"""
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    
    if not os.path.exists(dataset_path):
        flash(f'Dataset "{dataset_name}" not found', 'error')
        return redirect(url_for('index'))
    
    # Get class information
    classes = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
            # Count images in this class
            image_count = 0
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                        image_count += 1
            
            classes.append({
                'name': item,
                'path': item_path,
                'image_count': image_count
            })
    
    return render_template('manage_dataset.html', 
                         dataset_name=dataset_name, 
                         classes=classes)

@app.route('/upload_images/<dataset_name>/<class_name>', methods=['POST'])
def upload_images(dataset_name, class_name):
    """Upload images to a specific class"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    class_path = os.path.join(UPLOAD_FOLDER, dataset_name, class_name)
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(class_path, filename)
            
            # Handle duplicate filenames
            counter = 1
            original_filename = filename
            while os.path.exists(file_path):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                file_path = os.path.join(class_path, filename)
                counter += 1
            
            try:
                file.save(file_path)
                uploaded_count += 1
            except Exception as e:
                errors.append(f"Error saving {file.filename}: {str(e)}")
        else:
            errors.append(f"Invalid file: {file.filename}")
    
    return jsonify({
        'success': True,
        'uploaded_count': uploaded_count,
        'errors': errors
    })

@app.route('/delete_image/<dataset_name>/<class_name>/<filename>', methods=['DELETE'])
def delete_image(dataset_name, class_name, filename):
    """Delete a specific image"""
    file_path = os.path.join(UPLOAD_FOLDER, dataset_name, class_name, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/start_training/<dataset_name>', methods=['POST'])
def start_training(dataset_name):
    """Start training a model for a dataset"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    # Get training parameters
    epochs = int(request.form.get('epochs', 30))
    batch_size = int(request.form.get('batch_size', 32))
    learning_rate = float(request.form.get('learning_rate', 0.001))
    
    # Start training in background thread
    def train_model():
        global training_status
        training_status['is_training'] = True
        training_status['current_crop'] = dataset_name
        training_status['progress'] = 0
        training_status['status_message'] = 'Starting training...'
        training_status['logs'] = []
        
        try:
            # Use the virtual environment Python
            python_cmd = 'python'
            venv_path = os.path.join(os.getcwd(), 'tomato_sorter_env/bin/python')
            if os.path.exists(venv_path):
                python_cmd = venv_path
            
            # Run the automated training
            cmd = [
                python_cmd, 'auto_train.py',
                '--dataset_path', os.path.join(UPLOAD_FOLDER, dataset_name),
                '--crop_name', dataset_name,
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--learning_rate', str(learning_rate)
            ]
            
            training_status['status_message'] = f'Running: {" ".join(cmd)}'
            training_status['logs'].append(f'Command: {" ".join(cmd)}')
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor training progress
            for line in iter(process.stdout.readline, ''):
                if line:
                    training_status['logs'].append(line.strip())
                    
                    # Extract progress from training output
                    if 'Epoch' in line and '%' in line:
                        try:
                            # Extract epoch number and progress
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Epoch':
                                    epoch_num = int(parts[i+1].split('/')[0])
                                    total_epochs = int(parts[i+1].split('/')[1])
                                    training_status['progress'] = int((epoch_num / total_epochs) * 100)
                                    break
                        except:
                            pass
                    
                    # Update status message
                    if 'Training completed' in line:
                        training_status['status_message'] = 'Training completed successfully!'
                    elif 'Error' in line or 'Failed' in line:
                        training_status['status_message'] = f'Training error: {line.strip()}'
            
            process.wait()
            
            if process.returncode == 0:
                training_status['status_message'] = 'Training completed successfully!'
                training_status['progress'] = 100
            else:
                training_status['status_message'] = f'Training failed with return code: {process.returncode}'
                
        except Exception as e:
            training_status['status_message'] = f'Training error: {str(e)}'
            training_status['logs'].append(f'Error: {str(e)}')
        finally:
            training_status['is_training'] = False
    
    # Start training thread
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for continuous learning"""
    data = request.get_json()
    
    if not all(key in data for key in ['image_path', 'predicted_class', 'correct_class', 'confidence']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Use continuous learning system
        cmd = [
            sys.executable, 'continuous_learning.py',
            '--action', 'feedback',
            '--image', data['image_path'],
            '--predicted', data['predicted_class'],
            '--correct', data['correct_class'],
            '--confidence', str(data['confidence'])
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                if response.get('success'):
                    return jsonify({'success': True, 'message': response.get('message', 'Feedback recorded for continuous learning')})
                else:
                    return jsonify({'success': False, 'error': response.get('error', 'Failed to save feedback')}), 500
            except json.JSONDecodeError:
                # Fallback if output is not JSON
                return jsonify({'success': True, 'message': 'Feedback recorded for continuous learning'})
        else:
            return jsonify({'success': False, 'error': f'Failed to record feedback: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Feedback error: {str(e)}'}), 500

@app.route('/learning_stats')
def get_learning_stats():
    """Get continuous learning statistics"""
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'stats']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse JSON output
            try:
                stats = json.loads(result.stdout.strip())
                return jsonify({'success': True, 'stats': stats})
            except json.JSONDecodeError:
                # Fallback to old parsing method if JSON fails
                stats = {}
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().replace(' ', '_').lower()
                        try:
                            stats[key] = int(value.strip())
                        except:
                            stats[key] = value.strip()
                return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': f'Failed to get stats: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Stats error: {str(e)}'}), 500

@app.route('/prepare_multi_tomato_dataset', methods=['POST'])
def prepare_multi_tomato_dataset():
    """Prepare multi-tomato dataset by cropping individual tomatoes"""
    try:
        data = request.get_json()
        source_dir = data.get('source_dir')
        output_dir = data.get('output_dir')
        split = data.get('split', 'train')
        all_splits = data.get('all_splits', False)
        
        if not source_dir or not output_dir:
            return jsonify({'success': False, 'error': 'Source and output directories are required'}), 400
        
        # Get the script path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'prepare_multi_tomato_dataset.py')
        
        # Build command
        cmd = [sys.executable, script_path, '--source', source_dir, '--output', output_dir]
        if all_splits:
            cmd.append('--all-splits')
        else:
            cmd.extend(['--split', split])
        
        # Run in background thread to avoid blocking
        def run_preparation():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
                # Store result in a file or return via status endpoint
                result_file = os.path.join(script_dir, 'dataset_prep_result.json')
                with open(result_file, 'w') as f:
                    json.dump({
                        'success': result.returncode == 0,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    }, f)
            except Exception as e:
                result_file = os.path.join(script_dir, 'dataset_prep_result.json')
                with open(result_file, 'w') as f:
                    json.dump({
                        'success': False,
                        'error': str(e)
                    }, f)
        
        thread = threading.Thread(target=run_preparation)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Dataset preparation started. Check status for progress.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error starting preparation: {str(e)}'}), 500

@app.route('/dataset_prep_status')
def dataset_prep_status():
    """Get status of dataset preparation"""
    try:
        result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_prep_result.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            return jsonify(result)
        else:
            return jsonify({'success': False, 'status': 'not_started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/datasets/list', methods=['GET'])
def list_datasets_api():
    """API endpoint to list available datasets"""
    try:
        datasets_info = []
        for dataset in get_datasets():
            datasets_info.append({
                'name': dataset['name'],
                'path': os.path.join(UPLOAD_FOLDER, dataset['name']),
                'display': dataset['name']
            })
        return jsonify({'success': True, 'datasets': datasets_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Trigger model retraining with continuous learning data"""
    try:
        import os
        # Get the script path relative to web_interface.py location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'continuous_learning.py')
        
        cmd = [sys.executable, script_path, '--action', 'retrain']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            try:
                # Try to parse JSON from stdout
                stdout_lines = result.stdout.strip().split('\n')
                # Find the last JSON line (in case there are warnings)
                json_output = None
                for line in reversed(stdout_lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            json_output = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                
                if json_output:
                    if json_output.get('success'):
                        return jsonify({
                            'success': True, 
                            'message': json_output.get('message', 'Model retraining dataset prepared'),
                            'dataset_path': json_output.get('dataset_path'),
                            'images_copied': json_output.get('images_copied', 0)
                        })
                    else:
                        return jsonify({'success': False, 'error': json_output.get('error', 'Retraining failed')}), 500
                else:
                    # If no JSON found, check stderr for errors
                    if result.stderr:
                        return jsonify({'success': False, 'error': f'Retraining failed: {result.stderr}'}), 500
                    return jsonify({'success': True, 'message': 'Model retraining dataset prepared'})
            except json.JSONDecodeError as e:
                # If output is not JSON, check for errors
                error_msg = result.stderr if result.stderr else str(e)
                return jsonify({'success': False, 'error': f'Failed to parse response: {error_msg}'}), 500
        else:
            # Try to extract error from stderr or stdout
            error_msg = result.stderr if result.stderr else result.stdout
            # Try to parse JSON error from output
            try:
                error_json = json.loads(error_msg.strip().split('\n')[-1])
                if 'error' in error_json:
                    error_msg = error_json['error']
            except:
                pass
            return jsonify({'success': False, 'error': f'Retraining failed: {error_msg}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'success': False, 'error': f'Retraining error: {str(e)}\n{error_details}'}), 500

@app.route('/start_continuous_training', methods=['POST'])
def start_continuous_training():
    """Start actual training using the prepared retraining dataset"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    # Check if retraining dataset exists
    retraining_dataset_dir = os.path.join('learning_data', 'retraining_dataset')
    if not os.path.exists(retraining_dataset_dir):
        return jsonify({'error': 'Retraining dataset not found. Please prepare the dataset first using "Retrain Model" button.'}), 400
    
    # Check if dataset has images
    class_folders = ['unripe', 'ripe', 'spoilt']
    total_images = 0
    for class_folder in class_folders:
        class_path = os.path.join(retraining_dataset_dir, class_folder)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
    
    if total_images == 0:
        return jsonify({'error': 'No images found in retraining dataset. Please prepare the dataset first.'}), 400
    
    # Get training parameters
    epochs = int(request.json.get('epochs', 30)) if request.is_json else int(request.form.get('epochs', 30))
    batch_size = int(request.json.get('batch_size', 32)) if request.is_json else int(request.form.get('batch_size', 32))
    learning_rate = float(request.json.get('learning_rate', 0.001)) if request.is_json else float(request.form.get('learning_rate', 0.001))
    
    # Prepare dataset with train/val splits
    # The training script expects: dataset/train/Unripe/, dataset/train/Ripe/, dataset/train/Old/
    # But retraining dataset has: retraining_dataset/unripe/, retraining_dataset/ripe/, retraining_dataset/spoilt/
    # We need to create a temporary dataset with proper structure
    temp_dataset_dir = os.path.join('learning_data', 'temp_training_dataset')
    
    def prepare_and_train():
        global training_status
        training_status['is_training'] = True
        training_status['current_crop'] = 'continuous_learning'
        training_status['progress'] = 0
        training_status['status_message'] = 'Preparing dataset...'
        training_status['logs'] = []
        
        try:
            import shutil
            import random
            
            # Create temp dataset structure
            os.makedirs(temp_dataset_dir, exist_ok=True)
            train_dir = os.path.join(temp_dataset_dir, 'train')
            val_dir = os.path.join(temp_dataset_dir, 'val')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            # Class mapping: retraining dataset -> training script expected names
            class_mapping = {
                'unripe': 'Unripe',
                'ripe': 'Ripe',
                'spoilt': 'Old'  # Training script uses 'Old' for spoilt
            }
            
            # Split images 80/20 train/val
            val_split = 0.2
            total_copied = 0
            
            for source_class, target_class in class_mapping.items():
                source_dir = os.path.join(retraining_dataset_dir, source_class)
                if not os.path.exists(source_dir):
                    continue
                
                # Get all images
                images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) == 0:
                    continue
                
                # Shuffle and split
                random.shuffle(images)
                val_count = max(1, int(len(images) * val_split))
                val_images = images[:val_count]
                train_images = images[val_count:]
                
                # Create class folders
                train_class_dir = os.path.join(train_dir, target_class)
                val_class_dir = os.path.join(val_dir, target_class)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)
                
                # Copy images
                for img in train_images:
                    shutil.copy2(
                        os.path.join(source_dir, img),
                        os.path.join(train_class_dir, img)
                    )
                    total_copied += 1
                
                for img in val_images:
                    shutil.copy2(
                        os.path.join(source_dir, img),
                        os.path.join(val_class_dir, img)
                    )
                    total_copied += 1
                
                training_status['logs'].append(f'Prepared {len(train_images)} train + {len(val_images)} val images for {target_class}')
            
            if total_copied == 0:
                training_status['status_message'] = 'Error: No images to train with'
                training_status['is_training'] = False
                return
            
            training_status['status_message'] = f'Starting training with {total_copied} images...'
            training_status['logs'].append(f'Total images: {total_copied}')
            
            # Find existing model to continue training from
            pretrained_model_path = None
            # Check common model locations
            model_locations = [
                os.path.join('models', 'tomato', 'best_model.pth'),
                os.path.join('models', 'tomato', 'tomato_classifier.pth'),
                'tomato_classifier.pth',
                os.path.join('models', 'tomato', 'model.pth')
            ]
            
            for model_path in model_locations:
                if os.path.exists(model_path):
                    pretrained_model_path = model_path
                    training_status['logs'].append(f'Found existing model: {model_path}')
                    training_status['logs'].append('Continuing training from existing model (fine-tuning)...')
                    break
            
            if not pretrained_model_path:
                training_status['logs'].append('No existing model found. Training from scratch.')
            
            # Use the virtual environment Python
            python_cmd = 'python'
            venv_path = os.path.join(os.getcwd(), 'tomato_sorter_env/bin/python')
            if os.path.exists(venv_path):
                python_cmd = venv_path
            
            # Run training
            cmd = [
                python_cmd, 'train_tomato_classifier.py',
                '--dataset', temp_dataset_dir,
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--lr', str(learning_rate)
            ]
            
            # Add pretrained model path if found
            if pretrained_model_path:
                cmd.extend(['--resume', pretrained_model_path])
            
            training_status['status_message'] = f'Running: {" ".join(cmd)}'
            training_status['logs'].append(f'Command: {" ".join(cmd)}')
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor training progress
            for line in iter(process.stdout.readline, ''):
                if line:
                    training_status['logs'].append(line.strip())
                    
                    # Extract progress from training output
                    if 'Epoch' in line and '/' in line:
                        try:
                            # Extract epoch number and progress
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Epoch' and i + 1 < len(parts):
                                    epoch_str = parts[i+1]
                                    if '/' in epoch_str:
                                        epoch_num = int(epoch_str.split('/')[0])
                                        total_epochs = int(epoch_str.split('/')[1])
                                        training_status['progress'] = int((epoch_num / total_epochs) * 100)
                                        break
                        except:
                            pass
                    
                    # Update status message
                    if 'Training completed' in line or 'Model saved' in line:
                        training_status['status_message'] = 'Training completed successfully!'
                    elif 'Error' in line or 'Failed' in line:
                        training_status['status_message'] = f'Training error: {line.strip()}'
            
            process.wait()
            
            if process.returncode == 0:
                training_status['status_message'] = 'Training completed successfully!'
                training_status['progress'] = 100
                training_status['logs'].append('Training completed. Model saved to models/tomato/')
                
                # Clear retraining dataset after successful training
                try:
                    retraining_dataset_dir = os.path.join('learning_data', 'retraining_dataset')
                    if os.path.exists(retraining_dataset_dir):
                        # Remove all files in class folders but keep the folder structure
                        for class_folder in ['unripe', 'ripe', 'spoilt']:
                            class_path = os.path.join(retraining_dataset_dir, class_folder)
                            if os.path.exists(class_path):
                                for file in os.listdir(class_path):
                                    file_path = os.path.join(class_path, file)
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                        training_status['logs'].append('Retraining dataset cleared. Ready for new feedback data.')
                except Exception as e:
                    training_status['logs'].append(f'Warning: Could not clear retraining dataset: {str(e)}')
            else:
                training_status['status_message'] = f'Training failed with return code: {process.returncode}'
                
        except Exception as e:
            training_status['status_message'] = f'Training error: {str(e)}'
            training_status['logs'].append(f'Error: {str(e)}')
            import traceback
            training_status['logs'].append(traceback.format_exc())
        finally:
            training_status['is_training'] = False
            # Clean up temp dataset
            try:
                if os.path.exists(temp_dataset_dir):
                    shutil.rmtree(temp_dataset_dir)
            except:
                pass
    
    # Start training thread
    thread = threading.Thread(target=prepare_and_train)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Continuous learning training started'})

@app.route('/clear_retraining_dataset', methods=['POST'])
def clear_retraining_dataset():
    """Clear the retraining dataset to make room for new feedback data"""
    try:
        retraining_dataset_dir = os.path.join('learning_data', 'retraining_dataset')
        
        if not os.path.exists(retraining_dataset_dir):
            return jsonify({'success': True, 'message': 'Retraining dataset does not exist (already empty)'})
        
        # Count images before clearing
        total_images = 0
        for class_folder in ['unripe', 'ripe', 'spoilt']:
            class_path = os.path.join(retraining_dataset_dir, class_folder)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
        
        # Remove all files in class folders but keep the folder structure
        removed_count = 0
        for class_folder in ['unripe', 'ripe', 'spoilt']:
            class_path = os.path.join(retraining_dataset_dir, class_folder)
            if os.path.exists(class_path):
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        removed_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Retraining dataset cleared. Removed {removed_count} images.',
            'images_removed': removed_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/training_curves')
def get_training_curves():
    """Serve training curves image"""
    # Check multiple possible locations
    chart_paths = [
        'training_curves.png',
        os.path.join('models', 'tomato', 'training_curves.png')
    ]
    
    for chart_path in chart_paths:
        if os.path.exists(chart_path):
            return send_file(chart_path, mimetype='image/png')
    
    return jsonify({'error': 'Training curves not found'}), 404

@app.route('/recent_learning_images')
def get_recent_learning_images():
    """Get list of recent test images for continuous learning"""
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'recent_images', '--limit', '20']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.strip())
                return jsonify(data)
            except json.JSONDecodeError:
                return jsonify({'success': False, 'error': 'Failed to parse image list'}), 500
        else:
            return jsonify({'success': False, 'error': f'Failed to get images: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500

@app.route('/learning_image/<path:image_path>')
def serve_learning_image(image_path):
    """Serve learning images from the learning_data directory"""
    try:
        # Security: ensure path is within learning_data directory
        full_path = os.path.join('learning_data', image_path)
        if not os.path.abspath(full_path).startswith(os.path.abspath('learning_data')):
            return jsonify({'error': 'Invalid path'}), 403
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return send_file(full_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_learning_image', methods=['DELETE'])
def delete_learning_image():
    """Delete a learning image and its metadata"""
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'error': 'Missing image_path'}), 400
        
        image_path = data['image_path']
        
        # Security: ensure path is within learning_data directory
        if not os.path.abspath(image_path).startswith(os.path.abspath('learning_data')):
            return jsonify({'success': False, 'error': 'Invalid path'}), 403
        
        # Use continuous learning system to delete
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'delete_image', '--image', image_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                return jsonify(response)
            except json.JSONDecodeError:
                return jsonify({'success': True, 'message': 'Image deleted'})
        else:
            return jsonify({'success': False, 'error': f'Failed to delete image: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500

@app.route('/continuous_learning')
def continuous_learning_page():
    """Continuous learning management page"""
    try:
        return render_template('continuous_learning.html', training_status=training_status)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR in continuous_learning_page: {e}")
        print(error_msg)
        return f"Error rendering template: {str(e)}<br><pre>{error_msg}</pre>", 500

# Multi-tomato detection route removed - not needed for robotic sorting system

@app.route('/camera_feed')
def camera_feed():
    """Live camera feed for monitoring"""
    return render_template('camera_feed.html')

@app.route('/camera_status')
def camera_status():
    """Check if camera is available - tests multiple indices"""
    import cv2
    import os
    
    # Check for video devices
    video_devices = []
    if os.path.exists('/dev'):
        for item in os.listdir('/dev'):
            if item.startswith('video'):
                video_devices.append(f"/dev/{item}")
    video_devices = sorted(video_devices)
    
    # Try multiple camera indices
    working_camera = None
    tested_indices = []
    
    for i in range(5):  # Test indices 0-4
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                tested_indices.append({
                    'index': i,
                    'opened': True,
                    'can_read': ret and frame is not None
                })
                if ret and frame is not None:
                    working_camera = i
                    height, width = frame.shape[:2]
                    cap.release()
                    break
            else:
                tested_indices.append({
                    'index': i,
                    'opened': False,
                    'can_read': False
                })
            cap.release()
        except Exception as e:
            tested_indices.append({
                'index': i,
                'opened': False,
                'can_read': False,
                'error': str(e)
            })
    
    if working_camera is not None:
        return jsonify({
            'available': True,
            'message': f'Camera is working properly at index {working_camera}',
            'camera_index': working_camera,
            'video_devices': video_devices,
            'tested_indices': tested_indices
        })
    else:
        suggestions = [
            'Connect a USB camera or webcam',
            'Check if camera is being used by another application',
            'On Linux: Check permissions with: ls -l /dev/video*',
            'On Linux: Try: sudo chmod 666 /dev/video0',
            'On Raspberry Pi: Enable camera in raspi-config'
        ]
        
        if not video_devices:
            suggestions.insert(0, 'No /dev/video* devices found - camera may not be connected')
        
        return jsonify({
            'available': False,
            'message': 'No working camera found',
            'camera_index': None,
            'video_devices': video_devices,
            'tested_indices': tested_indices,
            'suggestions': suggestions
        })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/feed')
def api_camera_feed():
    """Pi-specific camera feed route using hardware controller with detection"""
    if HARDWARE_AVAILABLE and hw_controller:
        def generate_frames():
            """Video streaming generator function using hardware controller with detection."""
            frame_count = 0
            last_detection_result = (False, 0, [])
            try:
                while True:
                    # Check if camera is still connected
                    if not hw_controller.camera_connected or not hw_controller.frame_update_running:
                        break
                    
                    frame = hw_controller.get_frame()
                    if frame is None:
                        time.sleep(0.1)  # Wait a bit if no frame
                        continue
                    
                    # Add timestamp (only every 10th frame to reduce CPU)
                    if frame_count % 10 == 0:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        cv2.putText(frame, timestamp, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Only detect tomatoes every 5th frame to reduce CPU usage
                    if frame_count % 5 == 0:
                        last_detection_result = detect_tomatoes_with_boxes(frame)
                    
                    tomato_detected, tomato_count, tomato_boxes = last_detection_result
                    
                    # Draw bounding boxes
                    for i, (x, y, w, h) in enumerate(tomato_boxes):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Tomato {i+1}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add detection status (only update text every 5 frames)
                    if frame_count % 5 == 0:
                        if tomato_detected:
                            cv2.putText(frame, f"TOMATOES DETECTED: {tomato_count}", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "NO TOMATOES DETECTED", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                           
                    # Encode with lower quality for calibration page (faster)
                    # Use higher quality JPEG compression for smaller file size
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ret:
                        continue  # Skip this frame if encoding failed
                    
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    frame_count += 1
                    # Limit to ~10 FPS for calibration page (reduces CPU/network load)
                    time.sleep(0.1)
            except GeneratorExit:
                # Client disconnected, stop gracefully
                pass
            except Exception as e:
                print(f"Camera feed error: {e}")
        
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Fallback to regular video feed
        return video_feed()

def gen_frames():
    """Generate frames from camera - uses hardware controller if available"""
    # Try hardware controller first (for Pi)
    if HARDWARE_AVAILABLE and hw_controller:
        frame_count = 0
        last_detection_result = (False, 0, [])
        
        while True:
            frame = hw_controller.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Add timestamp (only update every 10 frames to reduce CPU)
            if frame_count % 10 == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, timestamp, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Only detect tomatoes every 5th frame to reduce CPU usage
            # This significantly improves performance while maintaining reasonable detection
            if frame_count % 5 == 0:
                last_detection_result = detect_tomatoes_with_boxes(frame)
            
            tomato_detected, tomato_count, tomato_boxes = last_detection_result
            
            # Draw bounding boxes
            for i, (x, y, w, h) in enumerate(tomato_boxes):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tomato {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add detection status (only update text every 5 frames)
            if frame_count % 5 == 0:
                if tomato_detected:
                    cv2.putText(frame, f"TOMATOES DETECTED: {tomato_count}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO TOMATOES DETECTED", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Encode frame with optimized quality
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            time.sleep(0.03)  # ~30 FPS
        return
    
    # Fallback to direct camera access
    global CURRENT_CAMERA_INDEX
    
    # Try to open the selected camera
    cap = cv2.VideoCapture(CURRENT_CAMERA_INDEX)
    if not cap.isOpened():
        # Try default 0 if selected fails
        if CURRENT_CAMERA_INDEX != 0:
            cap = cv2.VideoCapture(0)
            
    if not cap.isOpened():
        # If no camera available, generate a placeholder image
        while True:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (50, 50, 50)
            
            cv2.putText(placeholder, "No Camera Available", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Connect a camera to see live feed", (100, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        return
    
    # Camera is available, stream live video
    frame_count = 0
    last_detection_result = (False, 0, [])
    
    while True:
        success, frame = cap.read()
        if not success:
            # Try to reconnect
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(CURRENT_CAMERA_INDEX)
            if not cap.isOpened() and CURRENT_CAMERA_INDEX != 0:
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                # Show placeholder if reconnection fails
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Disconnected", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue
            continue
        
        # Add timestamp (only update every 10 frames)
        if frame_count % 10 == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Only detect tomatoes every 5th frame to reduce CPU usage
        if frame_count % 5 == 0:
            last_detection_result = detect_tomatoes_with_boxes(frame)
        
        tomato_detected, tomato_count, tomato_boxes = last_detection_result
        
        for i, (x, y, w, h) in enumerate(tomato_boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tomato {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update text only every 5 frames
        if frame_count % 5 == 0:
            if tomato_detected:
                cv2.putText(frame, f"TOMATOES DETECTED: {tomato_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO TOMATOES DETECTED", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Encode frame with optimized quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        
        frame_count += 1
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def detect_tomatoes_in_frame(frame):
    """Detect if tomatoes are present in the frame using computer vision"""
    import cv2
    import numpy as np
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range for tomatoes
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Add green color range for unripe tomatoes
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Add orange/yellow range for overripe tomatoes
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine all masks
    combined_mask = red_mask + green_mask + orange_mask
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours look like tomatoes with stricter criteria
    tomato_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Increased minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Stricter shape analysis
            aspect_ratio = w / h
            if 0.6 < aspect_ratio < 1.6 and w > 40 and h > 40:  # More circular and larger
                # Additional circularity check
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Must be reasonably circular
                        tomato_count += 1
    
    return tomato_count > 0

def detect_tomatoes_with_boxes(frame):
    """Detect tomatoes in the frame and return detection status, count, and bounding boxes"""
    import cv2
    import numpy as np
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range for tomatoes
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Add green color range for unripe tomatoes
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Add orange/yellow range for overripe tomatoes
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine all masks
    combined_mask = red_mask + green_mask + orange_mask
    
    # Apply minimal morphological operations - only opening to remove noise
    # NO CLOSING to avoid merging tomatoes
    kernel_small = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes for tomatoes
    tomato_boxes = []
    tomato_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this bounding box is suspiciously large (likely multiple tomatoes)
            # Typical single tomato: ~100-200 pixels wide, area ~10000-40000
            typical_tomato_area = 30000  # Approximate area for a single tomato
            typical_tomato_width = 150
            
            if area > typical_tomato_area * 2 or w > typical_tomato_width * 2:  # Likely multiple tomatoes
                print(f"[DETECT] Large detection found: {w}x{h}, area={area}, attempting to split...")
                
                # Extract ROI mask for this contour
                roi_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)
                roi_mask = roi_mask[y:y+h, x:x+w]
                
                if roi_mask.size > 0 and roi_mask.sum() > 0:
                    # Use distance transform to find tomato centers
                    dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
                    
                    # Find peaks in distance transform (these are likely tomato centers)
                    # Use a threshold based on the max distance
                    max_dist = dist_transform.max()
                    if max_dist > 10:  # Only if we have substantial distance
                        # Find local maxima using a more aggressive approach
                        # Create markers for watershed
                        _, markers = cv2.connectedComponents(np.uint8(dist_transform > max_dist * 0.3))
                        
                        # If we found multiple components, extract them
                        if markers.max() > 1:  # More than just background
                            for marker_id in range(1, markers.max() + 1):
                                marker_mask = (markers == marker_id).astype(np.uint8) * 255
                                
                                # Find contours in this marker
                                sub_contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for sub_contour in sub_contours:
                                    sub_area = cv2.contourArea(sub_contour)
                                    if sub_area > 500:  # Minimum area for a tomato
                                        sx, sy, sw, sh = cv2.boundingRect(sub_contour)
                                        # Adjust coordinates to full image
                                        sx += x
                                        sy += y
                                        
                                        # Validate size
                                        if sw > 40 and sh > 40 and sw < w and sh < h:
                                            print(f"[DETECT] Split tomato: {sw}x{sh} at ({sx},{sy})")
                                            tomato_boxes.append((sx, sy, sw, sh))
                                            tomato_count += 1
                            
                            if tomato_count > 0:
                                continue  # Skip the original large contour
                    
                    # Fallback: Try simpler approach - find connected components in the ROI
                    num_labels, labels = cv2.connectedComponents(roi_mask)
                    if num_labels > 2:  # More than background + 1 component
                        print(f"[DETECT] Found {num_labels-1} connected components")
                        for label_id in range(1, num_labels):
                            component_mask = (labels == label_id).astype(np.uint8) * 255
                            component_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for comp_contour in component_contours:
                                comp_area = cv2.contourArea(comp_contour)
                                if comp_area > 500:
                                    cx, cy, cw, ch = cv2.boundingRect(comp_contour)
                                    cx += x
                                    cy += y
                                    if cw > 40 and ch > 40:
                                        print(f"[DETECT] Component tomato: {cw}x{ch} at ({cx},{cy})")
                                        tomato_boxes.append((cx, cy, cw, ch))
                                        tomato_count += 1
                        
                        if tomato_count > 0:
                            continue  # Skip the original large contour
            
            # Single tomato detection (for normal-sized detections)
            aspect_ratio = w / h
            if 0.4 < aspect_ratio < 2.5:  # Lenient aspect ratio
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.15:  # Lower circularity threshold
                        if w > 30 and h > 30:  # Minimum size
                            roi = frame[y:y+h, x:x+w]
                            if roi.size > 0:
                                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                mean_saturation = np.mean(hsv_roi[:, :, 1])
                                if mean_saturation > 20:  # Lower saturation threshold
                                    tomato_boxes.append((x, y, w, h))
                                    tomato_count += 1
    
    return tomato_count > 0, tomato_count, tomato_boxes

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture image from camera feed"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file and file.filename:
        # Save captured image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'captured_{timestamp}.jpg'
        filepath = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'message': 'Image captured successfully'
        })
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/tomato_detection_status')
def tomato_detection_status():
    """Get current tomato detection status from camera"""
    import cv2
    
    frame = None
    
    # Try hardware controller first (if available and connected)
    if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
        try:
            frame = hw_controller.get_frame()
            if frame is None:
                return jsonify({
                    'detected': False,
                    'message': 'Camera not available',
                    'tomato_count': 0,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Error getting frame from hardware controller: {e}")
            frame = None
    
    # Fallback to direct camera access
    if frame is None:
        # Use current camera index
        camera_index = CURRENT_CAMERA_INDEX
        if HARDWARE_AVAILABLE and hw_controller and hw_controller.camera_connected:
            camera_index = hw_controller.camera_index
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return jsonify({
                'detected': False,
                'message': f'Camera not available (index {camera_index})',
                'tomato_count': 0,
                'timestamp': datetime.now().isoformat()
            })
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return jsonify({
                'detected': False,
                'message': 'Could not read from camera',
                'tomato_count': 0,
                'timestamp': datetime.now().isoformat()
            })
    
    # Detect tomatoes in the frame using the detection function with boxes
    tomato_detected, tomato_count, tomato_boxes = detect_tomatoes_with_boxes(frame)
    
    return jsonify({
        'detected': tomato_detected,
        'tomato_count': tomato_count,
        'message': f'{"Tomatoes detected" if tomato_detected else "No tomatoes detected"}',
        'timestamp': datetime.now().isoformat()
    })

def count_tomatoes_in_frame(frame):
    """Count the number of tomatoes in the frame"""
    import cv2
    import numpy as np
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range for tomatoes
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Add green color range for unripe tomatoes
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Add orange/yellow range for overripe tomatoes
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine all masks
    combined_mask = red_mask + green_mask + orange_mask
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count tomatoes
    tomato_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Check if it's roughly circular (tomato-like)
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                tomato_count += 1
    
    return tomato_count

@app.route('/test_model/<model_name>', methods=['POST'])
def test_model(model_name):
    """Test a trained model with an uploaded image - handles both single and multi-tomato images"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file and file.filename and allowed_file(file.filename):
        # Save temporary image with original extension
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get original file extension
        original_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        temp_path = os.path.join(temp_dir, f'test_image.{original_ext}')
        file.save(temp_path)
        
        # Load image for processing
        frame = cv2.imread(temp_path)
        if frame is None:
            os.remove(temp_path)
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Try to use hardware controller's classifier if available, otherwise load model directly
        classifier = None
        if HARDWARE_AVAILABLE and hw_controller and hasattr(hw_controller, 'classifier') and hw_controller.classifier:
            classifier = hw_controller.classifier
            print(f"[TEST] Using hardware controller's classifier")
        
        # If no classifier available, try to load model directly
        if not classifier:
            try:
                from models.tomato.tomato_inference import TomatoClassifier
                # Try to find model - first check model_name folder, then default tomato folder
                model_path = None
                model_folder = os.path.join(MODELS_FOLDER, model_name)
                if os.path.exists(model_folder):
                    # Check for best_model.pth in model folder
                    potential_path = os.path.join(model_folder, 'best_model.pth')
                    if os.path.exists(potential_path):
                        model_path = potential_path
                # Fallback to default tomato model
                if not model_path:
                    default_path = os.path.join(MODELS_FOLDER, 'tomato', 'best_model.pth')
                    if os.path.exists(default_path):
                        model_path = default_path
                
                if model_path and os.path.exists(model_path):
                    print(f"[TEST] Loading classifier from: {model_path}")
                    classifier = TomatoClassifier(model_path=model_path)
                    print(f"[TEST] Classifier loaded successfully")
                else:
                    print(f"[TEST] Model file not found. Checked: {model_folder}/best_model.pth and {MODELS_FOLDER}/tomato/best_model.pth")
            except Exception as e:
                print(f"[TEST] Could not load classifier: {e}")
                import traceback
                traceback.print_exc()
        
        results = []
        saved_crops = []
        
        # Use color-based detection to find tomatoes, then classify each one individually
        if classifier:
            try:
                print(f"[TEST] Classifier available, starting detection...")
                # First, detect tomato bounding boxes using color-based detection
                tomato_detected, tomato_count, tomato_boxes = detect_tomatoes_with_boxes(frame)
                
                print(f"[TEST] Detected {tomato_count} tomatoes with bounding boxes: {tomato_boxes}")
                
                if tomato_count == 0:
                    print(f"[TEST] No tomatoes detected via color detection, will try whole image classification")
                
                # If we only got 1 large detection, try a more aggressive splitting approach
                if tomato_count == 1 and len(tomato_boxes) == 1:
                    x, y, w, h = tomato_boxes[0]
                    # Check if it's suspiciously large (likely multiple tomatoes merged)
                    if w > 300 or h > 300 or (w * h) > 100000:
                        print(f"[TEST] Large single detection ({w}x{h}), attempting aggressive splitting...")
                        try:
                            # Re-detect with more aggressive settings
                            # Use the original mask but with different processing
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            # More specific red detection - but keep saturation reasonable
                            lower_red1 = np.array([0, 50, 50])  # Lower saturation to catch all tomatoes
                            upper_red1 = np.array([15, 255, 255])
                            lower_red2 = np.array([165, 50, 50])
                            upper_red2 = np.array([180, 255, 255])
                            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                            red_mask_aggressive = mask1 + mask2
                            
                            # Add green and orange masks too
                            lower_green = np.array([35, 50, 50])
                            upper_green = np.array([85, 255, 255])
                            green_mask_aggressive = cv2.inRange(hsv, lower_green, upper_green)
                            
                            lower_orange = np.array([10, 50, 50])
                            upper_orange = np.array([25, 255, 255])
                            orange_mask_aggressive = cv2.inRange(hsv, lower_orange, upper_orange)
                            
                            combined_mask_aggressive = red_mask_aggressive + green_mask_aggressive + orange_mask_aggressive
                            
                            # Apply minimal opening to remove noise but keep tomatoes separate
                            kernel_tiny = np.ones((3,3), np.uint8)
                            combined_mask_aggressive = cv2.morphologyEx(combined_mask_aggressive, cv2.MORPH_OPEN, kernel_tiny)
                            
                            # Find ALL contours (including small ones that might be individual tomatoes)
                            contours_aggressive, _ = cv2.findContours(combined_mask_aggressive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            print(f"[TEST] Aggressive detection found {len(contours_aggressive)} contours")
                            
                            # Extract individual tomato boxes with better filtering
                            new_boxes = []
                            for i, contour in enumerate(contours_aggressive):
                                area = cv2.contourArea(contour)
                                cx, cy, cw, ch = cv2.boundingRect(contour)
                                
                                # Check if this contour is within the large detection area
                                if not (x <= cx <= x+w and y <= cy <= y+h):
                                    continue  # Skip contours outside the large detection
                                
                                print(f"[TEST] Contour {i}: area={area}, size={cw}x{ch} at ({cx},{cy})")
                                
                                # More conservative area threshold - typical tomato is 10000-40000 pixels
                                if area > 3000:  # Increased from 500 to filter out noise
                                    # Check if it's a reasonable tomato size
                                    if 50 < cw < 350 and 50 < ch < 350:  # More reasonable size range
                                        aspect_ratio = cw / ch
                                        if 0.5 < aspect_ratio < 2.0:  # More reasonable aspect ratio
                                            # Check if this box overlaps significantly with existing boxes
                                            overlaps = False
                                            for existing_box in new_boxes:
                                                ex, ey, ew, eh = existing_box
                                                # Calculate overlap
                                                overlap_x = max(0, min(cx + cw, ex + ew) - max(cx, ex))
                                                overlap_y = max(0, min(cy + ch, ey + eh) - max(cy, ey))
                                                overlap_area = overlap_x * overlap_y
                                                box_area = cw * ch
                                                if overlap_area > box_area * 0.5:  # More than 50% overlap
                                                    overlaps = True
                                                    break
                                            
                                            if not overlaps:
                                                new_boxes.append((cx, cy, cw, ch))
                                                print(f"[TEST] ✓ Added tomato: {cw}x{ch} at ({cx},{cy}), area={area}")
                            
                            if len(new_boxes) > 0:  # Changed from > 1 to > 0 to accept even single detections
                                print(f"[TEST] Aggressive detection found {len(new_boxes)} tomatoes")
                                tomato_boxes = new_boxes
                                tomato_count = len(new_boxes)
                            else:
                                print(f"[TEST] Aggressive detection found 0 valid tomatoes, trying alternative method...")
                                
                                # Try watershed segmentation to separate individual tomatoes
                                print(f"[TEST] Attempting watershed segmentation to separate tomatoes...")
                                roi_mask = np.zeros(combined_mask_aggressive.shape, dtype=np.uint8)
                                roi_mask[y:y+h, x:x+w] = combined_mask_aggressive[y:y+h, x:x+w]
                                
                                watershed_success = False
                                # Use distance transform to find tomato centers
                                dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
                                
                                # Find local maxima (potential tomato centers)
                                # Use a more aggressive threshold to find more centers
                                max_dist = dist_transform.max()
                                if max_dist > 20:
                                    # Find peaks using a lower threshold
                                    peaks_mask = (dist_transform > max_dist * 0.4).astype(np.uint8) * 255
                                    
                                    # Find connected components in peaks (each should be a tomato center)
                                    num_peaks, peak_labels = cv2.connectedComponents(peaks_mask)
                                    print(f"[TEST] Watershed found {num_peaks-1} potential tomato centers")
                                    
                                    if num_peaks > 1:  # More than just background
                                        watershed_boxes = []
                                        for peak_id in range(1, num_peaks):
                                            # Get the region around this peak
                                            peak_mask = (peak_labels == peak_id).astype(np.uint8) * 255
                                            
                                            # Find contours in this peak region
                                            peak_contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            for peak_contour in peak_contours:
                                                peak_area = cv2.contourArea(peak_contour)
                                                if peak_area > 100:  # Valid peak region
                                                    # Expand around the peak to get tomato bounding box
                                                    px, py, pw, ph = cv2.boundingRect(peak_contour)
                                                    # Expand by typical tomato radius
                                                    expand = 75
                                                    px = max(x, px - expand)
                                                    py = max(y, py - expand)
                                                    pw = min(w - (px - x), pw + 2*expand)
                                                    ph = min(h - (py - y), ph + 2*expand)
                                                    
                                                    if pw > 50 and ph > 50 and pw < 300 and ph < 300:
                                                        watershed_boxes.append((px, py, pw, ph))
                                                        print(f"[TEST] Watershed tomato: {pw}x{ph} at ({px},{py})")
                                        
                                        if len(watershed_boxes) > 0:
                                            print(f"[TEST] Watershed segmentation found {len(watershed_boxes)} tomatoes")
                                            tomato_boxes = watershed_boxes
                                            tomato_count = len(watershed_boxes)
                                            watershed_success = True
                                
                                # Only do grid splitting if watershed didn't work
                                if not watershed_success:
                                    # Alternative: Use the large detection but split it into a grid
                                    # Much more conservative estimation: count actual tomato regions in mask
                                    roi_mask_cropped = combined_mask_aggressive[y:y+h, x:x+w]
                                    
                                    # Count connected components in the mask to estimate tomato count
                                    num_components, component_labels = cv2.connectedComponents(roi_mask_cropped)
                                    estimated_from_components = max(1, min(6, num_components - 1))  # Subtract 1 for background
                                    print(f"[TEST] Mask has {num_components-1} connected components")
                                    
                                    # Also use area-based estimation as fallback
                                    avg_tomato_size = 150
                                    spacing_factor = 2.0  # Even more conservative
                                    estimated_from_area = max(2, min(6, int((w * h) / (avg_tomato_size * avg_tomato_size * spacing_factor * 1.5))))
                                    
                                    # Use the minimum of the two estimates
                                    estimated_tomatoes = min(estimated_from_components, estimated_from_area)
                                    print(f"[TEST] Attempting grid-based split into ~{estimated_tomatoes} tomatoes (from {estimated_from_components} components, {estimated_from_area} area-based)")
                                    
                                    # Use a more reasonable grid - aim for roughly square cells
                                    # Calculate optimal grid dimensions
                                    aspect_ratio = w / h
                                    if aspect_ratio > 1:
                                        # Wider than tall
                                        cols = int(np.ceil(np.sqrt(estimated_tomatoes * aspect_ratio)))
                                        rows = int(np.ceil(estimated_tomatoes / cols))
                                    else:
                                        # Taller than wide
                                        rows = int(np.ceil(np.sqrt(estimated_tomatoes / aspect_ratio)))
                                        cols = int(np.ceil(estimated_tomatoes / rows))
                                    
                                    # Ensure we don't create too many cells - very conservative limits
                                    cols = min(cols, 3)  # Max 3 columns (reduced from 4)
                                    rows = min(rows, 3)  # Max 3 rows (reduced from 4)
                                    
                                    cell_w = w // cols
                                    cell_h = h // rows
                                    
                                    grid_boxes = []
                                    for row in range(rows):
                                        for col in range(cols):
                                            gx = x + col * cell_w
                                            gy = y + row * cell_h
                                            gw = cell_w
                                            gh = cell_h
                                            
                                            # Ensure within bounds
                                            gx = max(0, min(gx, frame.shape[1] - gw))
                                            gy = max(0, min(gy, frame.shape[0] - gh))
                                            gw = min(gw, frame.shape[1] - gx)
                                            gh = min(gh, frame.shape[0] - gy)
                                            
                                            # Only add if size is reasonable
                                            if gw > 60 and gh > 60:  # Minimum size for a tomato
                                                # Validate that this cell actually contains tomato pixels
                                                # Extract the cell region from the mask
                                                cell_mask = roi_mask_cropped[row*cell_h:min((row+1)*cell_h, h), 
                                                                             col*cell_w:min((col+1)*cell_w, w)]
                                                
                                                if cell_mask.size > 0:
                                                    # Calculate tomato pixel density in this cell
                                                    tomato_pixels = np.sum(cell_mask > 0)
                                                    cell_area = cell_mask.size
                                                    tomato_density = tomato_pixels / cell_area if cell_area > 0 else 0
                                                    
                                                    # Only add if at least 15% of the cell contains tomato pixels
                                                    # This filters out empty or mostly-empty grid cells
                                                    if tomato_density > 0.15:
                                                        grid_boxes.append((gx, gy, gw, gh))
                                                        print(f"[TEST] Grid cell ({row},{col}): {gw}x{gh} at ({gx},{gy}), density={tomato_density:.2f}")
                                                    
                                    if len(grid_boxes) > 0:
                                        print(f"[TEST] Grid split created {len(grid_boxes)} valid boxes from {rows}x{cols} grid (filtered by tomato density)")
                                        tomato_boxes = grid_boxes
                                        tomato_count = len(grid_boxes)
                                    else:
                                        print(f"[TEST] Grid split found no valid cells with sufficient tomato content")
                        except Exception as e:
                            print(f"[TEST] Error in aggressive splitting: {e}")
                            import traceback
                            traceback.print_exc()
                
                if tomato_count > 0 and tomato_boxes and len(tomato_boxes) > 0:
                    # Classify each detected tomato individually using the classifier
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    learning_dir = os.path.join('learning_data', 'new_images', 'test_uploads')
                    os.makedirs(learning_dir, exist_ok=True)
                    
                    for i, (x, y, w, h) in enumerate(tomato_boxes):
                        # Add padding
                        padding = 15
                        x_padded = max(0, x - padding)
                        y_padded = max(0, y - padding)
                        w_padded = min(frame.shape[1] - x_padded, w + 2 * padding)
                        h_padded = min(frame.shape[0] - y_padded, h + 2 * padding)
                        
                        # Crop the tomato
                        crop = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
                        
                        if crop.size == 0:
                            print(f"[TEST] Skipping empty crop {i+1}")
                            continue
                        
                        # Classify the individual tomato crop
                        try:
                            # Convert BGR to RGB for classification
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            
                            # Perform multiple classifications with slight variations for stability
                            # This helps reduce noise from detection variations
                            classification_results = []
                            confidence_threshold = 0.5  # Increased from 0.3 for more stable results
                            predicted_class = None
                            confidence = None
                            classification_result = None
                            
                            # First classification
                            result1 = classifier.classify_crop(crop_rgb, confidence_threshold=confidence_threshold)
                            if result1:
                                classification_results.append(result1)
                            
                            # Second classification with slight crop variation (if first succeeded)
                            if result1:
                                # Try with slightly different crop (add small random padding variation)
                                h, w = crop_rgb.shape[:2]
                                if h > 20 and w > 20:
                                    # Crop a slightly different region
                                    offset_x = max(0, min(5, w // 20))
                                    offset_y = max(0, min(5, h // 20))
                                    crop_variant = crop_rgb[offset_y:h-offset_y, offset_x:w-offset_x]
                                    if crop_variant.size > 0:
                                        result2 = classifier.classify_crop(crop_variant, confidence_threshold=confidence_threshold)
                                        if result2:
                                            classification_results.append(result2)
                            
                            # Average the results if we have multiple classifications
                            if classification_results:
                                # Count votes for each class
                                class_votes = {}
                                total_confidence = 0
                                
                                for result in classification_results:
                                    cls = result['class']
                                    conf = result['confidence']
                                    if cls not in class_votes:
                                        class_votes[cls] = {'count': 0, 'confidence_sum': 0}
                                    class_votes[cls]['count'] += 1
                                    class_votes[cls]['confidence_sum'] += conf
                                    total_confidence += conf
                                
                                # Get the class with most votes, or highest average confidence if tie
                                best_class = None
                                best_score = -1
                                
                                for cls, data in class_votes.items():
                                    # Score = vote count * average confidence
                                    avg_conf = data['confidence_sum'] / data['count']
                                    score = data['count'] * avg_conf
                                    if score > best_score:
                                        best_score = score
                                        best_class = cls
                                
                                # Use the best class and average confidence
                                predicted_class = best_class
                                avg_confidence = class_votes[best_class]['confidence_sum'] / class_votes[best_class]['count']
                                confidence = avg_confidence
                                
                                print(f"[TEST] Tomato {i+1}: {predicted_class} ({confidence:.2f}) [from {len(classification_results)} classifications]")
                            else:
                                # Try with lower threshold if no results
                                classification_result = classifier.classify_crop(crop_rgb, confidence_threshold=0.3)
                                if classification_result:
                                    predicted_class = classification_result['class']
                                    confidence = classification_result['confidence']
                                    print(f"[TEST] Tomato {i+1}: {predicted_class} ({confidence:.2f}) [low confidence]")
                                else:
                                    print(f"[TEST] Tomato {i+1}: Classification below threshold")
                                    continue
                            
                            # Ensure we have valid classification before proceeding
                            if predicted_class is None or confidence is None:
                                print(f"[TEST] Tomato {i+1}: No valid classification obtained")
                                continue
                            
                            # Save cropped tomato
                            crop_filename = f'test_{timestamp}_tomato_{i+1}.{original_ext}'
                            crop_path = os.path.join(learning_dir, crop_filename)
                            cv2.imwrite(crop_path, crop)
                            saved_crops.append(crop_path)
                            
                            # Save metadata for this crop
                            try:
                                cmd = [
                                    sys.executable, 'continuous_learning.py',
                                    '--action', 'save_metadata',
                                    '--image', crop_path,
                                    '--predicted', predicted_class,
                                    '--confidence', str(confidence)
                                ]
                                subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                            except Exception as e:
                                print(f"Error saving metadata: {e}")
                            
                            # Ensure prediction is a valid string
                            if not predicted_class or not isinstance(predicted_class, str):
                                predicted_class = 'Unknown'
                            
                            results.append({
                                'tomato_number': i + 1,
                                'prediction': predicted_class,
                                'confidence': float(confidence) if confidence is not None else 0.0,
                                'bbox': [x_padded, y_padded, w_padded, h_padded],
                                'crop_path': crop_path
                            })
                        except Exception as e:
                            print(f"[TEST] Error classifying crop {i+1}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                else:
                    print(f"[TEST] No tomatoes detected or empty boxes. Count: {tomato_count}, Boxes: {tomato_boxes}")
            except Exception as e:
                print(f"[TEST] Error in multi-tomato detection/classification: {e}")
                import traceback
                traceback.print_exc()
        
        # If no tomatoes detected, try to classify the whole image as fallback
        if len(results) == 0 and classifier:
            print(f"[TEST] No individual tomatoes detected, trying whole image classification...")
            try:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Classify the whole frame with lower threshold
                classification_result = classifier.classify_crop(frame_rgb, confidence_threshold=0.1)
                
                if classification_result:
                    print(f"[TEST] Whole image classification: {classification_result['class']} ({classification_result['confidence']:.2f})")
                    # Save image for continuous learning
                    learning_image_path = os.path.join('learning_data', 'new_images', 'test_uploads', f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{original_ext}')
                    os.makedirs(os.path.dirname(learning_image_path), exist_ok=True)
                    cv2.imwrite(learning_image_path, frame)
                    
                    # Ensure prediction is a valid string
                    predicted_class = classification_result.get('class', 'Unknown')
                    if not predicted_class or not isinstance(predicted_class, str):
                        predicted_class = 'Unknown'
                    
                    results.append({
                        'prediction': predicted_class,
                        'confidence': float(classification_result.get('confidence', 0.0)) if classification_result.get('confidence') is not None else 0.0,
                        'learning_image_path': learning_image_path,
                        'note': 'Whole image classified (no individual tomatoes detected)'
                    })
                else:
                    print(f"[TEST] Whole image classification below threshold")
            except Exception as e:
                print(f"[TEST] Error in whole image classification: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if len(results) > 0:
            # For single tomato, also include direct prediction/confidence for backward compatibility
            response_data = {
                'success': True,
                'tomato_count': tomato_count,
                'results': results,
                'multi_tomato': tomato_count > 1,
                'continuous_learning': True
            }
            
            # Add direct prediction/confidence for single tomato (for frontend compatibility)
            if len(results) == 1:
                response_data['prediction'] = results[0].get('prediction', 'Unknown') or 'Unknown'
                response_data['confidence'] = results[0].get('confidence', 0.0) or 0.0
            elif len(results) > 1:
                # For multiple tomatoes, use the first one as primary result
                response_data['prediction'] = results[0].get('prediction', 'Unknown') or 'Unknown'
                response_data['confidence'] = results[0].get('confidence', 0.0) or 0.0
            
            return jsonify(response_data)
        else:
            # No results at all - classifier might not be working or image has no tomatoes
            error_msg = 'Could not process image or no tomatoes detected.'
            if not classifier:
                error_msg += ' Classifier not available - model may not be loaded.'
            else:
                error_msg += ' Make sure tomatoes are clearly visible in the frame.'
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'prediction': 'Unknown',
                'confidence': 0.0
            }), 400
    else:
        return jsonify({'error': 'Invalid image file'}), 400

@app.route('/download_model/<model_name>')
def download_model(model_name):
    """Download a trained model"""
    model_path = os.path.join(MODELS_FOLDER, model_name)
    best_model_path = os.path.join(model_path, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        return send_file(best_model_path, as_attachment=True, 
                        download_name=f'{model_name}_model.pth')
    else:
        flash('Model file not found', 'error')
        return redirect(url_for('index'))

@app.route('/delete_dataset/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    """Delete a dataset"""
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

@app.route('/delete_model/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Delete a trained model"""
    model_path = os.path.join(MODELS_FOLDER, model_name)
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Model not found'}), 404

# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    import traceback
    error_msg = traceback.format_exc()
    print(f"ERROR 500: {error}")
    print(error_msg)
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

# Initialize camera list cache on startup (non-blocking)
def init_camera_cache():
    """Initialize camera list cache in background"""
    global CAMERA_LIST_CACHE, LAST_CAMERA_SCAN_TIME
    try:
        if HARDWARE_AVAILABLE and hw_controller and hasattr(hw_controller, 'available_cameras') and hw_controller.available_cameras:
            cameras = []
            for idx in hw_controller.available_cameras:
                cameras.append({
                    'index': idx,
                    'name': f"Camera {idx}",
                    'type': 'Built-in' if idx == 0 else 'USB',
                    'backend': 'V4L2',
                    'resolution': '640x480',
                    'current': idx == (hw_controller.camera_index if hw_controller.camera_connected else None)
                })
            CAMERA_LIST_CACHE = cameras
            LAST_CAMERA_SCAN_TIME = time.time()
            print(f"📷 Camera list cache initialized: {len(cameras)} camera(s)")
    except Exception as e:
        print(f"⚠️  Could not initialize camera cache: {e}")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    os.makedirs('learning_data/new_images/test_uploads', exist_ok=True)
    
    # Initialize monitoring stats
    initialize_stats()
    
    # Load existing stats to sync with system_state
    stats = load_stats()
    system_state['ripe_count'] = stats.get('ripe_count', 0)
    system_state['unripe_count'] = stats.get('unripe_count', 0)
    system_state['spoilt_count'] = stats.get('spoilt_count', 0)
    
    print("🤖 FarmBOT - Unified Web Interface")
    print("=" * 60)
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("💾 Models folder:", MODELS_FOLDER)
    print("📊 Monitoring stats:", STATS_FILE)
    print("📝 Detection log:", LOG_FILE)
    print("🔧 Hardware controller:", "Available" if HARDWARE_AVAILABLE else "Not available")
    print("🌐 Web interface: http://0.0.0.0:5000")
    print("📱 Access from any device on the network")
    print("=" * 60)
    print("Features:")
    print("  ✅ AI Model Training")
    print("  ✅ Dataset Management")
    print("  ✅ Live Camera Feed")
    print("  ✅ Tomato Detection")
    if HARDWARE_AVAILABLE:
        print("  ✅ Hardware Control (Arduino + Camera)")
        print("  ✅ Robotic Arm Control")
    print("  ✅ System Monitoring")
    if SOCKETIO_AVAILABLE:
        print("  ✅ WebSocket Support (Real-time communication)")
    else:
        print("  ⚠️  WebSocket Support (Not available - install flask-socketio)")
    print("=" * 60)
    
    # Initialize camera cache in background thread (non-blocking)
    threading.Thread(target=init_camera_cache, daemon=True).start()
    
    if SOCKETIO_AVAILABLE:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        # Fallback to regular Flask if SocketIO not available
        app.run(host='0.0.0.0', port=5000, debug=False)
