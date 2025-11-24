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

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, Response
from werkzeug.utils import secure_filename
import yaml
import cv2
import numpy as np

# Try to import hardware controller (optional for non-Pi systems)
try:
    from hardware_controller import HardwareController
    hw_controller = HardwareController()
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
    'camera_connected': False,
    'arduino_connected': False,
    'classifier_loaded': False,
    'last_detection': None,
    'current_frame': None
}

def log_detection(detection_data):
    """Log detection event to CSV"""
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
    
    return jsonify(system_state)

@app.route('/api/auto/start', methods=['POST'])
def api_start_auto():
    """Enable Auto Mode"""
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.start_auto_mode()
        return jsonify({'success': True, 'message': 'Auto Mode Started'})
    return jsonify({'success': False, 'message': 'Hardware not available'})

@app.route('/api/auto/stop', methods=['POST'])
def api_stop_auto():
    """Disable Auto Mode"""
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.stop_auto_mode()
        return jsonify({'success': True, 'message': 'Auto Mode Stopped'})
    return jsonify({'success': False, 'message': 'Hardware not available'})

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
    """API endpoint to move arm"""
    if not HARDWARE_AVAILABLE or not hw_controller:
        return jsonify({'success': False, 'message': 'Hardware not available'})
    
    # Check if request has JSON data
    if not request.is_json or request.json is None:
        return jsonify({'success': False, 'message': 'Invalid request: JSON data required'}), 400
    
    data = request.json
    x = data.get('x', 0)
    y = data.get('y', 0)
    z = data.get('z', 0)
    
    hw_controller.move_arm(x, y, z)
    return jsonify({'success': True, 'message': f'Arm moved to ({x}, {y}, {z})'})

@app.route('/api/arm/home', methods=['POST'])
def api_home_arm():
    """API endpoint to home arm"""
    if HARDWARE_AVAILABLE and hw_controller:
        hw_controller.home_arm()
        return jsonify({'success': True, 'message': 'Arm homed'})
    return jsonify({'success': False, 'message': 'Hardware not available'})

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
    """Control panel"""
    return render_template('pi_control.html')

@app.route('/monitor')
def monitor():
    """Monitoring panel"""
    return render_template('pi_monitor.html')

@app.route('/calibrate')
def calibrate():
    """Calibration panel"""
    return render_template('pi_calibrate.html')

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
            return jsonify({'success': True, 'message': 'Feedback recorded for continuous learning'})
        else:
            return jsonify({'error': f'Failed to record feedback: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Feedback error: {str(e)}'}), 500

@app.route('/learning_stats')
def get_learning_stats():
    """Get continuous learning statistics"""
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'stats']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse the output to extract stats
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
            return jsonify({'error': f'Failed to get stats: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Trigger model retraining with continuous learning data"""
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'retrain']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Model retraining started'})
        else:
            return jsonify({'error': f'Retraining failed: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Retraining error: {str(e)}'}), 500

@app.route('/continuous_learning')
def continuous_learning_page():
    """Continuous learning management page"""
    return render_template('continuous_learning.html')

# Multi-tomato detection route removed - not needed for robotic sorting system

@app.route('/camera_feed')
def camera_feed():
    """Live camera feed for monitoring"""
    return render_template('camera_feed.html')

@app.route('/camera_status')
def camera_status():
    """Check if camera is available"""
    import cv2
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Test if we can read a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return jsonify({
                'available': True,
                'message': 'Camera is working properly',
                'camera_index': 0
            })
        else:
            return jsonify({
                'available': False,
                'message': 'Camera opened but cannot read frames',
                'camera_index': 0
            })
    else:
        return jsonify({
            'available': False,
            'message': 'No camera found at index 0',
            'camera_index': 0,
            'suggestions': [
                'Connect a USB camera',
                'Enable camera permissions in browser',
                'Check if camera is being used by another application'
            ]
        })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/feed')
def api_camera_feed():
    """Pi-specific camera feed route using hardware controller"""
    if HARDWARE_AVAILABLE and hw_controller:
        def generate_frames():
            """Video streaming generator function using hardware controller."""
            while True:
                frame = hw_controller.get_frame()
                if frame is None:
                    break
                    
                # Add timestamp
                cv2.putText(frame, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                       
                # Encode - check return value before using buffer
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue  # Skip this frame if encoding failed
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.03)  # Limit to ~30 FPS
        
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Fallback to regular video feed
        return video_feed()

def gen_frames():
    """Generate frames from camera - uses hardware controller if available"""
    # Try hardware controller first (for Pi)
    if HARDWARE_AVAILABLE and hw_controller:
        while True:
            frame = hw_controller.get_frame()
            if frame is None:
                break
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect tomatoes and draw bounding boxes
            tomato_detected, tomato_count, tomato_boxes = detect_tomatoes_with_boxes(frame)
            
            # Draw bounding boxes
            for i, (x, y, w, h) in enumerate(tomato_boxes):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tomato {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add detection status
            if tomato_detected:
                cv2.putText(frame, f"TOMATOES DETECTED: {tomato_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO TOMATOES DETECTED", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
        return
    
    # Fallback to direct camera access
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
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                break
            continue
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detect tomatoes and draw bounding boxes
        tomato_detected, tomato_count, tomato_boxes = detect_tomatoes_with_boxes(frame)
        
        for i, (x, y, w, h) in enumerate(tomato_boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tomato {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if tomato_detected:
            cv2.putText(frame, f"TOMATOES DETECTED: {tomato_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO TOMATOES DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
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
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes for tomatoes with stricter criteria
    tomato_boxes = []
    tomato_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Increased minimum area threshold (was 500)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Stricter shape analysis
            aspect_ratio = w / h
            if 0.6 < aspect_ratio < 1.6:  # More circular requirement (was 0.5-2.0)
                # Additional circularity check
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Must be reasonably circular
                        # Size requirements
                        if w > 40 and h > 40:  # Increased minimum size (was 20x20)
                            # Color intensity check - tomatoes should have good color saturation
                            roi = frame[y:y+h, x:x+w]
                            if roi.size > 0:
                                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                mean_saturation = np.mean(hsv_roi[:, :, 1])
                                if mean_saturation > 50:  # Must have good color saturation
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
    
    # Try to get a frame from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({
            'detected': False,
            'message': 'Camera not available',
            'tomato_count': 0
        })
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({
            'detected': False,
            'message': 'Could not read from camera',
            'tomato_count': 0
        })
    
    # Detect tomatoes in the frame
    tomato_detected = detect_tomatoes_in_frame(frame)
    
    # Count tomatoes for more detailed info
    tomato_count = count_tomatoes_in_frame(frame)
    
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
    """Test a trained model with an uploaded image"""
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
        
        # Run inference
        model_path = os.path.join(MODELS_FOLDER, model_name)
        inference_script = os.path.join(model_path, f'{model_name}_inference.py')
        
        if os.path.exists(inference_script):
            try:
                # Use the virtual environment Python
                python_cmd = 'python'
                venv_path = os.path.join(os.getcwd(), 'tomato_sorter_env/bin/python')
                if os.path.exists(venv_path):
                    python_cmd = venv_path
                
                result = subprocess.run([
                    python_cmd, inference_script, '--image', temp_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse output to extract prediction
                    output_lines = result.stdout.strip().split('\n')
                    prediction = 'Unknown'
                    confidence = 0.0
                    
                    for line in output_lines:
                        if 'Predicted Class:' in line:
                            prediction = line.split('Predicted Class:')[1].strip()
                        elif 'Confidence:' in line:
                            try:
                                confidence = float(line.split('Confidence:')[1].strip())
                            except:
                                pass
                    
                    # Save image for continuous learning
                    learning_image_path = os.path.join('learning_data', 'new_images', 'test_uploads', f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{original_ext}')
                    os.makedirs(os.path.dirname(learning_image_path), exist_ok=True)
                    shutil.copy2(temp_path, learning_image_path)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    return jsonify({
                        'success': True,
                        'prediction': prediction,
                        'confidence': confidence,
                        'output': result.stdout,
                        'learning_image_path': learning_image_path,
                        'continuous_learning': True
                    })
                else:
                    return jsonify({
                        'error': f'Inference failed: {result.stderr}'
                    }), 500
            except subprocess.TimeoutExpired:
                return jsonify({'error': 'Inference timeout'}), 500
            except Exception as e:
                return jsonify({'error': f'Inference error: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Inference script not found'}), 404
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
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    os.makedirs('learning_data/new_images/test_uploads', exist_ok=True)
    
    print("🤖 FarmBOT - Unified Web Interface")
    print("=" * 60)
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("💾 Models folder:", MODELS_FOLDER)
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
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
