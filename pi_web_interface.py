#!/usr/bin/env python3
"""
Raspberry Pi Web Interface for AI Tomato Sorter
Provides remote control and monitoring via web browser
"""

from flask import Flask, render_template, jsonify, request, send_file, Response, redirect, url_for, flash
import json
import os
import io
import csv
import shutil
import subprocess
from werkzeug.utils import secure_filename
import yaml
import time
from datetime import datetime
import threading
import cv2
import numpy as np
from pathlib import Path
import psutil
from hardware_controller import HardwareController

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configuration
UPLOAD_FOLDER = 'datasets'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize Hardware Controller
hw_controller = HardwareController()

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

# Global variables for training status
training_status = {
    'is_training': False,
    'current_crop': None,
    'progress': 0,
    'status_message': '',
    'logs': []
}

LOG_FILE = 'detection_log.csv'

def log_detection(detection_data):
    """Log detection event to CSV"""
    file_exists = os.path.isfile(LOG_FILE)
    
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['timestamp', 'detection_id', 'class', 'confidence'])
            
            # Write data
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                detection_data.get('id', 'unknown'),
                detection_data.get('class', 'unknown'),
                detection_data.get('confidence', 0.0)
            ])
    except Exception as e:
        print(f"Error logging detection: {e}")

# Pi-specific routes
@app.route('/pi/status')
def pi_status():
    """Get system status"""
    # Update system state with hardware status
    hw_status = hw_controller.get_status()
    system_state['camera_connected'] = hw_status['camera_connected']
    system_state['arduino_connected'] = hw_status['arduino_connected']
    system_state['auto_mode'] = hw_status['auto_mode']
    
    return jsonify(system_state)

@app.route('/api/auto/start', methods=['POST'])
def api_start_auto():
    """Enable Auto Mode"""
    hw_controller.start_auto_mode()
    return jsonify({'success': True, 'message': 'Auto Mode Started'})

@app.route('/api/auto/stop', methods=['POST'])
def api_stop_auto():
    """Disable Auto Mode"""
    hw_controller.stop_auto_mode()
    return jsonify({'success': True, 'message': 'Auto Mode Stopped'})

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
    hw_controller.home_arm()
    return jsonify({'status': 'home', 'message': 'Arm moved to home position'})

@app.route('/pi/control/calibrate')
def start_calibration():
    """Start coordinate calibration"""
    return jsonify({'status': 'calibration', 'message': 'Calibration started'})

@app.route('/api/camera/feed')
def generate_frames():
    """Video streaming generator function."""
    while True:
        frame = hw_controller.get_frame()
        if frame is None:
            break
            
        # Add timestamp
        cv2.putText(frame, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # Limit to ~30 FPS

@app.route('/api/camera/feed')
def camera_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

# Main dashboard
@app.route('/')
def dashboard():
    """Main dashboard"""
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

# API endpoints for system control
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
    data = request.json
    x = data.get('x', 0)
    y = data.get('y', 0)
    z = data.get('z', 0)
    
    z = data.get('z', 0)
    
    # Send command to hardware
    hw_controller.move_arm(x, y, z)
    return jsonify({'success': True, 'message': f'Arm moved to ({x}, {y}, {z})'})

@app.route('/api/arm/home', methods=['POST'])
def api_home_arm():
    """API endpoint to home arm"""
    hw_controller.home_arm()
    return jsonify({'success': True, 'message': 'Arm homed'})

@app.route('/api/camera/capture', methods=['POST'])
def api_capture_image():
    """API endpoint to capture image"""
    img = hw_controller.get_frame()
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, img)
    return jsonify({'success': True, 'message': f'Image captured: {filename}'})

@app.route('/api/detection/run', methods=['POST'])
def api_run_detection():
    """API endpoint to run detection on current frame"""
    # Run AI detection
    system_state['detection_count'] += 1
    system_state['last_detection'] = datetime.now().strftime("%H:%M:%S")
    
    # Log the detection (simulated data for now)
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
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
            
        # Get system stats
        uptime = datetime.now().strftime("%H:%M:%S") # Placeholder for real uptime
        
        # Get real system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Try to get temperature
        temp = "N/A"
        try:
            # Try standard Linux thermal zone
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = int(f.read()) / 1000.0
                temp = f"{temp_c:.1f}°C"
        except:
            # Fallback for non-Pi systems
            temp = "N/A"
        
        return jsonify({
            'ip': ip,
            'uptime': uptime,
            'temperature': temp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'memory': f"{memory.percent}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==========================================
# Training Module Integration
# ==========================================

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
                image_count = 0
                class_folders = set()
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                            image_count += 1
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
        
        dataset_path = os.path.join(UPLOAD_FOLDER, crop_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            readme_content = f"# {class_name.title()} {crop_name.title()}\nPlace images here."
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
        return redirect(url_for('training_dashboard'))
    
    classes = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
            image_count = 0
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                        image_count += 1
            classes.append({'name': item, 'path': item_path, 'image_count': image_count})
    
    return render_template('manage_dataset.html', dataset_name=dataset_name, classes=classes)

@app.route('/upload_images/<dataset_name>/<class_name>', methods=['POST'])
def upload_images(dataset_name, class_name):
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
    
    return jsonify({'success': True, 'uploaded_count': uploaded_count, 'errors': errors})

@app.route('/start_training/<dataset_name>', methods=['POST'])
def start_training(dataset_name):
    global training_status
    if training_status['is_training']:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    epochs = int(request.form.get('epochs', 30))
    batch_size = int(request.form.get('batch_size', 32))
    learning_rate = float(request.form.get('learning_rate', 0.001))
    
    def train_model():
        global training_status
        training_status['is_training'] = True
        training_status['current_crop'] = dataset_name
        training_status['progress'] = 0
        training_status['status_message'] = 'Starting training...'
        training_status['logs'] = []
        
        try:
            python_cmd = 'python'
            venv_path = os.path.join(os.getcwd(), 'tomato_sorter_env/bin/python')
            if os.path.exists(venv_path):
                python_cmd = venv_path
            
            cmd = [python_cmd, 'auto_train.py', '--dataset_path', os.path.join(UPLOAD_FOLDER, dataset_name),
                   '--crop_name', dataset_name, '--epochs', str(epochs),
                   '--batch_size', str(batch_size), '--learning_rate', str(learning_rate)]
            
            training_status['status_message'] = f'Running: {" ".join(cmd)}'
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    training_status['logs'].append(line.strip())
                    if 'Epoch' in line and '%' in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Epoch':
                                    epoch_num = int(parts[i+1].split('/')[0])
                                    total_epochs = int(parts[i+1].split('/')[1])
                                    training_status['progress'] = int((epoch_num / total_epochs) * 100)
                                    break
                        except: pass
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
        finally:
            training_status['is_training'] = False
    
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    if not all(key in data for key in ['image_path', 'predicted_class', 'correct_class', 'confidence']):
        return jsonify({'error': 'Missing required fields'}), 400
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'feedback',
               '--image', data['image_path'], '--predicted', data['predicted_class'],
               '--correct', data['correct_class'], '--confidence', str(data['confidence'])]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Feedback recorded'})
        else:
            return jsonify({'error': f'Failed: {result.stderr}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/learning_stats')
def get_learning_stats():
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'stats']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            stats = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    stats[key.strip().replace(' ', '_').lower()] = value.strip()
            return jsonify({'success': True, 'stats': stats})
        return jsonify({'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    try:
        cmd = [sys.executable, 'continuous_learning.py', '--action', 'retrain']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Retraining started'})
        return jsonify({'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/continuous_learning')
def continuous_learning_page():
    return render_template('continuous_learning.html')

@app.route('/test_model/<model_name>', methods=['POST'])
def test_model(model_name):
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file and file.filename and allowed_file(file.filename):
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        original_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        temp_path = os.path.join(temp_dir, f'test_image.{original_ext}')
        file.save(temp_path)
        
        model_path = os.path.join(MODELS_FOLDER, model_name)
        inference_script = os.path.join(model_path, f"{model_name}_inference.py")
        
        if not os.path.exists(inference_script):
            return jsonify({'error': 'Inference script not found'}), 404
            
        try:
            cmd = [sys.executable, inference_script, temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                last_line = output_lines[-1]
                try:
                    prediction_data = json.loads(last_line)
                    return jsonify({'success': True, 'prediction': prediction_data['class'], 'confidence': prediction_data['confidence']})
                except:
                    return jsonify({'success': True, 'prediction': last_line, 'confidence': 1.0})
            return jsonify({'error': result.stderr}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/delete_dataset/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        return jsonify({'success': True})
    return jsonify({'error': 'Not found'}), 404

@app.route('/delete_model/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    model_path = os.path.join(MODELS_FOLDER, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        return jsonify({'success': True})
    return jsonify({'error': 'Not found'}), 404

@app.route('/download_model/<model_name>')
def download_model(model_name):
    model_path = os.path.join(MODELS_FOLDER, model_name)
    zip_path = shutil.make_archive(os.path.join('temp', model_name), 'zip', model_path)
    return send_file(zip_path, as_attachment=True, download_name=f'{model_name}.zip')

if __name__ == '__main__':
    print("🍅 AI Tomato Sorter - Pi Web Interface")
    print("=====================================")
    print("🌐 Web Interface: http://0.0.0.0:5000")
    print("📱 Access from any device on the network")
    print("🔧 Use the web interface to control the system")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
