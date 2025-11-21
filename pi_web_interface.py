#!/usr/bin/env python3
"""
Raspberry Pi Web Interface for AI Tomato Sorter
Provides remote control and monitoring via web browser
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import io
import csv
import time
from datetime import datetime
import threading
import cv2
import numpy as np
from pathlib import Path
import psutil
from hardware_controller import HardwareController

app = Flask(__name__)

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
def camera_feed():
    """Get camera feed"""
    img = hw_controller.get_frame()
    
    # Add timestamp
    cv2.putText(img, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='feed.jpg'
    )

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

if __name__ == '__main__':
    print("🍅 AI Tomato Sorter - Pi Web Interface")
    print("=====================================")
    print("🌐 Web Interface: http://0.0.0.0:5000")
    print("📱 Access from any device on the network")
    print("🔧 Use the web interface to control the system")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
