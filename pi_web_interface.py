#!/usr/bin/env python3
"""
Raspberry Pi Web Interface for AI Tomato Sorter
Provides remote control and monitoring via web browser
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import time
from datetime import datetime
import threading
import cv2
import numpy as np
from pathlib import Path

app = Flask(__name__)

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

# Pi-specific routes
@app.route('/pi/status')
def pi_status():
    """Get system status"""
    return jsonify(system_state)

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
    # Send command to Arduino
    return jsonify({'status': 'home', 'message': 'Arm moved to home position'})

@app.route('/pi/control/calibrate')
def start_calibration():
    """Start coordinate calibration"""
    return jsonify({'status': 'calibration', 'message': 'Calibration started'})

@app.route('/pi/camera/feed')
def camera_feed():
    """Get camera feed"""
    # This would stream the camera feed
    # For now, return a placeholder
    return jsonify({'status': 'camera_feed', 'message': 'Camera feed endpoint'})

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
    
    # Send command to Arduino
    return jsonify({'success': True, 'message': f'Arm moved to ({x}, {y}, {z})'})

@app.route('/api/arm/home', methods=['POST'])
def api_home_arm():
    """API endpoint to home arm"""
    # Send home command to Arduino
    return jsonify({'success': True, 'message': 'Arm homed'})

@app.route('/api/camera/capture', methods=['POST'])
def api_capture_image():
    """API endpoint to capture image"""
    # Capture image from camera
    return jsonify({'success': True, 'message': 'Image captured'})

@app.route('/api/detection/run', methods=['POST'])
def api_run_detection():
    """API endpoint to run detection on current frame"""
    # Run AI detection
    return jsonify({'success': True, 'message': 'Detection completed'})

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
