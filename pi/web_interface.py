#!/usr/bin/env python3
"""
AI Tomato Sorter - Web Interface
Flask web interface for monitoring and controlling the tomato sorter
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import base64
import json
import threading
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tomato_sorter_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebInterface:
    def __init__(self, sorter_system):
        self.sorter = sorter_system
        self.is_running = False
        self.stats = {
            'total_detections': 0,
            'not_ready_count': 0,
            'ready_count': 0,
            'spoilt_count': 0,
            'avg_inference_time': 0.0,
            'fps': 0.0
        }
        self.detection_history = []
        
    def start_camera_stream(self):
        """Start camera streaming thread"""
        def generate_frames():
            while self.is_running:
                ret, frame = self.sorter.camera.read()
                if not ret:
                    break
                
                # Process frame
                detections, inference_time = self.sorter.process_frame(frame)
                
                # Update statistics
                self.update_stats(detections, inference_time)
                
                # Draw detections
                frame = self.sorter.draw_detections(frame, detections)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Convert to base64
                frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                
                # Emit to web clients
                socketio.emit('frame', {'image': frame_b64, 'detections': detections})
                
                time.sleep(0.033)  # ~30 FPS
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=generate_frames)
        self.stream_thread.daemon = True
        self.stream_thread.start()
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        self.is_running = False
    
    def update_stats(self, detections, inference_time):
        """Update statistics"""
        self.stats['total_detections'] += len(detections)
        
        for detection in detections:
            class_id = detection['class_id']
            if class_id == 0:
                self.stats['not_ready_count'] += 1
            elif class_id == 1:
                self.stats['ready_count'] += 1
            elif class_id == 2:
                self.stats['spoilt_count'] += 1
        
        # Update inference time
        if hasattr(self.sorter, 'inference_times'):
            if self.sorter.inference_times:
                self.stats['avg_inference_time'] = sum(self.sorter.inference_times) / len(self.sorter.inference_times)
                self.stats['fps'] = 1.0 / self.stats['avg_inference_time'] if self.stats['avg_inference_time'] > 0 else 0
        
        # Add to detection history
        self.detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'inference_time': inference_time
        })
        
        # Keep only last 100 detections
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]

# Global web interface instance
web_interface = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    if web_interface:
        return jsonify(web_interface.stats)
    return jsonify({})

@app.route('/api/detections')
def get_detections():
    """Get detection history"""
    if web_interface:
        return jsonify(web_interface.detection_history[-20:])  # Last 20 detections
    return jsonify([])

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the tomato sorter system"""
    global web_interface
    if web_interface:
        web_interface.start_camera_stream()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'error', 'message': 'System not initialized'})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the tomato sorter system"""
    global web_interface
    if web_interface:
        web_interface.stop_camera_stream()
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'error', 'message': 'System not initialized'})

@app.route('/api/calibrate', methods=['POST'])
def calibrate_system():
    """Calibrate coordinate mapping"""
    data = request.get_json()
    calibration_points = data.get('points', [])
    
    if web_interface and web_interface.sorter:
        web_interface.sorter.coordinate_mapper = CoordinateMapper(calibration_points)
        return jsonify({'status': 'calibrated'})
    
    return jsonify({'status': 'error', 'message': 'System not initialized'})

@app.route('/api/manual_sort', methods=['POST'])
def manual_sort():
    """Manual sorting command"""
    data = request.get_json()
    x = data.get('x', 0)
    y = data.get('y', 0)
    class_id = data.get('class_id', 0)
    
    if web_interface and web_interface.sorter:
        web_interface.sorter.arduino.move_to_position(x, y, class_id)
        return jsonify({'status': 'sent'})
    
    return jsonify({'status': 'error', 'message': 'System not initialized'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to tomato sorter system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_frame')
def handle_frame_request():
    """Handle frame request from client"""
    if web_interface and web_interface.is_running:
        emit('frame_ready', {'status': 'streaming'})
    else:
        emit('frame_ready', {'status': 'stopped'})

def create_html_template():
    """Create HTML template for the web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Tomato Sorter</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        .btn-danger { background-color: #dc3545; color: white; }
        .btn-warning { background-color: #ffc107; color: black; }
        .btn:hover { opacity: 0.8; }
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        #video-stream {
            width: 100%;
            height: auto;
            display: block;
        }
        .stats-panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
        }
        .stat-label {
            font-weight: bold;
        }
        .stat-value {
            color: #007bff;
        }
        .detection-log {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .detection-item {
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
            font-size: 12px;
        }
        .class-not-ready { color: #dc3545; }
        .class-ready { color: #28a745; }
        .class-spoilt { color: #6c757d; }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-running { background-color: #28a745; }
        .status-stopped { background-color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍅 AI Tomato Sorter</h1>
            <p>Real-time tomato detection and sorting system</p>
        </div>
        
        <div class="controls">
            <button class="btn btn-success" onclick="startSystem()">Start System</button>
            <button class="btn btn-danger" onclick="stopSystem()">Stop System</button>
            <button class="btn btn-warning" onclick="calibrateSystem()">Calibrate</button>
            <button class="btn btn-primary" onclick="manualSort()">Manual Sort</button>
        </div>
        
        <div class="main-content">
            <div class="video-container">
                <img id="video-stream" src="" alt="Camera Stream">
                <div id="status" style="position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 5px;">
                    <span class="status-indicator status-stopped"></span>System Stopped
                </div>
            </div>
            
            <div class="stats-panel">
                <h3>Statistics</h3>
                <div id="stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Detections:</span>
                        <span class="stat-value" id="total-detections">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Not Ready:</span>
                        <span class="stat-value" id="not-ready">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Ready:</span>
                        <span class="stat-value" id="ready">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Spoilt:</span>
                        <span class="stat-value" id="spoilt">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Inference:</span>
                        <span class="stat-value" id="avg-inference">0ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">FPS:</span>
                        <span class="stat-value" id="fps">0</span>
                    </div>
                </div>
                
                <h3>Recent Detections</h3>
                <div id="detection-log" class="detection-log">
                    <div class="detection-item">No detections yet</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let isRunning = false;
        
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('frame', function(data) {
            if (data.image) {
                document.getElementById('video-stream').src = 'data:image/jpeg;base64,' + data.image;
            }
            
            if (data.detections) {
                updateDetectionLog(data.detections);
            }
        });
        
        function startSystem() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        isRunning = true;
                        updateStatus('System Running', 'status-running');
                    }
                });
        }
        
        function stopSystem() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'stopped') {
                        isRunning = false;
                        updateStatus('System Stopped', 'status-stopped');
                    }
                });
        }
        
        function calibrateSystem() {
            // Simple calibration - in real implementation, you'd collect calibration points
            const points = [
                {pixel: [100, 100], world: [0, 0]},
                {pixel: [540, 100], world: [100, 0]},
                {pixel: [540, 380], world: [100, 100]},
                {pixel: [100, 380], world: [0, 100]}
            ];
            
            fetch('/api/calibrate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({points: points})
            });
        }
        
        function manualSort() {
            const x = prompt('Enter X coordinate:');
            const y = prompt('Enter Y coordinate:');
            const class_id = prompt('Enter class ID (0=not_ready, 1=ready, 2=spoilt):');
            
            if (x && y && class_id) {
                fetch('/api/manual_sort', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x: parseFloat(x), y: parseFloat(y), class_id: parseInt(class_id)})
                });
            }
        }
        
        function updateStatus(text, className) {
            const statusEl = document.getElementById('status');
            statusEl.innerHTML = `<span class="status-indicator ${className}"></span>${text}`;
        }
        
        function updateDetectionLog(detections) {
            const logEl = document.getElementById('detection-log');
            if (detections.length > 0) {
                const detectionHtml = detections.map(det => 
                    `<div class="detection-item class-${det.class_name.replace('_', '-')}">
                        ${det.class_name} (${(det.confidence * 100).toFixed(1)}%)
                    </div>`
                ).join('');
                logEl.innerHTML = detectionHtml;
            }
        }
        
        // Update statistics periodically
        setInterval(function() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-detections').textContent = data.total_detections || 0;
                    document.getElementById('not-ready').textContent = data.not_ready_count || 0;
                    document.getElementById('ready').textContent = data.ready_count || 0;
                    document.getElementById('spoilt').textContent = data.spoilt_count || 0;
                    document.getElementById('avg-inference').textContent = 
                        data.avg_inference_time ? (data.avg_inference_time * 1000).toFixed(1) + 'ms' : '0ms';
                    document.getElementById('fps').textContent = data.fps ? data.fps.toFixed(1) : '0';
                });
        }, 1000);
    </script>
</body>
</html>
    """
    
    # Create templates directory and save template
    import os
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

def main():
    """Main function to run the web interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tomato Sorter Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create HTML template
    create_html_template()
    
    print("🍅 AI Tomato Sorter - Web Interface")
    print("=" * 50)
    print(f"Starting web interface on http://{args.host}:{args.port}")
    
    # Run the Flask app
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
