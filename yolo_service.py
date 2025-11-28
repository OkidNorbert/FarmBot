#!/usr/bin/env python3
"""
YOLO Detection Service for Tomato Picker
=========================================

Runs YOLO inference on camera feed and sends detection events to the web backend.
The web backend then forwards pick commands to Arduino via WebSocket.
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using mock detection.")

try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Warning: python-socketio not available. Using HTTP fallback.")

# Configuration
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
DETECTION_INTERVAL = 0.5   # Seconds between detections
CAMERA_INDEX = 0
MODEL_PATH = "models/tomato/best_model.pth"

# WebSocket/HTTP Configuration
WS_SERVER = "http://localhost:5000"
WS_NAMESPACE = "/arduino"

class YOLODetectionService:
    def __init__(self, model_path=None, camera_index=0):
        self.camera_index = camera_index
        self.model = None
        self.camera = None
        self.running = False
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Initialize Socket.IO client
        if SOCKETIO_AVAILABLE:
            self.sio = socketio.Client()
            self._setup_socketio_handlers()
        else:
            self.sio = None
        
        # Load model if available
        if model_path and TORCH_AVAILABLE:
            self.load_model(model_path)
        elif model_path:
            print(f"Warning: Model path provided but PyTorch not available: {model_path}")
    
    def _setup_socketio_handlers(self):
        """Setup Socket.IO event handlers"""
        @self.sio.on('connect', namespace=WS_NAMESPACE)
        def on_connect():
            print("✅ Connected to WebSocket server")
        
        @self.sio.on('disconnect', namespace=WS_NAMESPACE)
        def on_disconnect():
            print("❌ Disconnected from WebSocket server")
    
    def load_model(self, model_path):
        """Load YOLO model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using mock detection")
            return False
        
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
            
            # Load model (simplified - adjust based on your model architecture)
            from models.tomato.tomato_inference import TomatoClassifier
            self.model = TomatoClassifier(model_path)
            self.model.eval()
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✅ Camera initialized: {self.camera_index}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_tomatoes(self, frame):
        """Run detection on frame"""
        if self.model is None:
            # Mock detection for testing
            return self._mock_detection(frame)
        
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(frame_resized).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Map class index to name
            class_names = ["not_ready", "ready", "spoilt"]
            class_name = class_names[predicted.item()] if predicted.item() < len(class_names) else "unknown"
            conf_value = confidence.item()
            
            # For single-tomato detection, return center of frame as bbox
            h, w = frame.shape[:2]
            bbox = {
                "x": w // 2,
                "y": h // 2,
                "w": 100,
                "h": 100
            }
            
            if conf_value >= CONFIDENCE_THRESHOLD:
                return [{
                    "bbox": bbox,
                    "class": class_name,
                    "confidence": conf_value
                }]
            else:
                return []
        
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _mock_detection(self, frame):
        """Mock detection for testing without model"""
        # Simulate detection in center of frame
        h, w = frame.shape[:2]
        return [{
            "bbox": {"x": w // 2, "y": h // 2, "w": 100, "h": 100},
            "class": "ripe",
            "confidence": 0.85
        }]
    
    def send_detection(self, detection):
        """Send detection to web backend"""
        detection_id = f"det_{int(time.time() * 1000)}"
        
        payload = {
            "event": "detect",
            "bbox": detection["bbox"],
            "class": detection["class"],
            "confidence": detection["confidence"],
            "camera_id": f"cam{self.camera_index}",
            "timestamp": datetime.now().isoformat(),
            "id": detection_id
        }
        
        # Send via WebSocket if available
        if self.sio and self.sio.connected:
            try:
                self.sio.emit('yolo_detection', payload, namespace=WS_NAMESPACE)
                print(f"📤 Detection sent: {detection_id} - {detection['class']} ({detection['confidence']:.2f})")
                return True
            except Exception as e:
                print(f"Error sending via WebSocket: {e}")
                return False
        else:
            # Fallback to HTTP POST
            try:
                import requests
                response = requests.post(f"{WS_SERVER}/api/vision/detection", json=payload, timeout=1)
                if response.status_code == 200:
                    print(f"📤 Detection sent via HTTP: {detection_id}")
                    return True
            except Exception as e:
                print(f"Error sending via HTTP: {e}")
                return False
    
    def connect_to_server(self):
        """Connect to WebSocket server"""
        if not SOCKETIO_AVAILABLE:
            print("Socket.IO not available, using HTTP fallback")
            return False
        
        try:
            self.sio.connect(WS_SERVER, namespaces=[WS_NAMESPACE])
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def run(self):
        """Main detection loop"""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return
        
        # Try to connect to server
        self.connect_to_server()
        
        self.running = True
        print("🎥 YOLO Detection Service Started")
        print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"   Detection Interval: {DETECTION_INTERVAL}s")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Check if enough time has passed
                current_time = time.time()
                if current_time - self.last_detection_time < DETECTION_INTERVAL:
                    time.sleep(0.01)
                    continue
                
                # Run detection
                detections = self.detect_tomatoes(frame)
                
                # Send detections
                for detection in detections:
                    if detection["confidence"] >= CONFIDENCE_THRESHOLD:
                        self.send_detection(detection)
                        self.detection_count += 1
                        self.last_detection_time = current_time
                
                time.sleep(0.01)  # Small delay to prevent CPU spinning
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping YOLO service...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the service"""
        self.running = False
        if self.camera:
            self.camera.release()
        if self.sio and self.sio.connected:
            self.sio.disconnect()
        print(f"✅ YOLO service stopped. Total detections: {self.detection_count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Detection Service for Tomato Picker")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model file")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--interval", type=float, default=DETECTION_INTERVAL, help="Detection interval (seconds)")
    
    args = parser.parse_args()
    
    # Update global config
    global CONFIDENCE_THRESHOLD, DETECTION_INTERVAL, CAMERA_INDEX
    CONFIDENCE_THRESHOLD = args.confidence
    DETECTION_INTERVAL = args.interval
    CAMERA_INDEX = args.camera
    
    # Create and run service
    service = YOLODetectionService(model_path=args.model, camera_index=args.camera)
    service.run()

if __name__ == "__main__":
    main()

