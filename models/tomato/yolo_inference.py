#!/usr/bin/env python3
"""
YOLO Inference Module for Tomato Detection and Classification
Handles YOLOv8 model loading and inference with graceful fallback
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

class YOLOTomatoDetector:
    """YOLO-based tomato detector and classifier"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file (.pt)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.available = False
        
        if not YOLO_AVAILABLE:
            print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
            print("   YOLO detection will not be available until ultralytics is installed.")
            return
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.available = True
                print(f"✅ YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load YOLO model: {e}")
                self.available = False
        else:
            print(f"⚠️  YOLO model not found at: {model_path}")
            print("   Train a YOLO model first or provide correct path.")
    
    def detect(self, frame, conf=None):
        """
        Detect and classify tomatoes in a frame
        
        Args:
            frame: OpenCV BGR frame (numpy array)
            conf: Confidence threshold (overrides default if provided)
        
        Returns:
            List of detections, each with:
            {
                'class': str,  # 'not_ready', 'ready', 'spoilt'
                'confidence': float,
                'bbox': [x, y, w, h],  # Bounding box
                'center': [cx, cy]  # Center coordinates
            }
        """
        if not self.available or self.model is None:
            return []
        
        if conf is None:
            conf = self.confidence_threshold
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                conf=conf,
                verbose=False,
                imgsz=640  # Standard YOLO input size
            )
            
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                # Check if we have detections
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get class names from model
                    class_names = result.names
                    
                    for box in result.boxes:
                        # Get bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Convert to (x, y, w, h) format
                        x = x1
                        y = y1
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Map class ID to class name
                        # YOLO classes should match: 0=not_ready, 1=ready, 2=spoilt
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        
                        # Normalize class names to match expected format
                        if class_name.lower() in ['unripe', 'not_ready', 'notready']:
                            class_name = 'not_ready'
                        elif class_name.lower() in ['ripe', 'ready']:
                            class_name = 'ready'
                        elif class_name.lower() in ['spoilt', 'spoiled', 'damaged', 'old']:
                            class_name = 'spoilt'
                        
                        # Calculate center
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x, y, w, h],
                            'center': [center_x, center_y],
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_with_boxes(self, frame, conf=None):
        """
        Detect tomatoes and return in format compatible with detect_tomatoes_with_boxes
        
        Returns:
            (detected: bool, count: int, boxes: list of (x, y, w, h) tuples)
        """
        detections = self.detect(frame, conf)
        
        if len(detections) == 0:
            return False, 0, []
        
        # Extract bounding boxes
        boxes = [det['bbox'] for det in detections]
        
        return True, len(detections), boxes
    
    def is_available(self):
        """Check if YOLO is available and model is loaded"""
        return self.available and YOLO_AVAILABLE

def load_yolo_model(model_path=None, confidence_threshold=0.5):
    """
    Convenience function to load YOLO model
    
    Args:
        model_path: Path to YOLO model. If None, tries common locations:
            - models/tomato/best.pt
            - models/tomato/yolov8_tomato.pt
            - runs/detect/train/weights/best.pt
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        YOLOTomatoDetector instance or None if not available
    """
    if not YOLO_AVAILABLE:
        return None
    
    # Try to find model if path not provided
    if model_path is None:
        possible_paths = [
            'models/tomato/best.pt',
            'models/tomato/yolov8_tomato.pt',
            'runs/detect/train/weights/best.pt',
            'runs/detect/tomato_detector/weights/best.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("⚠️  No YOLO model found. Train a model first.")
            return None
    
    return YOLOTomatoDetector(model_path, confidence_threshold)

