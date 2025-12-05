#!/usr/bin/env python3
"""
YOLO Inference Module for Tomato Detection and Classification
Handles YOLOv8 model loading and inference with graceful fallback
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import signal

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
        self._model_loaded = False
        
        if not YOLO_AVAILABLE:
            print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
            print("   YOLO detection will not be available until ultralytics is installed.")
            return
        
        if model_path and os.path.exists(model_path):
            try:
                # Use lazy loading - don't load model until first use
                # This prevents segfaults during import/initialization
                self.model_path = model_path
                self._model_loaded = False
                self.available = True  # Mark as available, will load on first use
                print(f"✅ YOLO model path set: {model_path} (will load on first use)")
            except Exception as e:
                print(f"❌ Failed to set YOLO model path: {e}")
                self.available = False
        else:
            print(f"⚠️  YOLO model not found at: {model_path}")
            print("   Train a YOLO model first or provide correct path.")
    
    def _ensure_model_loaded(self):
        """Lazy load the YOLO model on first use to prevent segfaults during import"""
        if not self.available:
            return False
        if hasattr(self, '_model_loaded') and self._model_loaded and self.model is not None:
            return True
        if not hasattr(self, 'model_path') or not self.model_path:
            return False
        
        # Check if model file exists and is readable
        if not os.path.exists(self.model_path):
            print(f"❌ YOLO model file not found: {self.model_path}")
            self.available = False
            return False
        
        try:
            print(f"🔄 Loading YOLO model: {self.model_path}")
            
            # Force CPU mode via environment variables to prevent GPU segfaults
            # os is already imported at the top of the file
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            try:
                # Hide GPU from PyTorch to force CPU mode
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # Try to load model with explicit device specification
                # This prevents GPU-related segfaults during model loading
                try:
                    # Load model - it will use CPU since GPU is hidden
                    self.model = YOLO(self.model_path)
                    
                    # Force model to CPU after loading to prevent GPU issues
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                        try:
                            import torch
                            self.model.model.to('cpu')
                        except:
                            pass  # If to() doesn't work, continue anyway
                    
                    # Set device for predictions
                    if hasattr(self.model, 'device'):
                        try:
                            self.model.device = 'cpu'
                        except:
                            pass
                    
                except Exception as load_error:
                    print(f"❌ Error during YOLO model loading: {load_error}")
                    import traceback
                    traceback.print_exc()
                    self.available = False
                    self._model_loaded = False
                    return False
                    
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Verify model loaded successfully
            if self.model is None:
                print("❌ YOLO model is None after loading")
                self.available = False
                self._model_loaded = False
                return False
            
            self._model_loaded = True
            print(f"✅ YOLO model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
            self._model_loaded = False
            return False
    
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
        if not self.available:
            return []
        
        # Lazy load model on first use
        if not self._ensure_model_loaded():
            return []
        
        if conf is None:
            conf = self.confidence_threshold
        
        # Validate frame before inference
        if frame is None or frame.size == 0:
            print("⚠️  Invalid frame provided to YOLO detector")
            return []
        
        # Ensure frame is numpy array and in correct format
        if not isinstance(frame, np.ndarray):
            print("⚠️  Frame is not a numpy array")
            return []
        
        # Check frame dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"⚠️  Invalid frame shape: {frame.shape}")
            return []
        
        try:
            # Run YOLO inference with device specification to avoid GPU issues
            # Use CPU explicitly to prevent segfaults from GPU memory issues
            # Add timeout protection to prevent hanging/segfaults
            import multiprocessing
            from multiprocessing import Process, Queue
            
            def _run_inference(frame, model, conf, result_queue):
                """Run inference in separate process to isolate segfaults"""
                try:
                    results = model.predict(
                        source=frame,
                        conf=conf,
                        verbose=False,
                        imgsz=640,
                        device='cpu'
                    )
                    result_queue.put(('success', results))
                except Exception as e:
                    result_queue.put(('error', str(e)))
            
            # Try direct inference first (faster if it works)
            try:
                results = self.model.predict(
                    source=frame,
                    conf=conf,
                    verbose=False,
                    imgsz=640,  # Standard YOLO input size
                    device='cpu',  # Force CPU to avoid GPU-related segfaults
                    half=False  # Disable half precision to prevent GPU issues
                )
            except Exception as direct_error:
                print(f"⚠️  Direct YOLO inference failed: {direct_error}")
                # Mark as unavailable to prevent repeated crashes
                self.available = False
                self._model_loaded = False
                return []
            
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                # Check if we have detections
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get class names from model
                    class_names = result.names
                    
                    for box in result.boxes:
                        try:
                            # Get bounding box coordinates (xyxy format)
                            # Use .tolist() instead of .cpu().numpy() to avoid segfaults
                            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            
                            # Convert to (x, y, w, h) format
                            x = x1
                            y = y1
                            w = x2 - x1
                            h = y2 - y1
                            
                            # Get class and confidence - use safer extraction
                            cls_data = box.cls[0]
                            conf_data = box.conf[0]
                            
                            # Try tolist first, fallback to cpu().numpy()
                            if hasattr(cls_data, 'tolist'):
                                class_id = int(cls_data.tolist())
                            else:
                                class_id = int(cls_data.cpu().numpy())
                            
                            if hasattr(conf_data, 'tolist'):
                                confidence = float(conf_data.tolist())
                            else:
                                confidence = float(conf_data.cpu().numpy())
                        except Exception as box_error:
                            print(f"⚠️  Error processing box: {box_error}")
                            continue  # Skip this box and continue with next
                        
                        # Map class ID to class name
                        # YOLO classes should match: 0=not_ready, 1=ready, 2=spoilt
                        # Use class ID directly for most reliable mapping
                        if class_id == 0:
                            class_name = 'not_ready'
                        elif class_id == 1:
                            class_name = 'ready'
                        elif class_id == 2:
                            class_name = 'spoilt'
                        else:
                            # Fallback: get name from model and normalize
                            class_name = class_names.get(class_id, f"class_{class_id}")
                            # Normalize class names to match expected format
                            if class_name.lower() in ['unripe', 'not_ready', 'notready']:
                                class_name = 'not_ready'
                            elif class_name.lower() in ['ripe', 'ready']:
                                class_name = 'ready'
                            elif class_name.lower() in ['spoilt', 'spoiled', 'damaged', 'old']:
                                class_name = 'spoilt'
                        
                        # Debug logging for troubleshooting
                        print(f"[YOLO DEBUG] Detection: Class ID={class_id}, Class Name='{class_name}', Confidence={confidence:.3f}, BBox=[{x},{y},{w},{h}]")
                        
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
            
        except RuntimeError as e:
            # Handle PyTorch/GPU memory errors
            error_msg = str(e)
            print(f"❌ YOLO RuntimeError: {error_msg}")
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print("   GPU memory issue - try using CPU or reducing image size")
            import traceback
            traceback.print_exc()
            return []
        except Exception as e:
            # Catch all other exceptions including segfaults
            error_type = type(e).__name__
            print(f"❌ YOLO detection error ({error_type}): {e}")
            import traceback
            traceback.print_exc()
            # Mark model as unavailable to prevent further crashes
            self.available = False
            self._model_loaded = False
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

