#!/usr/bin/env python3
"""
AI Tomato Sorter - Raspberry Pi Inference Script
Real-time tomato detection and classification on Raspberry Pi 5
"""

import cv2
import numpy as np
import time
import serial
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tomato_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TomatoDetector:
    def __init__(self, model_path, confidence_threshold=0.5, nms_threshold=0.4):
        """Initialize tomato detector"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_names = ['not_ready', 'ready', 'spoilt']
        self.class_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
        
        # Load model based on format
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model based on file extension"""
        model_ext = Path(self.model_path).suffix.lower()
        
        if model_ext == '.onnx':
            return self._load_onnx_model()
        elif model_ext == '.tflite':
            return self._load_tflite_model()
        else:
            raise ValueError(f"Unsupported model format: {model_ext}")
    
    def _load_onnx_model(self):
        """Load ONNX model using OpenCV DNN"""
        logger.info("Loading ONNX model...")
        net = cv2.dnn.readNetFromONNX(self.model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def _load_tflite_model(self):
        """Load TFLite model"""
        logger.info("Loading TFLite model...")
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        except ImportError:
            logger.error("TFLite runtime not available. Install with: pip install tflite-runtime")
            raise
    
    def preprocess_image(self, image, target_size=640):
        """Preprocess image for inference"""
        if isinstance(self.model, cv2.dnn_Net):
            # ONNX preprocessing
            blob = cv2.dnn.blobFromImage(
                image, 1/255.0, (target_size, target_size), 
                swapRB=True, crop=False
            )
            return blob
        else:
            # TFLite preprocessing
            resized = cv2.resize(image, (target_size, target_size))
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)
            return input_data
    
    def postprocess_detections(self, outputs, original_shape, target_size=640):
        """Post-process model outputs to get detections"""
        if isinstance(self.model, cv2.dnn_Net):
            return self._postprocess_onnx(outputs, original_shape, target_size)
        else:
            return self._postprocess_tflite(outputs, original_shape, target_size)
    
    def _postprocess_onnx(self, outputs, original_shape, target_size):
        """Post-process ONNX model outputs"""
        # ONNX output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        detections = []
        
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        for detection in outputs:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls = detection[:6]
                
                if conf > self.confidence_threshold:
                    # Convert to original image coordinates
                    h, w = original_shape[:2]
                    x1 = int(x1 * w / target_size)
                    y1 = int(y1 * h / target_size)
                    x2 = int(x2 * w / target_size)
                    y2 = int(y2 * h / target_size)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.class_names[int(cls)]
                    })
        
        return detections
    
    def _postprocess_tflite(self, outputs, original_shape, target_size):
        """Post-process TFLite model outputs"""
        # TFLite output format varies - this is a simplified version
        # You may need to adjust based on your specific model export
        detections = []
        
        # This is a placeholder - adjust based on your TFLite model output format
        logger.warning("TFLite post-processing not fully implemented. Adjust based on your model.")
        
        return detections
    
    def detect(self, image):
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Run inference
        if isinstance(self.model, cv2.dnn_Net):
            # ONNX inference
            self.model.setInput(input_data)
            outputs = self.model.forward()
        else:
            # TFLite inference
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()
            outputs = self.model.get_tensor(output_details[0]['index'])
        
        # Post-process
        detections = self.postprocess_detections(outputs, image.shape)
        
        inference_time = time.time() - start_time
        
        return detections, inference_time

class ArduinoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        """Initialize Arduino controller"""
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connect()
    
    def connect(self):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            logger.info(f"Connected to Arduino on {self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.serial_conn = None
    
    def send_command(self, command):
        """Send command to Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                logger.debug(f"Sent command: {command}")
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
        else:
            logger.warning("Arduino not connected")
    
    def move_to_position(self, x, y, class_id):
        """Send move command to Arduino"""
        command = f"MOVE {x:.3f} {y:.3f} {int(class_id)}"
        self.send_command(command)
    
    def set_servo_angles(self, angle1, angle2, angle3):
        """Set servo angles directly"""
        command = f"ANGLE {int(angle1)} {int(angle2)} {int(angle3)}"
        self.send_command(command)
    
    def emergency_stop(self):
        """Emergency stop command"""
        self.send_command("STOP")
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn:
            self.serial_conn.close()

class CoordinateMapper:
    def __init__(self, calibration_points=None):
        """Initialize coordinate mapper for pixel to world coordinates"""
        self.calibration_points = calibration_points
        self.transform_matrix = None
        
        if calibration_points:
            self.calibrate()
    
    def calibrate(self):
        """Calibrate coordinate mapping using calibration points"""
        if not self.calibration_points:
            logger.warning("No calibration points provided")
            return
        
        # Extract pixel and world coordinates
        pixel_points = np.array([pt['pixel'] for pt in self.calibration_points], dtype=np.float32)
        world_points = np.array([pt['world'] for pt in self.calibration_points], dtype=np.float32)
        
        # Compute transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(pixel_points, world_points)
        logger.info("Coordinate mapping calibrated")
    
    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates"""
        if self.transform_matrix is None:
            logger.warning("Coordinate mapping not calibrated")
            return pixel_x, pixel_y
        
        # Transform point
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point, self.transform_matrix)
        
        return world_point[0][0]

class TomatoSorter:
    def __init__(self, model_path, arduino_port='/dev/ttyUSB0', camera_id=0):
        """Initialize tomato sorter system"""
        self.detector = TomatoDetector(model_path)
        self.arduino = ArduinoController(arduino_port)
        self.coordinate_mapper = CoordinateMapper()
        
        # Camera setup
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Statistics
        self.detection_count = 0
        self.sorting_accuracy = 0.0
        self.inference_times = []
        
        logger.info("Tomato sorter system initialized")
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Run detection
        detections, inference_time = self.detector.detect(frame)
        self.inference_times.append(inference_time)
        
        # Process each detection
        for detection in detections:
            self.detection_count += 1
            
            # Get bounding box center
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Convert to world coordinates
            world_x, world_y = self.coordinate_mapper.pixel_to_world(center_x, center_y)
            
            # Send command to Arduino
            self.arduino.move_to_position(world_x, world_y, detection['class_id'])
            
            # Log detection
            logger.info(f"Detection {self.detection_count}: {detection['class_name']} "
                       f"at ({world_x:.2f}, {world_y:.2f}) "
                       f"confidence: {detection['confidence']:.3f}")
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            color = self.detector.class_colors[class_id]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def run(self, display=True, save_video=False):
        """Main processing loop"""
        logger.info("Starting tomato sorter system...")
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('tomato_sorter_output.avi', fourcc, 20.0, (640, 480))
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                detections, inference_time = self.process_frame(frame)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Add info overlay
                info_text = f"Detections: {len(detections)} | Inference: {inference_time*1000:.1f}ms"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save video if requested
                if save_video:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Tomato Sorter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except KeyboardInterrupt:
            logger.info("Stopping tomato sorter system...")
        
        finally:
            # Cleanup
            self.camera.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            self.arduino.close()
            
            # Print statistics
            if self.inference_times:
                avg_inference = np.mean(self.inference_times)
                fps = 1.0 / avg_inference
                logger.info(f"Average inference time: {avg_inference*1000:.2f}ms")
                logger.info(f"Average FPS: {fps:.2f}")
                logger.info(f"Total detections: {self.detection_count}")

def main():
    parser = argparse.ArgumentParser(description='Tomato Sorter - Raspberry Pi Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.onnx or .tflite)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--arduino_port', type=str, default='/dev/ttyUSB0', help='Arduino serial port')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--calibration', type=str, help='Path to calibration file')
    
    args = parser.parse_args()
    
    print("🍅 AI Tomato Sorter - Raspberry Pi Inference")
    print("=" * 50)
    
    # Load calibration if provided
    calibration_points = None
    if args.calibration:
        try:
            with open(args.calibration, 'r') as f:
                calibration_data = json.load(f)
                calibration_points = calibration_data['points']
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    # Initialize and run tomato sorter
    sorter = TomatoSorter(
        model_path=args.model,
        arduino_port=args.arduino_port,
        camera_id=args.camera
    )
    
    if calibration_points:
        sorter.coordinate_mapper = CoordinateMapper(calibration_points)
    
    sorter.run(display=not args.no_display, save_video=args.save_video)

if __name__ == "__main__":
    main()
