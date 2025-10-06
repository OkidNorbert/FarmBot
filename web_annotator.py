#!/usr/bin/env python3
"""
Web-based Annotation Tool for Tomato Dataset
A simple web interface for annotating images without GUI dependencies
"""

import os
import sys
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

class WebAnnotator:
    def __init__(self, images_dir, labels_dir, classes):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = classes
        self.current_index = 0
        self.image_files = []
        self.current_boxes = []
        
        # Create labels directory
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images to annotate")
        print(f"Classes: {self.classes}")
    
    def get_current_image(self):
        """Get current image as base64 string"""
        if self.current_index >= len(self.image_files):
            return None
        
        image_path = self.image_files[self.current_index]
        image = cv2.imread(str(image_path))
        
        if image is None:
            return None
        
        # Resize image for display
        height, width = image.shape[:2]
        max_size = 800
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    
    def load_annotations(self):
        """Load existing annotations for current image"""
        if self.current_index >= len(self.image_files):
            return []
        
        image_path = self.image_files[self.current_index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        
        return boxes
    
    def save_annotations(self, boxes):
        """Save annotations to YOLO format"""
        if self.current_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n")
    
    def get_progress(self):
        """Get annotation progress"""
        total_images = len(self.image_files)
        annotated_count = len(list(self.labels_dir.glob('*.txt')))
        
        return {
            'current': self.current_index + 1,
            'total': total_images,
            'annotated': annotated_count,
            'progress': (self.current_index + 1) / total_images * 100
        }

# Global annotator instance
annotator = None

@app.route('/')
def index():
    return render_template('annotator.html', classes=annotator.classes)

@app.route('/api/current_image')
def current_image():
    image_base64 = annotator.get_current_image()
    if image_base64 is None:
        return jsonify({'error': 'No more images'})
    
    return jsonify({
        'image': image_base64,
        'filename': annotator.image_files[annotator.current_index].name,
        'progress': annotator.get_progress()
    })

@app.route('/api/annotations')
def get_annotations():
    boxes = annotator.load_annotations()
    return jsonify({'boxes': boxes})

@app.route('/api/save_annotations', methods=['POST'])
def save_annotations():
    data = request.json
    boxes = data.get('boxes', [])
    annotator.save_annotations(boxes)
    return jsonify({'success': True})

@app.route('/api/next_image')
def next_image():
    if annotator.current_index < len(annotator.image_files) - 1:
        annotator.current_index += 1
    return jsonify({'success': True})

@app.route('/api/previous_image')
def previous_image():
    if annotator.current_index > 0:
        annotator.current_index -= 1
    return jsonify({'success': True})

@app.route('/api/progress')
def get_progress():
    return jsonify(annotator.get_progress())

def main():
    global annotator
    
    parser = argparse.ArgumentParser(description='Web-based Annotation Tool')
    parser.add_argument('--images', required=True, help='Images directory')
    parser.add_argument('--labels', required=True, help='Labels directory')
    parser.add_argument('--classes', nargs='+', default=['not_ready', 'ready', 'spoilt'], 
                       help='Class names')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    
    args = parser.parse_args()
    
    annotator = WebAnnotator(args.images, args.labels, args.classes)
    
    print(f"🍅 Web Annotation Tool")
    print(f"Images: {args.images}")
    print(f"Labels: {args.labels}")
    print(f"Classes: {args.classes}")
    print(f"Starting server on http://localhost:{args.port}")
    print("Open your browser and go to the URL above")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == "__main__":
    import argparse
    main()
