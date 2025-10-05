#!/usr/bin/env python3
"""
Simple Annotation Tool for Tomato Dataset
A lightweight annotation tool that works with Python 3.13
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
import argparse

class SimpleAnnotator:
    def __init__(self, images_dir, labels_dir, classes):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = classes
        self.current_image = None
        self.current_boxes = []
        self.current_class = 0
        self.image_files = []
        self.current_index = 0
        
        # Create labels directory if it doesn't exist
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images to annotate")
        print(f"Classes: {self.classes}")
        print("Controls:")
        print("  Mouse: Draw bounding box")
        print("  Keys: 0,1,2 - Select class")
        print("  Keys: n - Next image")
        print("  Keys: p - Previous image")
        print("  Keys: s - Save annotations")
        print("  Keys: q - Quit")
        print("  Keys: d - Delete last box")
        print("  Keys: c - Clear all boxes")
    
    def draw_boxes(self, image):
        """Draw bounding boxes on image"""
        display_image = image.copy()
        
        for i, (box, class_id) in enumerate(self.current_boxes):
            x1, y1, x2, y2 = box
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][class_id]  # Red, Green, Blue
            
            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            label = f"{self.classes[class_id]}"
            cv2.putText(display_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw box number
            cv2.putText(display_image, str(i+1), (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return display_image
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.end_point = (x, y)
            self.drawing = False
            
            # Add bounding box
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Ensure proper order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Only add if box is large enough
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.current_boxes.append([(x1, y1, x2, y2), self.current_class])
                print(f"Added box for class: {self.classes[self.current_class]}")
    
    def save_annotations(self):
        """Save annotations to YOLO format"""
        if not self.current_boxes:
            return
        
        image_path = self.image_files[self.current_index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        h, w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for box, class_id in self.current_boxes:
                x1, y1, x2, y2 = box
                
                # Convert to YOLO format (normalized)
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Saved annotations to: {label_path}")
    
    def load_annotations(self):
        """Load existing annotations"""
        image_path = self.image_files[self.current_index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        self.current_boxes = []
        
        if label_path.exists():
            h, w = self.current_image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert back to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        self.current_boxes.append([(x1, y1, x2, y2), class_id])
    
    def run(self):
        """Main annotation loop"""
        if not self.image_files:
            print("No images found!")
            return
        
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        self.drawing = False
        
        while True:
            if self.current_index < len(self.image_files):
                image_path = self.image_files[self.current_index]
                self.current_image = cv2.imread(str(image_path))
                
                if self.current_image is None:
                    print(f"Could not load image: {image_path}")
                    self.current_index += 1
                    continue
                
                # Load existing annotations
                self.load_annotations()
                
                # Draw boxes
                display_image = self.draw_boxes(self.current_image)
                
                # Add info text
                info_text = f"Image {self.current_index + 1}/{len(self.image_files)} | Class: {self.classes[self.current_class]} | Boxes: {len(self.current_boxes)}"
                cv2.putText(display_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Annotation Tool', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):  # Next image
                self.save_annotations()
                self.current_index += 1
            elif key == ord('p'):  # Previous image
                self.save_annotations()
                self.current_index = max(0, self.current_index - 1)
            elif key == ord('s'):  # Save
                self.save_annotations()
            elif key == ord('d'):  # Delete last box
                if self.current_boxes:
                    self.current_boxes.pop()
                    print("Deleted last box")
            elif key == ord('c'):  # Clear all boxes
                self.current_boxes = []
                print("Cleared all boxes")
            elif key == ord('0'):  # Class 0
                self.current_class = 0
                print(f"Selected class: {self.classes[0]}")
            elif key == ord('1'):  # Class 1
                self.current_class = 1
                print(f"Selected class: {self.classes[1]}")
            elif key == ord('2'):  # Class 2
                self.current_class = 2
                print(f"Selected class: {self.classes[2]}")
        
        # Save final annotations
        self.save_annotations()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simple Annotation Tool')
    parser.add_argument('--images', required=True, help='Images directory')
    parser.add_argument('--labels', required=True, help='Labels directory')
    parser.add_argument('--classes', nargs='+', default=['not_ready', 'ready', 'spoilt'], 
                       help='Class names')
    
    args = parser.parse_args()
    
    print("🍅 Simple Tomato Annotation Tool")
    print("=" * 40)
    
    annotator = SimpleAnnotator(args.images, args.labels, args.classes)
    annotator.run()

if __name__ == "__main__":
    main()
