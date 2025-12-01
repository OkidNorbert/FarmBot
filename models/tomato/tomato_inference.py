#!/usr/bin/env python3
"""
Inference script for tomato classifier
Auto-generated for web interface
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse
import os

# OpenCV for frame processing (optional, only needed for detect_tomatoes)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class TomatoClassifier(nn.Module):
    """ResNet-based classifier that matches the trained model"""
    def __init__(self, model_path=None, num_classes=3):
        super(TomatoClassifier, self).__init__()
        # This matches the actual trained model architecture
        from torchvision.models import resnet18
        try:
            # Try new API first (torchvision 0.13+)
            self.backbone = resnet18(weights=None)
        except (TypeError, ValueError):
            # Fall back to old API
            self.backbone = resnet18(pretrained=False)
        
        # If model_path is provided, load metadata and model
        if model_path and os.path.exists(model_path):
            try:
                # Load metadata to get num_classes
                metadata_path = os.path.join(os.path.dirname(model_path), "training_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    num_classes = metadata.get('num_classes', num_classes)
                
                # Set up the final layer
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
                
                # Load the model weights
                state_dict = torch.load(model_path, map_location='cpu')
                self.load_state_dict(state_dict)
                self.eval()
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                # Fall back to default num_classes
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            # Just create the model structure
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
    def classify_crop(self, crop_image, confidence_threshold=0.3):
        """Classify a single cropped tomato image
        
        Args:
            crop_image: PIL Image or numpy array (RGB) of a single tomato
            confidence_threshold: Minimum confidence to consider a detection
        
        Returns:
            Dictionary with 'class', 'confidence', or None if below threshold
        """
        try:
            # Convert numpy array to PIL if needed
            if isinstance(crop_image, np.ndarray):
                # Crop is already RGB from frame_rgb, just convert to PIL
                if crop_image.dtype != np.uint8:
                    crop_image = (crop_image * 255).astype(np.uint8)
                crop_image = Image.fromarray(crop_image)
            
            # Ensure it's RGB
            if isinstance(crop_image, Image.Image):
                crop_image = crop_image.convert("RGB")
            
            # Transform for model input
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(crop_image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
                confidence = max_prob.item()
                class_idx = predicted_idx.item()
            
            # Load class names
            metadata_path = os.path.join(os.path.dirname(__file__), "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                class_names = metadata.get('class_names', ['not_ready', 'ready', 'spoilt'])
            else:
                class_names = ['not_ready', 'ready', 'spoilt']
            
            class_name = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
            
            # Only return if confidence is above threshold
            if confidence >= confidence_threshold:
                return {
                    'class': class_name,
                    'confidence': confidence
                }
            return None
        except Exception as e:
            print(f"Classification error: {e}")
            return None
    
    def detect_tomatoes(self, frame, confidence_threshold=0.3, enhance_for_ugandan=True, bboxes=None):
        """Detect and classify tomatoes in a video frame (OpenCV format)
        
        Args:
            frame: OpenCV BGR frame (numpy array)
            confidence_threshold: Minimum confidence to consider a detection (lowered for Ugandan tomatoes)
            enhance_for_ugandan: Apply image enhancement optimized for Ugandan tomato varieties
            bboxes: Optional list of bounding boxes [(x, y, w, h), ...] to classify individually
        
        Returns:
            List of detections with 'class', 'confidence', 'bbox', 'center'
        """
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available for frame detection")
            return []
        
        detections = []
        
        # Image enhancement for Ugandan tomatoes
        if enhance_for_ugandan:
            # Convert to LAB color space for better color enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance lightness (helps with varying lighting conditions)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Enhance color saturation (helps with different tomato colors)
            a = cv2.multiply(a, 1.2)  # Increase red-green channel
            b = cv2.multiply(b, 1.1)  # Increase blue-yellow channel
            
            # Merge back and convert to RGB
            enhanced = cv2.merge([l, a, b])
            frame_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If bounding boxes are provided, classify each one individually
        if bboxes and len(bboxes) > 0:
            h, w = frame.shape[:2]
            for bbox in bboxes:
                x, y, box_w, box_h = bbox
                
                # Ensure bbox is within frame bounds
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                box_w = min(box_w, w - x)
                box_h = min(box_h, h - y)
                
                if box_w > 0 and box_h > 0:
                    # Extract crop from RGB frame
                    crop = frame_rgb[y:y+box_h, x:x+box_w]
                    
                    if crop.size > 0:
                        # Classify the individual tomato
                        result = self.classify_crop(crop, confidence_threshold)
                        
                        if result:
                            # Calculate center of bounding box
                            center_x = x + box_w // 2
                            center_y = y + box_h // 2
                            
                            detections.append({
                                'class': result['class'],
                                'confidence': result['confidence'],
                                'bbox': [x, y, box_w, box_h],
                                'center': [center_x, center_y]
                            })
        
        # Fallback: If no bboxes provided, use full frame detection (backward compatibility)
        if not bboxes or len(bboxes) == 0:
            try:
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(frame_resized).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, 1)
                    confidence = max_prob.item()
                    class_idx = predicted_idx.item()
                
                # Load class names
                metadata_path = os.path.join(os.path.dirname(__file__), "training_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    class_names = metadata.get('class_names', ['not_ready', 'ready', 'spoilt'])
                else:
                    class_names = ['not_ready', 'ready', 'spoilt']
                
                class_name = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
                
                # Only return if confidence is above threshold
                if confidence >= confidence_threshold:
                    h, w = frame.shape[:2]
                    # Return center of frame as detection
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [w//2 - 50, h//2 - 50, 100, 100],
                        'center': [w//2, h//2]
                    })
            except Exception as e:
                print(f"Detection error: {e}")
        
        return detections
    
    def predict(self, image_path):
        """Predict class for a single image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
            confidence = probabilities[0][predicted_idx.item()].item()
        
        return predicted_idx.item(), confidence

def main():
    parser = argparse.ArgumentParser(description="Tomato Classifier Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Model file name")
    args = parser.parse_args()
    
    # Load metadata
    metadata_path = os.path.join(os.path.dirname(__file__), "training_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    model = TomatoClassifier(num_classes=metadata['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = metadata['class_names'][predicted_idx.item()]
        confidence = probabilities[0][predicted_idx.item()].item()
    
    print(f"🌱 Tomato Classification Result:")
    print(f"📸 Image: {os.path.basename(args.image)}")
    print(f"🏷️  Predicted Class: {predicted_class}")
    print(f"🎯 Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()