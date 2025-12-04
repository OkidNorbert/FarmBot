#!/usr/bin/env python3
"""
Tomato Classification Inference Script
Uses trained classifier for real-time tomato classification
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import argparse
import time

class TomatoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(TomatoClassifier, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    """Load the trained model"""
    model = TomatoClassifier(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, transform):
    """Preprocess image for inference"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = transform(image)
    return image.unsqueeze(0)

def predict_tomato(model, image, transform, device):
    """Predict tomato class"""
    with torch.no_grad():
        image_tensor = preprocess_image(image, transform)
        image_tensor = image_tensor.to(device)
        
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Tomato Classification Inference')
    parser.add_argument('--model', default='tomato_classifier.pth', help='Path to trained model')
    parser.add_argument('--source', default=0, help='Camera source (0 for webcam)')
    parser.add_argument('--image', help='Path to single image for testing')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model, device)
    
    # Class names
    class_names = ['not_ready', 'ready', 'spoilt']
    class_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.image:
        # Single image inference
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        predicted_class, confidence = predict_tomato(model, image, transform, device)
        
        print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        
        # Display result
        cv2.putText(image, f"{class_names[predicted_class]}: {confidence:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors[predicted_class], 2)
        cv2.imshow('Tomato Classification', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Real-time camera inference
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.source}")
            return
        
        print("Starting real-time classification...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            predicted_class, confidence = predict_tomato(model, frame, transform, device)
            
            # Draw result
            color = class_colors[predicted_class]
            cv2.putText(frame, f"{class_names[predicted_class]}: {confidence:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw confidence bar
            bar_width = int(confidence * 200)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), color, -1)
            cv2.rectangle(frame, (10, 50), (210, 70), (255, 255, 255), 2)
            
            cv2.imshow('Tomato Classification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
