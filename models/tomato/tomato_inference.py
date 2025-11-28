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