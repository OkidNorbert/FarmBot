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
    def __init__(self, num_classes=3):
        super(TomatoClassifier, self).__init__()
        # This matches the actual trained model architecture
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

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