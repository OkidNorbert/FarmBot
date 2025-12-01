#!/usr/bin/env python3
"""
Tomato Classification Training Script
Uses the existing classified dataset (Unripe, Ripe, Old, Damaged)
Maps to project classes: not_ready, ready, spoilt
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TomatoDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Class mapping: 4 original classes -> 3 project classes
        self.class_mapping = {
            'Unripe': 0,    # not_ready
            'Ripe': 1,      # ready
            'Old': 2,       # spoilt
            'Damaged': 2    # spoilt
        }
        
        self.class_names = ['not_ready', 'ready', 'spoilt']
        
        # Load dataset
        self.images = []
        self.labels = []
        
        split_dir = self.data_dir / split
        for class_folder in split_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                if class_name in self.class_mapping:
                    label = self.class_mapping[class_name]
                    
                    # Get all images in this class folder
                    for img_path in class_folder.glob('*.jpg'):
                        self.images.append(str(img_path))
                        self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TomatoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(TomatoClassifier, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def create_data_yaml(dataset_path):
    """Create data.yaml for the dataset"""
    data_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': 3,
        'names': {
            0: 'not_ready',
            1: 'ready', 
            2: 'spoilt'
        }
    }
    
    yaml_path = Path(dataset_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {yaml_path}")
    return yaml_path

def train_classifier(dataset_path, epochs=50, batch_size=32, lr=0.001, pretrained_model=None):
    """Train the tomato classifier, optionally continuing from a pretrained model"""
    if pretrained_model:
        print("🔄 Continuing training from existing model (fine-tuning)...")
    else:
        print("🚀 Starting classification training from scratch...")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TomatoDataset(dataset_path, split='train', transform=train_transform)
    val_dataset = TomatoDataset(dataset_path, split='val', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TomatoClassifier(num_classes=3).to(device)
    
    # Load pretrained model if provided
    if pretrained_model and os.path.exists(pretrained_model):
        try:
            print(f"📂 Loading pretrained model from: {pretrained_model}")
            state_dict = torch.load(pretrained_model, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Pretrained model loaded successfully!")
            # Use lower learning rate for fine-tuning
            if lr >= 0.001:
                lr = lr * 0.1  # Reduce learning rate by 10x for fine-tuning
                print(f"📉 Using reduced learning rate for fine-tuning: {lr}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained model: {e}")
            print("   Starting training from scratch instead...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Save model
    model_path = 'tomato_classifier.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Also save to models/tomato/ if directory exists
    models_dir = os.path.join('models', 'tomato')
    if os.path.exists(models_dir):
        best_model_path = os.path.join(models_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Model also saved to {best_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', linewidth=2)
    plt.plot(val_accs, label='Val Acc', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save training curves to multiple locations
    curves_path = 'training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to {curves_path}")
    
    # Also save to models/tomato/ if directory exists
    if os.path.exists(models_dir):
        curves_model_path = os.path.join(models_dir, 'training_curves.png')
        plt.savefig(curves_model_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves also saved to {curves_model_path}")
    
    # Close figure to free memory (don't show in headless environment)
    plt.close()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Tomato Classifier')
    parser.add_argument('--dataset', default='tomato_dataset', 
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', '--pretrained_model', type=str, default=None,
                       help='Path to pretrained model to continue training from (fine-tuning)')
    
    args = parser.parse_args()
    
    print("🍅 Tomato Classification Training")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print()
    
    # Create data.yaml
    data_yaml = create_data_yaml(args.dataset)
    
    # Train classifier
    model = train_classifier(args.dataset, args.epochs, args.batch_size, args.lr, args.resume)
    
    print("✅ Training completed!")
    print(f"📁 Dataset: {args.dataset}")
    print(f"📄 Data config: {data_yaml}")
    print("🎯 Model ready for deployment!")

if __name__ == "__main__":
    main()
