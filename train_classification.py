#!/usr/bin/env python3
"""
Image Classification Training for Tomato Dataset
Uses the existing classified dataset (Unripe, Ripe, Old, Damaged)
Maps to project classes: not_ready, ready, spoilt
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
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
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
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

def organize_dataset(dataset_path, output_path):
    """Organize the existing classified dataset"""
    print("🍅 Organizing existing classified dataset...")
    
    # Create output directories
    train_dir = Path(output_path) / "train"
    val_dir = Path(output_path) / "val"
    
    for split in [train_dir, val_dir]:
        for class_name in ['not_ready', 'ready', 'spoilt']:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Class mapping
    class_mapping = {
        'u': 'not_ready',    # Unripe -> not_ready
        'r': 'ready',        # Ripe -> ready
        'o': 'spoilt',       # Old -> spoilt
        'd': 'spoilt'        # Damaged -> spoilt
    }
    
    # Get all images
    all_images = []
    for class_prefix in ['u', 'r', 'o', 'd']:
        images = list(Path(dataset_path).glob(f"**/{class_prefix} (*).jpg"))
        all_images.extend(images)
    
    print(f"Found {len(all_images)} images")
    
    # Split into train/val (90/10)
    train_images, val_images = train_test_split(
        all_images, test_size=0.1, random_state=42, stratify=None
    )
    
    # Copy images to organized structure
    for split, images in [('train', train_images), ('val', val_images)]:
        print(f"Processing {split} set: {len(images)} images")
        
        for img_path in tqdm(images):
            # Extract class from filename
            filename = img_path.name
            if filename.startswith('u '):
                class_name = 'not_ready'
            elif filename.startswith('r '):
                class_name = 'ready'
            elif filename.startswith('o ') or filename.startswith('d '):
                class_name = 'spoilt'
            else:
                continue
            
            # Copy to organized structure
            dest_path = Path(output_path) / split / class_name / filename
            shutil.copy2(img_path, dest_path)
    
    print("✅ Dataset organized successfully!")
    return train_dir, val_dir

def create_data_yaml(output_path):
    """Create data.yaml for the organized dataset"""
    data_config = {
        'path': str(Path(output_path).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': 3,
        'names': {
            0: 'not_ready',
            1: 'ready', 
            2: 'spoilt'
        }
    }
    
    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {yaml_path}")
    return yaml_path

def train_classifier(train_dir, val_dir, epochs=50, batch_size=32, lr=0.001):
    """Train the tomato classifier"""
    print("🚀 Starting classification training...")
    
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
    
    # Load datasets
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    
    class_to_idx = {'not_ready': 0, 'ready': 1, 'spoilt': 2}
    
    # Load training data
    for class_name, class_idx in class_to_idx.items():
        class_dir = train_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            train_images.append(str(img_path))
            train_labels.append(class_idx)
    
    # Load validation data
    for class_name, class_idx in class_to_idx.items():
        class_dir = val_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            val_images.append(str(img_path))
            val_labels.append(class_idx)
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Create datasets
    train_dataset = TomatoDataset(train_images, train_labels, train_transform)
    val_dataset = TomatoDataset(val_images, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TomatoClassifier(num_classes=3).to(device)
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
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Tomato Classifier')
    parser.add_argument('--dataset', default='tomato_dataset/images/train', 
                       help='Path to dataset')
    parser.add_argument('--output', default='tomato_classification_dataset',
                       help='Output directory for organized dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🍅 Tomato Classification Training")
    print("=" * 50)
    
    # Organize dataset
    train_dir, val_dir = organize_dataset(args.dataset, args.output)
    
    # Create data.yaml
    data_yaml = create_data_yaml(args.output)
    
    # Train classifier
    model = train_classifier(train_dir, val_dir, args.epochs, args.batch_size, args.lr)
    
    print("✅ Training completed!")
    print(f"📁 Organized dataset: {args.output}")
    print(f"📄 Data config: {data_yaml}")
    print("🎯 Model ready for deployment!")

if __name__ == "__main__":
    main()
