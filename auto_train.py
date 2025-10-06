#!/usr/bin/env python3
"""
Automated AI Training Pipeline for Agricultural Crops
====================================================

This script automates the entire training process for new crops.
Just organize your photos in folders and run this script!

Usage:
    python auto_train.py --dataset_path /path/to/new_crop_dataset
    python auto_train.py --dataset_path /path/to/new_crop_dataset --crop_name "strawberry"
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import json
from datetime import datetime

class AutoCropClassifier(nn.Module):
    """Flexible CNN classifier that adapts to any number of classes"""
    def __init__(self, num_classes=3, input_size=224):
        super(AutoCropClassifier, self).__init__()
        
        # Calculate the size after convolutions
        # For 224x224 input: 224 -> 112 -> 56 -> 28
        conv_output_size = 28
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * conv_output_size * conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AutoTrainer:
    def __init__(self, dataset_path, crop_name=None, output_dir="models"):
        self.dataset_path = Path(dataset_path)
        self.crop_name = crop_name or self.dataset_path.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create crop-specific model directory
        self.model_dir = self.output_dir / self.crop_name
        self.model_dir.mkdir(exist_ok=True)
        
        print(f"🌱 Auto-Training Pipeline for: {self.crop_name}")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💾 Model output: {self.model_dir}")
        print("="*60)

    def analyze_dataset(self):
        """Analyze the dataset structure and count images"""
        print("🔍 Analyzing dataset structure...")
        
        # Find all class folders
        class_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
        class_counts = {}
        total_images = 0
        
        for class_folder in class_folders:
            # Count images (jpg, jpeg, png)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_folder.glob(ext))
            
            count = len(image_files)
            class_counts[class_folder.name] = count
            total_images += count
            
            print(f"  📂 {class_folder.name}: {count} images")
        
        print(f"\n📊 Total images: {total_images}")
        print(f"🏷️  Classes found: {list(class_counts.keys())}")
        
        return class_counts, total_images

    def create_data_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Create train/val/test splits automatically"""
        print(f"\n📋 Creating data splits (train:{train_ratio}, val:{val_ratio}, test:{test_ratio})...")
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.dataset_path / split).mkdir(exist_ok=True)
        
        class_folders = [f for f in self.dataset_path.iterdir() if f.is_dir() and f.name not in ['train', 'val', 'test']]
        
        for class_folder in class_folders:
            # Get all images in this class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_folder.glob(ext))
            
            # Shuffle and split
            import random
            random.shuffle(image_files)
            
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split the files
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Create class directories in each split
            for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                split_class_dir = self.dataset_path / split_name / class_folder.name
                split_class_dir.mkdir(exist_ok=True)
                
                # Copy files to split directories
                for img_file in files:
                    shutil.copy2(img_file, split_class_dir / img_file.name)
            
            print(f"  ✅ {class_folder.name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        print("\n📄 Creating data.yaml...")
        
        # Get class names from train directory
        train_dir = self.dataset_path / 'train'
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        data_config = {
            'path': str(self.dataset_path.resolve()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
        
        print(f"  ✅ Created {yaml_path}")
        return data_config

    def train_model(self, epochs=30, batch_size=32, learning_rate=0.001):
        """Train the model automatically"""
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Load data configuration
        data_yaml_path = self.dataset_path / 'data.yaml'
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        class_names = list(data_config['names'].values())
        num_classes = data_config['nc']
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dir = self.dataset_path / 'train'
        val_dir = self.dataset_path / 'val'
        
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"📊 Training with {len(train_dataset)} images, validating with {len(val_dataset)} images")
        print(f"🏷️  Classes: {class_names}")
        
        # Initialize model
        model = AutoCropClassifier(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"🖥️  Using device: {device}")
        
        # Training loop
        best_val_acc = 0.0
        training_history = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_dir / 'best_model.pth')
            
            # Record history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save final model and training info
        torch.save(model.state_dict(), self.model_dir / 'final_model.pth')
        
        # Save training metadata
        metadata = {
            'crop_name': self.crop_name,
            'dataset_path': str(self.dataset_path),
            'num_classes': num_classes,
            'class_names': class_names,
            'training_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': val_acc,
            'training_date': datetime.now().isoformat(),
            'device': str(device)
        }
        
        with open(self.model_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(self.model_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"💾 Model saved to: {self.model_dir}")
        
        return metadata

    def create_inference_script(self):
        """Create a crop-specific inference script"""
        print("\n📝 Creating inference script...")
        
        inference_script = f'''#!/usr/bin/env python3
"""
Inference script for {self.crop_name} classifier
Auto-generated by auto_train.py
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse
import os

class AutoCropClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AutoCropClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="{self.crop_name} Classifier Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Model file name")
    args = parser.parse_args()
    
    # Load metadata
    with open("training_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = AutoCropClassifier(num_classes=metadata['num_classes'])
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
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
    
    print(f"🌱 {self.crop_name} Classification Result:")
    print(f"📸 Image: {{os.path.basename(args.image)}}")
    print(f"🏷️  Predicted Class: {{predicted_class}}")
    print(f"🎯 Confidence: {{confidence:.2f}}")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.model_dir / f'{self.crop_name}_inference.py'
        with open(script_path, 'w') as f:
            f.write(inference_script)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        print(f"  ✅ Created {script_path}")
        
        return script_path

    def run_full_pipeline(self, epochs=30, batch_size=32, learning_rate=0.001):
        """Run the complete automated training pipeline"""
        print("🚀 Starting Automated Training Pipeline")
        print("="*60)
        
        # Step 1: Analyze dataset
        class_counts, total_images = self.analyze_dataset()
        
        if total_images < 50:
            print("⚠️  Warning: Very small dataset. Consider adding more images for better results.")
        
        # Step 2: Create data splits
        self.create_data_splits()
        
        # Step 3: Create data.yaml
        data_config = self.create_data_yaml()
        
        # Step 4: Train model
        metadata = self.train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # Step 5: Create inference script
        inference_script = self.create_inference_script()
        
        print("\n🎉 AUTOMATED TRAINING COMPLETE!")
        print("="*60)
        print(f"🌱 Crop: {self.crop_name}")
        print(f"📊 Classes: {metadata['class_names']}")
        print(f"🎯 Best Accuracy: {metadata['best_val_accuracy']:.2f}%")
        print(f"💾 Model Directory: {self.model_dir}")
        print(f"🔧 Inference Script: {inference_script}")
        print("\n📋 Next Steps:")
        print(f"   1. Test your model: python {inference_script} --image path/to/test/image.jpg")
        print(f"   2. Deploy to production using the model in: {self.model_dir}")
        print(f"   3. Check training_metadata.json for detailed information")

def main():
    parser = argparse.ArgumentParser(description="Automated AI Training for Agricultural Crops")
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to dataset with organized class folders")
    parser.add_argument("--crop_name", type=str, default=None,
                       help="Name of the crop (defaults to dataset folder name)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Directory to save trained models (default: models)")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path '{args.dataset_path}' does not exist!")
        return 1
    
    # Create trainer and run pipeline
    trainer = AutoTrainer(
        dataset_path=args.dataset_path,
        crop_name=args.crop_name,
        output_dir=args.output_dir
    )
    
    try:
        trainer.run_full_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        return 0
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
