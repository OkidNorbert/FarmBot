#!/usr/bin/env python3
"""
AI Tomato Sorter - Training Script
Trains YOLOv8 model to classify tomatoes as not_ready, ready, or spoilt
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd

def setup_training_environment():
    """Setup training environment and check dependencies"""
    print("🔧 Setting up training environment...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected, training will be slower")
    
    # Check data.yaml exists
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml not found at {data_yaml}")
        print("   Please ensure data.yaml is in the current directory")
        return False
    
    return True

def validate_dataset(data_yaml_path):
    """Validate dataset structure and annotations"""
    print("📊 Validating dataset...")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    train_path = dataset_path / config['train']
    val_path = dataset_path / config['val']
    
    # Check if directories exist
    if not train_path.exists():
        print(f"❌ Training images directory not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"❌ Validation images directory not found: {val_path}")
        return False
    
    # Count images
    train_images = len(list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')))
    val_images = len(list(val_path.glob('*.jpg')) + list(val_path.glob('*.png')))
    
    print(f"📈 Dataset statistics:")
    print(f"   Training images: {train_images}")
    print(f"   Validation images: {val_images}")
    print(f"   Classes: {config['names']}")
    
    if train_images < 100:
        print("⚠️  Warning: Very few training images. Consider collecting more data.")
    
    return True

def train_model(data_yaml, epochs=100, imgsz=640, batch=16, device='auto'):
    """Train YOLOv8 model with specified parameters"""
    print("🚀 Starting model training...")
    
    # Load YOLOv8n model (nano version for speed)
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'patience': 20,  # Early stopping
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,  # Cache images for faster training
        'workers': 8,  # Number of worker threads
        'project': 'runs/detect',
        'name': 'tomato_sorter',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 2.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,    # Image HSV-Saturation augmentation
        'hsv_v': 0.4,    # Image HSV-Value augmentation
        'degrees': 0.0,  # Image rotation
        'translate': 0.1,  # Image translation
        'scale': 0.5,   # Image scale
        'shear': 0.0,   # Image shear
        'perspective': 0.0,  # Image perspective
        'flipud': 0.0,  # Image flip up-down
        'fliplr': 0.5,  # Image flip left-right
        'mosaic': 1.0,  # Image mosaic
        'mixup': 0.0,   # Image mixup
        'copy_paste': 0.0,  # Segment copy-paste
    }
    
    print(f"📋 Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device}")
    
    # Start training
    results = model.train(**train_args)
    
    return results, model

def evaluate_model(model, data_yaml):
    """Evaluate trained model on validation set"""
    print("📊 Evaluating model...")
    
    # Run validation
    metrics = model.val(data=data_yaml)
    
    # Print key metrics
    print(f"📈 Model Performance:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    return metrics

def plot_training_curves(results_dir):
    """Plot training curves from results"""
    print("📊 Generating training plots...")
    
    # Read results.csv if it exists
    results_csv = Path(results_dir) / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mAP curves
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision/Recall
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(results_dir) / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {results_dir}/training_curves.png")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Tomato Sorter')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--plot', action='store_true', help='Generate training plots')
    
    args = parser.parse_args()
    
    print("🍅 AI Tomato Sorter - Training Script")
    print("=" * 50)
    
    # Setup environment
    if not setup_training_environment():
        sys.exit(1)
    
    # Validate dataset
    if not validate_dataset(args.data):
        print("❌ Dataset validation failed")
        sys.exit(1)
    
    # Train model
    try:
        results, model = train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
        
        # Evaluate model
        metrics = evaluate_model(model, args.data)
        
        # Generate plots if requested
        if args.plot:
            results_dir = "runs/detect/tomato_sorter"
            plot_training_curves(results_dir)
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: runs/detect/tomato_sorter/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
