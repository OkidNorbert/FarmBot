#!/usr/bin/env python3
"""
AI Tomato Sorter - Data Preparation Script
Prepares and validates dataset for YOLOv8 training
"""

import os
import shutil
import random
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def create_dataset_structure(base_path):
    """Create the required dataset directory structure"""
    print("📁 Creating dataset structure...")
    
    base_path = Path(base_path)
    directories = [
        'images/train',
        'images/val', 
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {full_path}")
    
    return base_path

def split_dataset(source_images, source_labels, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets"""
    print("📊 Splitting dataset...")
    
    # Get all image files
    image_files = list(Path(source_images).glob('*.jpg')) + list(Path(source_images).glob('*.png'))
    random.shuffle(image_files)
    
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    print(f"   Total images: {total_images}")
    print(f"   Train: {train_count} ({train_ratio:.1%})")
    print(f"   Val: {val_count} ({val_ratio:.1%})")
    print(f"   Test: {total_images - train_count - val_count} ({test_ratio:.1%})")
    
    # Split images
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:
            # Copy image
            dst_img = Path(output_path) / f'images/{split}' / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy corresponding label if it exists
            label_file = Path(source_labels) / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = Path(output_path) / f'labels/{split}' / f"{img_file.stem}.txt"
                shutil.copy2(label_file, dst_label)
    
    return len(train_files), len(val_files), len(test_files)

def validate_annotations(dataset_path):
    """Validate YOLO format annotations"""
    print("🔍 Validating annotations...")
    
    dataset_path = Path(dataset_path)
    issues = []
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / f'images/{split}'
        labels_dir = dataset_path / f'labels/{split}'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Check if label file exists
            if not label_file.exists():
                issues.append(f"Missing label for {img_file.name}")
                continue
            
            # Validate label format
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"{label_file.name}:{line_num} - Invalid format (expected 5 values)")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Check ranges
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            issues.append(f"{label_file.name}:{line_num} - Coordinates out of range [0,1]")
                        
                        if class_id not in [0, 1, 2]:
                            issues.append(f"{label_file.name}:{line_num} - Invalid class ID {class_id}")
                            
                    except ValueError:
                        issues.append(f"{label_file.name}:{line_num} - Non-numeric values")
                        
            except Exception as e:
                issues.append(f"Error reading {label_file.name}: {e}")
    
    if issues:
        print(f"⚠️  Found {len(issues)} annotation issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
    else:
        print("✅ All annotations are valid")
    
    return issues

def analyze_dataset(dataset_path):
    """Analyze dataset statistics"""
    print("📊 Analyzing dataset...")
    
    dataset_path = Path(dataset_path)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / f'images/{split}'
        labels_dir = dataset_path / f'labels/{split}'
        
        if not images_dir.exists():
            continue
            
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        # Count images and annotations
        total_images = len(image_files)
        total_annotations = 0
        class_counts = {0: 0, 1: 0, 2: 0}  # not_ready, ready, spoilt
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            total_annotations += 1
        
        stats[split] = {
            'images': total_images,
            'annotations': total_annotations,
            'class_counts': class_counts.copy()
        }
    
    # Print statistics
    print("\n📈 Dataset Statistics:")
    print("-" * 50)
    for split, data in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Images: {data['images']}")
        print(f"  Annotations: {data['annotations']}")
        print(f"  Avg annotations per image: {data['annotations']/data['images']:.2f}")
        print(f"  Class distribution:")
        print(f"    Not Ready (0): {data['class_counts'][0]}")
        print(f"    Ready (1): {data['class_counts'][1]}")
        print(f"    Spoilt (2): {data['class_counts'][2]}")
    
    return stats

def create_sample_visualization(dataset_path, num_samples=5):
    """Create sample visualization of annotated images"""
    print("🖼️  Creating sample visualizations...")
    
    dataset_path = Path(dataset_path)
    class_names = ['not_ready', 'ready', 'spoilt']
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    for split in ['train', 'val']:
        images_dir = dataset_path / f'images/{split}'
        labels_dir = dataset_path / f'labels/{split}'
        
        if not images_dir.exists():
            continue
            
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 3))
        if len(sample_files) == 1:
            axes = [axes]
        
        for idx, img_file in enumerate(sample_files):
            # Load image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Load annotations
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw bounding box
                        color = class_colors[class_id]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, class_names[class_id], (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_file.name}")
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = dataset_path / f'sample_{split}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {output_path}")
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for tomato sorter')
    parser.add_argument('--source_images', type=str, required=True, help='Source images directory')
    parser.add_argument('--source_labels', type=str, required=True, help='Source labels directory')
    parser.add_argument('--output', type=str, default='tomato_dataset', help='Output dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--validate', action='store_true', help='Validate annotations')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset statistics')
    parser.add_argument('--visualize', action='store_true', help='Create sample visualizations')
    
    args = parser.parse_args()
    
    print("🍅 AI Tomato Sorter - Data Preparation")
    print("=" * 50)
    
    # Create dataset structure
    dataset_path = create_dataset_structure(args.output)
    
    # Split dataset
    train_count, val_count, test_count = split_dataset(
        args.source_images, 
        args.source_labels, 
        args.output,
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    # Validate annotations
    if args.validate:
        issues = validate_annotations(args.output)
        if issues:
            print(f"❌ Found {len(issues)} annotation issues")
        else:
            print("✅ All annotations are valid")
    
    # Analyze dataset
    if args.analyze:
        stats = analyze_dataset(args.output)
    
    # Create visualizations
    if args.visualize:
        create_sample_visualization(args.output)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Dataset saved to: {args.output}")
    print(f"📊 Split: {train_count} train, {val_count} val, {test_count} test")

if __name__ == "__main__":
    main()
