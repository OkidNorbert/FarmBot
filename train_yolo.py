#!/usr/bin/env python3
"""
YOLOv8 Training Script for Tomato Detection
Converts classification dataset to YOLO format and trains YOLOv8 model
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
import argparse

# Check for optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def check_ultralytics():
    """Check if ultralytics is installed"""
    try:
        from ultralytics import YOLO
        return True, YOLO
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("   Install with: pip install ultralytics")
        return False, None

def convert_classification_to_yolo(dataset_path, output_path):
    """
    Convert classification dataset (folders by class) to YOLO format
    
    Args:
        dataset_path: Path to classification dataset (e.g., datasets/tomato/train/)
        output_path: Path to output YOLO dataset (e.g., datasets/tomato_yolo/)
    """
    print(f"🔄 Converting classification dataset to YOLO format...")
    print(f"   Input: {dataset_path}")
    print(f"   Output: {output_path}")
    
    # Create output directories
    images_train = Path(output_path) / "images" / "train"
    images_val = Path(output_path) / "images" / "val"
    labels_train = Path(output_path) / "labels" / "train"
    labels_val = Path(output_path) / "labels" / "val"
    
    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Class mapping: folder name -> class ID
    class_mapping = {
        'unripe': 0,
        'not_ready': 0,
        'ripe': 1,
        'ready': 1,
        'old': 2,
        'damaged': 2,
        'spoilt': 2,
        'spoiled': 2
    }
    
    # Process train and val splits
    for split in ['train', 'val']:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            print(f"⚠️  {split} directory not found: {split_path}")
            continue
        
        images_dir = images_train if split == 'train' else images_val
        labels_dir = labels_train if split == 'train' else labels_val
        
        # Process each class folder
        for class_folder in split_path.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name.lower()
            class_id = None
            
            # Find matching class ID
            for key, cid in class_mapping.items():
                if key in class_name:
                    class_id = cid
                    break
            
            if class_id is None:
                print(f"⚠️  Unknown class folder: {class_folder.name}, skipping...")
                continue
            
            print(f"   Processing {split}/{class_folder.name} -> class {class_id}")
            
            # Process images
            image_count = 0
            for image_file in class_folder.iterdir():
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                # Copy image to images directory
                dest_image = images_dir / image_file.name
                shutil.copy2(image_file, dest_image)
                
                # Create YOLO label file (normalized center format)
                # For classification dataset, we assume the whole image is the object
                # This is a placeholder - you should annotate with bounding boxes!
                label_file = labels_dir / (image_file.stem + '.txt')
                
                # Placeholder: whole image as bounding box (normalized)
                # Format: class_id x_center y_center width height
                # x_center, y_center, width, height are normalized (0-1)
                with open(label_file, 'w') as f:
                    # This is a placeholder - you need to annotate images with actual bounding boxes!
                    # For now, we'll use the whole image as the bounding box
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                image_count += 1
            
            print(f"      Processed {image_count} images")
    
    print(f"✅ Conversion complete!")
    print(f"⚠️  NOTE: This created placeholder labels (whole image as bounding box).")
    print(f"   For proper YOLO training, you need to annotate images with actual bounding boxes!")
    print(f"   Use tools like LabelImg (https://github.com/tzutalin/labelImg) to annotate.")
    
    return output_path

def create_data_yaml(dataset_path, output_path):
    """Create data.yaml for YOLO training"""
    yaml_path = Path(output_path) / "data.yaml"
    
    # Get absolute paths
    abs_dataset_path = os.path.abspath(dataset_path)
    
    data = {
        'path': abs_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'not_ready',
            1: 'ready',
            2: 'spoilt'
        },
        'nc': 3
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {yaml_path}")
    return yaml_path

def train_yolo(data_yaml, epochs=100, imgsz=640, batch=16, model_size='n', output_dir=None):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        output_dir: Directory to save training results and charts
    """
    available, YOLO = check_ultralytics()
    if not available:
        return None
    
    import json
    if MATPLOTLIB_AVAILABLE:
        import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"🚀 Starting YOLOv8 training...")
    print(f"   Model: yolov8{model_size}.pt")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    
    # Train with verbose output
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='tomato_detector',
        project='runs/detect',
        verbose=True
    )
    
    print(f"✅ Training complete!")
    print(f"   Best model: runs/detect/tomato_detector/weights/best.pt")
    print(f"   Last model: runs/detect/tomato_detector/weights/last.pt")
    
    # Extract training metrics from results
    try:
        # YOLO results contain metrics in results.results_dict
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        
        # Also try to get metrics from CSV file
        csv_path = Path('runs/detect/tomato_detector/results.csv')
        if csv_path.exists() and PANDAS_AVAILABLE:
            df = pd.read_csv(csv_path)
            
            # Extract metrics
            epochs_list = df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df)))
            train_box_loss = df['train/box_loss'].tolist() if 'train/box_loss' in df.columns else []
            train_cls_loss = df['train/cls_loss'].tolist() if 'train/cls_loss' in df.columns else []
            train_dfl_loss = df['train/dfl_loss'].tolist() if 'train/dfl_loss' in df.columns else []
            val_box_loss = df['val/box_loss'].tolist() if 'val/box_loss' in df.columns else []
            val_cls_loss = df['val/cls_loss'].tolist() if 'val/cls_loss' in df.columns else []
            val_dfl_loss = df['val/dfl_loss'].tolist() if 'val/dfl_loss' in df.columns else []
            precision = df['metrics/precision(B)'].tolist() if 'metrics/precision(B)' in df.columns else []
            recall = df['metrics/recall(B)'].tolist() if 'metrics/recall(B)' in df.columns else []
            map50 = df['metrics/mAP50(B)'].tolist() if 'metrics/mAP50(B)' in df.columns else []
            map50_95 = df['metrics/mAP50-95(B)'].tolist() if 'metrics/mAP50-95(B)' in df.columns else []
            
            # Save metrics to JSON
            metrics_data = {
                'epochs': epochs_list,
                'train_box_loss': train_box_loss,
                'train_cls_loss': train_cls_loss,
                'train_dfl_loss': train_dfl_loss,
                'train_total_loss': [(b + c + d) for b, c, d in zip(train_box_loss, train_cls_loss, train_dfl_loss)],
                'val_box_loss': val_box_loss,
                'val_cls_loss': val_cls_loss,
                'val_dfl_loss': val_dfl_loss,
                'val_total_loss': [(b + c + d) for b, c, d in zip(val_box_loss, val_cls_loss, val_dfl_loss)],
                'precision': precision,
                'recall': recall,
                'map50': map50,
                'map50_95': map50_95
            }
            
            # Save metrics JSON
            metrics_json_path = 'runs/detect/tomato_detector/training_metrics.json'
            if output_dir:
                metrics_json_path = os.path.join(output_dir, 'training_metrics.json')
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"📊 Training metrics saved to {metrics_json_path}")
            
            # Create training charts
            if len(epochs_list) > 0 and MATPLOTLIB_AVAILABLE:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Plot 1: Training and Validation Loss
                ax1 = axes[0, 0]
                if train_box_loss:
                    ax1.plot(epochs_list, train_box_loss, label='Train Box Loss', linewidth=2)
                if val_box_loss:
                    ax1.plot(epochs_list, val_box_loss, label='Val Box Loss', linewidth=2, linestyle='--')
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.set_title('Box Loss', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Total Loss
                ax2 = axes[0, 1]
                if metrics_data['train_total_loss']:
                    ax2.plot(epochs_list, metrics_data['train_total_loss'], label='Train Total Loss', linewidth=2)
                if metrics_data['val_total_loss']:
                    ax2.plot(epochs_list, metrics_data['val_total_loss'], label='Val Total Loss', linewidth=2, linestyle='--')
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Loss', fontsize=12)
                ax2.set_title('Total Loss (Box + Class + DFL)', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Precision and Recall
                ax3 = axes[1, 0]
                if precision:
                    ax3.plot(epochs_list, precision, label='Precision', linewidth=2, color='green')
                if recall:
                    ax3.plot(epochs_list, recall, label='Recall', linewidth=2, color='orange')
                ax3.set_xlabel('Epoch', fontsize=12)
                ax3.set_ylabel('Score', fontsize=12)
                ax3.set_title('Precision & Recall', fontsize=14, fontweight='bold')
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 1])
                
                # Plot 4: mAP
                ax4 = axes[1, 1]
                if map50:
                    ax4.plot(epochs_list, map50, label='mAP@0.5', linewidth=2, color='blue')
                if map50_95:
                    ax4.plot(epochs_list, map50_95, label='mAP@0.5:0.95', linewidth=2, color='red')
                ax4.set_xlabel('Epoch', fontsize=12)
                ax4.set_ylabel('mAP', fontsize=12)
                ax4.set_title('Mean Average Precision', fontsize=14, fontweight='bold')
                ax4.legend(fontsize=10)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim([0, 1])
                
                plt.tight_layout()
                
                # Save charts
                chart_path = 'runs/detect/tomato_detector/training_curves.png'
                if output_dir:
                    chart_path = os.path.join(output_dir, 'training_curves.png')
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                print(f"📊 Training curves saved to {chart_path}")
                
                # Also save to common location for web interface
                common_chart_path = 'training_curves.png'
                plt.savefig(common_chart_path, dpi=150, bbox_inches='tight')
                
                # Save to models/tomato/ if exists
                models_dir = os.path.join('models', 'tomato')
                if os.path.exists(models_dir):
                    model_chart_path = os.path.join(models_dir, 'training_curves.png')
                    plt.savefig(model_chart_path, dpi=150, bbox_inches='tight')
                    print(f"📊 Training curves also saved to {model_chart_path}")
                
                plt.close()
                
    except Exception as e:
        print(f"⚠️  Warning: Could not generate training charts: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for tomato detection')
    parser.add_argument('--dataset', type=str, default='datasets/tomato',
                       help='Path to classification dataset (default: datasets/tomato)')
    parser.add_argument('--output', type=str, default='datasets/tomato_yolo',
                       help='Output path for YOLO dataset (default: datasets/tomato_yolo)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)')
    parser.add_argument('--convert-only', action='store_true',
                       help='Only convert dataset, do not train')
    
    args = parser.parse_args()
    
    # Check if ultralytics is available
    if not args.convert_only:
        available, _ = check_ultralytics()
        if not available:
            print("\n💡 Tip: You can convert the dataset now and train later:")
            print(f"   python train_yolo.py --dataset {args.dataset} --output {args.output} --convert-only")
            sys.exit(1)
    
    # Convert dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    output_path = convert_classification_to_yolo(dataset_path, args.output)
    
    # Create data.yaml
    data_yaml = create_data_yaml(output_path, output_path)
    
    if args.convert_only:
        print("\n✅ Dataset conversion complete!")
        print(f"   Next steps:")
        print(f"   1. Annotate images with bounding boxes using LabelImg")
        print(f"   2. Train model: python train_yolo.py --dataset {args.dataset} --output {args.output}")
        return
    
    # Train model
    train_yolo(data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, 
               model_size=args.model, output_dir=None)

if __name__ == '__main__':
    main()

