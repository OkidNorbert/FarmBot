#!/usr/bin/env python3
"""
Quick Start Annotation Script
Starts the simple annotation tool for your tomato dataset
"""

import os
import sys
from pathlib import Path

def main():
    print("🍅 Starting Tomato Annotation Tool")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("tomato_dataset")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please extract your dataset first.")
        return
    
    # Check if images exist
    train_images = dataset_path / "images" / "train"
    if not train_images.exists():
        print("❌ Training images not found!")
        return
    
    # Create labels directory
    train_labels = dataset_path / "labels" / "train"
    train_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Found dataset at: {dataset_path}")
    print(f"✅ Training images: {train_images}")
    print(f"✅ Labels directory: {train_labels}")
    
    # Count images
    image_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    print(f"📊 Total images to annotate: {image_count}")
    
    # Check existing annotations
    existing_labels = len(list(train_labels.glob("*.txt")))
    print(f"📊 Existing annotations: {existing_labels}")
    
    print("\n🚀 Starting annotation tool...")
    print("Controls:")
    print("  Mouse: Draw bounding boxes")
    print("  Keys: 0,1,2 - Select class (not_ready, ready, spoilt)")
    print("  Keys: n - Next image")
    print("  Keys: p - Previous image")
    print("  Keys: s - Save annotations")
    print("  Keys: d - Delete last box")
    print("  Keys: c - Clear all boxes")
    print("  Keys: q - Quit")
    print("\n🚀 Starting annotation tool...")
    
    # Start annotation
    os.system(f"python simple_annotator.py --images {train_images} --labels {train_labels}")

if __name__ == "__main__":
    main()
