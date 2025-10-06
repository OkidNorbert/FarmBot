#!/usr/bin/env python3
"""
Quick Training Launcher
======================

One-command training for new crops.
Just run: python quick_train.py your_crop_name
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def quick_train(crop_name, dataset_path=None, epochs=30):
    """Quick training for a new crop"""
    
    if dataset_path is None:
        dataset_path = f"datasets/{crop_name}"
    
    print(f"🚀 QUICK TRAINING: {crop_name}")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"📁 Creating dataset structure for {crop_name}...")
        result = subprocess.run([
            "python", "setup_new_crop.py", crop_name
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create dataset structure: {result.stderr}")
            return False
        
        print(f"✅ Dataset structure created at: {dataset_path}")
        print(f"📝 Please add your {crop_name} images to the class folders")
        print(f"   Then run this command again to start training")
        return True
    
    # Check if dataset has images
    image_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
    
    if image_count == 0:
        print(f"📸 No images found in {dataset_path}")
        print(f"Please add your {crop_name} images to the class folders")
        print(f"Supported formats: .jpg, .jpeg, .png")
        return True
    
    print(f"📊 Found {image_count} images in dataset")
    print(f"🚀 Starting automated training...")
    
    # Run automated training
    cmd = [
        "python", "auto_train.py",
        "--dataset_path", dataset_path,
        "--crop_name", crop_name,
        "--epochs", str(epochs)
    ]
    
    print(f"💻 Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: models/{crop_name}")
        print(f"🔧 Test with: python models/{crop_name}/{crop_name}_inference.py --image your_test_image.jpg")
        return True
    else:
        print(f"❌ Training failed with return code: {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick training for new crops")
    parser.add_argument("crop_name", help="Name of the crop to train")
    parser.add_argument("--dataset_path", help="Path to dataset (default: datasets/crop_name)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    
    args = parser.parse_args()
    
    success = quick_train(
        crop_name=args.crop_name,
        dataset_path=args.dataset_path,
        epochs=args.epochs
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())