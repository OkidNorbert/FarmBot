#!/usr/bin/env python3
"""
Demo Script for Automated Crop Training
======================================

This script demonstrates how easy it is to train new crop models
using the automated training system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display the result"""
    print(f"\n🔄 {description}")
    print(f"💻 Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Error!")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def demo_automated_training():
    """Demonstrate the automated training process"""
    print("🤖 AUTOMATED CROP TRAINING DEMO")
    print("=" * 60)
    print("This demo shows how easy it is to train new crop models!")
    print("=" * 60)
    
    # Step 1: Setup a new crop dataset
    print("\n📋 STEP 1: Setting up a new crop dataset")
    print("Let's create a dataset structure for 'demo_crop'")
    
    success = run_command(
        "python setup_new_crop.py demo_crop",
        "Creating dataset structure for demo_crop"
    )
    
    if not success:
        print("❌ Failed to create dataset structure")
        return False
    
    # Step 2: Show the created structure
    print("\n📁 STEP 2: Dataset structure created")
    print("Let's see what was created:")
    
    run_command(
        "find datasets/demo_crop -type d | sort",
        "Showing created folder structure"
    )
    
    # Step 3: Show how to add images (simulated)
    print("\n📸 STEP 3: Adding images to your dataset")
    print("In real usage, you would:")
    print("1. Add your crop images to datasets/demo_crop/ripe/")
    print("2. Add your crop images to datasets/demo_crop/unripe/")
    print("3. Add your crop images to datasets/demo_crop/overripe/")
    print("4. Add your crop images to datasets/demo_crop/damaged/")
    print("\nFor this demo, let's copy some existing tomato images:")
    
    # Copy some existing images for demo
    run_command(
        "cp tomato_dataset/train/Ripe/*.jpg datasets/demo_crop/ripe/ 2>/dev/null || echo 'No tomato images found, that\'s okay for demo'",
        "Copying sample images (if available)"
    )
    
    # Step 4: Show training command
    print("\n🚀 STEP 4: Training the model")
    print("Now you would run the automated training:")
    print("python auto_train.py --dataset_path datasets/demo_crop --crop_name demo_crop")
    print("\nThis would:")
    print("✅ Analyze your dataset")
    print("✅ Create train/validation/test splits")
    print("✅ Train the AI model")
    print("✅ Generate inference script")
    print("✅ Save all training metadata")
    
    # Step 5: Show the complete workflow
    print("\n🎯 COMPLETE WORKFLOW SUMMARY")
    print("=" * 60)
    print("1. Setup:     python setup_new_crop.py your_crop_name")
    print("2. Add images: Place images in the class folders")
    print("3. Train:      python auto_train.py --dataset_path datasets/your_crop_name --crop_name your_crop_name")
    print("4. Test:       python models/your_crop_name/your_crop_name_inference.py --image test_image.jpg")
    print("5. Deploy:     Use the generated model in your application")
    
    print("\n🎉 That's it! Your AI model is ready!")
    print("The system handles everything automatically:")
    print("✅ Dataset analysis and validation")
    print("✅ Automatic data splitting")
    print("✅ Model training with best practices")
    print("✅ Performance monitoring")
    print("✅ Ready-to-use inference script")
    print("✅ Complete training metadata")
    
    return True

def show_examples():
    """Show example usage for different crops"""
    print("\n🌱 EXAMPLE USAGE FOR DIFFERENT CROPS")
    print("=" * 60)
    
    examples = [
        {
            "crop": "strawberry",
            "classes": ["ripe", "unripe", "overripe", "damaged"],
            "use_case": "Quality control for strawberry harvesting"
        },
        {
            "crop": "apple",
            "classes": ["fresh", "bruised", "rotten", "small"],
            "use_case": "Apple sorting and quality assessment"
        },
        {
            "crop": "corn",
            "classes": ["ready", "not_ready", "diseased", "damaged"],
            "use_case": "Corn maturity detection"
        },
        {
            "crop": "lettuce",
            "classes": ["fresh", "wilted", "diseased", "harvest_ready"],
            "use_case": "Lettuce quality monitoring"
        }
    ]
    
    for example in examples:
        print(f"\n🍅 {example['crop'].title()}")
        print(f"   Classes: {', '.join(example['classes'])}")
        print(f"   Use case: {example['use_case']}")
        print(f"   Command: python auto_train.py --dataset_path datasets/{example['crop']} --crop_name {example['crop']}")

def main():
    print("🤖 AUTOMATED CROP TRAINING SYSTEM")
    print("=" * 60)
    print("This system makes it incredibly easy to train AI models for new crops!")
    print("Just organize your photos and run one command!")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ["auto_train.py", "setup_new_crop.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the current directory")
        return 1
    
    # Run the demo
    success = demo_automated_training()
    
    if success:
        show_examples()
        
        print("\n🎯 READY TO USE!")
        print("=" * 60)
        print("Your automated training system is ready!")
        print("Check AUTO_TRAINING_GUIDE.md for detailed instructions.")
        print("Start with: python setup_new_crop.py your_crop_name")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
