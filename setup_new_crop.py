#!/usr/bin/env python3
"""
Quick Setup Script for New Crop Training
========================================

This script helps you set up a new crop dataset for automated training.
It creates the proper folder structure and guides you through the process.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

def create_crop_structure(crop_name, base_path="datasets"):
    """Create the proper folder structure for a new crop"""
    crop_path = Path(base_path) / crop_name
    crop_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🌱 Setting up crop dataset: {crop_name}")
    print(f"📁 Location: {crop_path.absolute()}")
    
    # Create example structure
    example_classes = ["ripe", "unripe", "overripe", "damaged"]
    
    for class_name in example_classes:
        class_dir = crop_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create a README in each class folder
        readme_content = f"""# {class_name.title()} {crop_name.title()}

Place your {class_name} {crop_name} images in this folder.

Supported formats: .jpg, .jpeg, .png
Recommended: At least 50-100 images per class for good results
"""
        with open(class_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    # Create main README
    main_readme = f"""# {crop_name.title()} Dataset

## Folder Structure
```
{crop_name}/
├── ripe/           # Ripe {crop_name} images
├── unripe/        # Unripe {crop_name} images  
├── overripe/      # Overripe {crop_name} images
└── damaged/       # Damaged {crop_name} images
```

## How to Use
1. Add your images to the appropriate class folders
2. Run: `python auto_train.py --dataset_path {crop_path} --crop_name {crop_name}`
3. Wait for training to complete
4. Test with: `python models/{crop_name}/{crop_name}_inference.py --image path/to/test/image.jpg`

## Tips
- Use at least 50-100 images per class
- Ensure good lighting and clear images
- Include variety in angles, sizes, and conditions
- Remove any blurry or unclear images
"""
    
    with open(crop_path / "README.md", 'w') as f:
        f.write(main_readme)
    
    print(f"✅ Created dataset structure at: {crop_path}")
    print(f"📂 Class folders created: {', '.join(example_classes)}")
    print(f"📝 README files added to guide you")
    
    return crop_path

def main():
    parser = argparse.ArgumentParser(description="Setup new crop dataset structure")
    parser.add_argument("crop_name", help="Name of the crop (e.g., strawberry, apple, corn)")
    parser.add_argument("--base_path", default="datasets", 
                       help="Base directory for datasets (default: datasets)")
    
    args = parser.parse_args()
    
    crop_path = create_crop_structure(args.crop_name, args.base_path)
    
    print("\n🎯 Next Steps:")
    print(f"1. Add your {args.crop_name} images to the class folders in: {crop_path}")
    print(f"2. Run training: python auto_train.py --dataset_path {crop_path} --crop_name {args.crop_name}")
    print(f"3. Check the README.md files in each folder for guidance")

if __name__ == "__main__":
    main()
