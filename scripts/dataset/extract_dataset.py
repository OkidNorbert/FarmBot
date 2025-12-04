#!/usr/bin/env python3
"""
AI Tomato Sorter - Dataset Extraction Utility
Extracts and organizes datasets from ZIP or TAR archives
"""

import os
import sys
import zipfile
import tarfile
import shutil
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_zip(zip_path, extract_to):
    """Extract ZIP archive"""
    logger.info(f"Extracting ZIP archive: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✅ Successfully extracted to: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract ZIP: {e}")
        return False

def extract_tar(tar_path, extract_to):
    """Extract TAR archive"""
    logger.info(f"Extracting TAR archive: {tar_path}")
    
    try:
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
        logger.info(f"✅ Successfully extracted to: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract TAR: {e}")
        return False

def organize_dataset(source_dir, target_dir):
    """Organize extracted dataset into proper structure"""
    logger.info(f"Organizing dataset from {source_dir} to {target_dir}")
    
    # Create target directory structure
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    dirs_to_create = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    for dir_path in dirs_to_create:
        (target_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Find and organize images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.rglob(f'*{ext}'))
        image_files.extend(source_path.rglob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Find and organize labels
    label_files = list(source_path.rglob('*.txt'))
    logger.info(f"Found {len(label_files)} label files")
    
    # Create mapping for organization
    organized_images = 0
    organized_labels = 0
    
    # Try to organize by existing structure first
    for img_file in image_files:
        try:
            # Check if it's in a train/val/test subdirectory
            relative_path = img_file.relative_to(source_path)
            path_parts = relative_path.parts
            
            if 'train' in path_parts:
                split = 'train'
            elif 'val' in path_parts or 'validation' in path_parts:
                split = 'val'
            elif 'test' in path_parts:
                split = 'test'
            else:
                # Default to train if no split specified
                split = 'train'
            
            # Copy image
            target_img = target_path / f'images/{split}' / img_file.name
            shutil.copy2(img_file, target_img)
            organized_images += 1
            
            # Look for corresponding label
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                target_label = target_path / f'labels/{split}' / label_file.name
                shutil.copy2(label_file, target_label)
                organized_labels += 1
                
        except Exception as e:
            logger.warning(f"Failed to organize {img_file}: {e}")
    
    logger.info(f"✅ Organized {organized_images} images and {organized_labels} labels")
    return organized_images, organized_labels

def create_data_yaml(dataset_path, output_path="data.yaml"):
    """Create data.yaml file for the organized dataset"""
    logger.info(f"Creating data.yaml for dataset at {dataset_path}")
    
    dataset_path = Path(dataset_path)
    
    # Count images in each split
    train_count = len(list((dataset_path / 'images' / 'train').glob('*')))
    val_count = len(list((dataset_path / 'images' / 'val').glob('*')))
    test_count = len(list((dataset_path / 'images' / 'test').glob('*')))
    
    # Create data.yaml content
    yaml_content = f"""# data.yaml for Ultralytics YOLOv8
path: {dataset_path.absolute()}  # absolute path to dataset root
train: images/train  # relative to 'path'
val: images/val      # relative to 'path'  
test: images/test    # relative to 'path'

# number of classes
nc: 3

# class names
names:
  0: not_ready
  1: ready
  2: spoilt

# dataset statistics
train_images: {train_count}
val_images: {val_count}
test_images: {test_count}
total_images: {train_count + val_count + test_count}
"""
    
    # Write data.yaml
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"✅ Created data.yaml with {train_count} train, {val_count} val, {test_count} test images")
    return output_path

def validate_dataset(dataset_path):
    """Validate the organized dataset"""
    logger.info(f"Validating dataset at {dataset_path}")
    
    dataset_path = Path(dataset_path)
    issues = []
    
    # Check directory structure
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            issues.append(f"Missing directory: {dir_path}")
    
    # Check for images and labels
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / f'images/{split}'
        label_dir = dataset_path / f'labels/{split}'
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*')))
            if img_count == 0:
                issues.append(f"No images found in {split}")
        else:
            issues.append(f"Images directory missing: {split}")
        
        if label_dir.exists():
            label_count = len(list(label_dir.glob('*.txt')))
            if label_count == 0:
                issues.append(f"No labels found in {split}")
        else:
            issues.append(f"Labels directory missing: {split}")
    
    if issues:
        logger.warning(f"⚠️  Found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    else:
        logger.info("✅ Dataset validation passed")
    
    return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description='Extract and organize dataset from archive')
    parser.add_argument('archive_path', help='Path to ZIP or TAR archive')
    parser.add_argument('--extract_to', default='extracted_dataset', help='Directory to extract to')
    parser.add_argument('--organize_to', default='tomato_dataset', help='Directory to organize dataset to')
    parser.add_argument('--create_yaml', action='store_true', help='Create data.yaml file')
    parser.add_argument('--validate', action='store_true', help='Validate organized dataset')
    parser.add_argument('--cleanup', action='store_true', help='Clean up extracted files after organization')
    
    args = parser.parse_args()
    
    print("🍅 AI Tomato Sorter - Dataset Extraction Utility")
    print("=" * 60)
    
    # Check if archive exists
    archive_path = Path(args.archive_path)
    if not archive_path.exists():
        logger.error(f"❌ Archive not found: {archive_path}")
        sys.exit(1)
    
    # Determine archive type
    archive_name = archive_path.name.lower()
    if archive_name.endswith('.zip'):
        extract_func = extract_zip
    elif any(archive_name.endswith(ext) for ext in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']):
        extract_func = extract_tar
    else:
        logger.error(f"❌ Unsupported archive format: {archive_path.suffix}")
        logger.info("Supported formats: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz")
        sys.exit(1)
    
    # Extract archive
    extract_to = Path(args.extract_to)
    extract_to.mkdir(exist_ok=True)
    
    if not extract_func(str(archive_path), str(extract_to)):
        sys.exit(1)
    
    # Organize dataset
    organize_to = Path(args.organize_to)
    organized_images, organized_labels = organize_dataset(extract_to, organize_to)
    
    if organized_images == 0:
        logger.error("❌ No images were organized. Check your dataset structure.")
        sys.exit(1)
    
    # Create data.yaml if requested
    if args.create_yaml:
        yaml_path = create_data_yaml(organize_to)
        logger.info(f"📄 Created data.yaml at: {yaml_path}")
    
    # Validate dataset if requested
    if args.validate:
        validate_dataset(organize_to)
    
    # Cleanup if requested
    if args.cleanup:
        logger.info("🧹 Cleaning up extracted files...")
        shutil.rmtree(extract_to)
        logger.info("✅ Cleanup completed")
    
    print(f"\n🎉 Dataset extraction completed!")
    print(f"📁 Organized dataset: {organize_to}")
    print(f"📊 Images organized: {organized_images}")
    print(f"📊 Labels organized: {organized_labels}")
    
    if args.create_yaml:
        print(f"📄 Data configuration: data.yaml")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Review your dataset: ls -la {organize_to}")
    print(f"   2. Train model: python train.py --data data.yaml")
    print(f"   3. Or use comprehensive training: python train/train_tomato_detector.py --data data.yaml")

if __name__ == "__main__":
    main()
