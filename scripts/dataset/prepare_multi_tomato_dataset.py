#!/usr/bin/env python3
"""
Dataset Preparation Script for Multi-Tomato Training
=====================================================
This script helps prepare a training dataset that includes:
1. Single-tomato images (keep as-is)
2. Multi-tomato images (crop individual tomatoes and label them)

The cropped tomatoes will be saved to the appropriate class folders.
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import argparse
import json
from tqdm import tqdm

def detect_tomatoes_in_image(image_path):
    """Detect tomato bounding boxes in an image using color-based detection"""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return []
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for tomatoes
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    combined_mask = red_mask + green_mask + orange_mask
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.6 < aspect_ratio < 1.6:  # Circular requirement
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3 and w > 40 and h > 40:
                        # Add padding
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2 * padding)
                        h = min(frame.shape[0] - y, h + 2 * padding)
                        bboxes.append((x, y, w, h))
    
    return bboxes

def crop_and_save_tomatoes(image_path, output_dir, class_name, min_size=50):
    """Crop individual tomatoes from an image and save them"""
    bboxes = detect_tomatoes_in_image(image_path)
    
    if len(bboxes) == 0:
        return 0
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        return 0
    
    saved_count = 0
    base_name = Path(image_path).stem
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # Ensure minimum size
        if w < min_size or h < min_size:
            continue
        
        # Crop the tomato
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            continue
        
        # Save cropped tomato
        output_path = output_dir / f"{base_name}_tomato_{i+1}.jpg"
        cv2.imwrite(str(output_path), crop)
        saved_count += 1
    
    return saved_count

def prepare_dataset(source_dir, output_dir, split='train'):
    """Prepare dataset from source directory"""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Class mapping
    class_mapping = {
        'Unripe': 'not_ready',
        'Ripe': 'ready',
        'Old': 'spoilt',
        'Damaged': 'spoilt',
        'not_ready': 'not_ready',
        'ready': 'ready',
        'spoilt': 'spoilt'
    }
    
    # Create output structure
    split_dir = output_path / split
    for class_name in ['not_ready', 'ready', 'spoilt']:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'single_tomato_images': 0,
        'multi_tomato_images': 0,
        'total_crops': 0,
        'by_class': {'not_ready': 0, 'ready': 0, 'spoilt': 0}
    }
    
    # Process each class folder
    source_split_dir = source_path / split
    if not source_split_dir.exists():
        print(f"Warning: {source_split_dir} does not exist")
        return stats
    
    for class_folder in source_split_dir.iterdir():
        if not class_folder.is_dir():
            continue
        
        source_class = class_folder.name
        if source_class not in class_mapping:
            print(f"Skipping unknown class: {source_class}")
            continue
        
        target_class = class_mapping[source_class]
        target_dir = split_dir / target_class
        
        print(f"\nProcessing {source_class} -> {target_class}...")
        
        # Get all images
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        for img_path in tqdm(image_files, desc=f"  {source_class}"):
            stats['total_images'] += 1
            
            # Detect tomatoes in image
            bboxes = detect_tomatoes_in_image(img_path)
            
            if len(bboxes) == 0:
                # No tomatoes detected, copy whole image (might be single tomato that wasn't detected)
                shutil.copy2(img_path, target_dir / img_path.name)
                stats['single_tomato_images'] += 1
                stats['by_class'][target_class] += 1
            elif len(bboxes) == 1:
                # Single tomato detected, copy whole image
                shutil.copy2(img_path, target_dir / img_path.name)
                stats['single_tomato_images'] += 1
                stats['by_class'][target_class] += 1
            else:
                # Multiple tomatoes detected, crop each one
                stats['multi_tomato_images'] += 1
                crops_saved = crop_and_save_tomatoes(img_path, target_dir, target_class)
                stats['total_crops'] += crops_saved
                stats['by_class'][target_class] += crops_saved
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Prepare multi-tomato training dataset')
    parser.add_argument('--source', required=True, help='Source dataset directory')
    parser.add_argument('--output', required=True, help='Output dataset directory')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], 
                       help='Dataset split to process')
    parser.add_argument('--all-splits', action='store_true', 
                       help='Process all splits (train, val, test)')
    
    args = parser.parse_args()
    
    print("🍅 Multi-Tomato Dataset Preparation")
    print("=" * 50)
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    
    if args.all_splits:
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")
        
        stats = prepare_dataset(args.source, args.output, split)
        all_stats[split] = stats
        
        print(f"\n{split.upper()} Split Statistics:")
        print(f"  Total source images: {stats['total_images']}")
        print(f"  Single-tomato images: {stats['single_tomato_images']}")
        print(f"  Multi-tomato images: {stats['multi_tomato_images']}")
        print(f"  Total cropped tomatoes: {stats['total_crops']}")
        print(f"  Final dataset size: {sum(stats['by_class'].values())}")
        print(f"  By class:")
        for class_name, count in stats['by_class'].items():
            print(f"    {class_name}: {count}")
    
    # Save statistics
    stats_file = Path(args.output) / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📊 Statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()

