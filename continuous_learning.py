#!/usr/bin/env python3
"""
Continuous Learning Handler
===========================
Manages feedback data and learning statistics.
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = 'learning_data/feedback.json'
LEARNING_IMAGES_DIR = 'learning_data/new_images'
IMAGE_METADATA_FILE = 'learning_data/image_metadata.json'  # Stores prediction metadata for images
MIN_FEEDBACK_FOR_RETRAIN = 10  # Minimum feedback entries before suggesting retraining

def ensure_data_dir():
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump([], f)

def save_feedback(args):
    ensure_data_dir()
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'image_path': args.image,
        'predicted_class': args.predicted,
        'correct_class': args.correct,
        'confidence': float(args.confidence)
    }
    
    try:
        with open(FEEDBACK_FILE, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
            data.append(feedback_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            
        print(json.dumps({'success': True, 'message': 'Feedback saved successfully'}))
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Error saving feedback: {str(e)}'}))
        sys.exit(1)

def get_stats():
    """Get learning statistics and return as JSON"""
    ensure_data_dir()
    
    try:
        # Load feedback data
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                try:
                    feedback_data = json.load(f)
                except json.JSONDecodeError:
                    feedback_data = []
        else:
            feedback_data = []
        
        # Count statistics
        total_feedback = len(feedback_data)
        incorrect_predictions = sum(1 for item in feedback_data if item['predicted_class'] != item['correct_class'])
        
        # Count learning images
        learning_images = 0
        if os.path.exists(LEARNING_IMAGES_DIR):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                learning_images += len(glob.glob(os.path.join(LEARNING_IMAGES_DIR, '**', ext), recursive=True))
        
        # Determine if retraining is needed
        retraining_needed = "Yes" if total_feedback >= MIN_FEEDBACK_FOR_RETRAIN and incorrect_predictions > 0 else "No"
        
        # Calculate accuracy if we have data
        if total_feedback > 0:
            correct_predictions = total_feedback - incorrect_predictions
            accuracy = (correct_predictions / total_feedback) * 100
            last_update = feedback_data[-1]['timestamp'] if feedback_data else "Never"
        else:
            accuracy = 0
            last_update = "Never"
        
        # Return as JSON for web interface
        stats = {
            'total_feedback': total_feedback,
            'incorrect_predictions': incorrect_predictions,
            'learning_images': learning_images,
            'retraining_needed': retraining_needed,
            'accuracy': round(accuracy, 1),
            'last_update': last_update
        }
        
        print(json.dumps(stats))
        
    except Exception as e:
        error_stats = {
            'total_feedback': 0,
            'incorrect_predictions': 0,
            'learning_images': 0,
            'retraining_needed': 'No',
            'accuracy': 0,
            'last_update': 'Never',
            'error': str(e)
        }
        print(json.dumps(error_stats))
        sys.exit(1)

def retrain_model():
    """Trigger model retraining using feedback data"""
    ensure_data_dir()
    
    try:
        # Load feedback data
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                try:
                    feedback_data = json.load(f)
                except json.JSONDecodeError:
                    feedback_data = []
        else:
            feedback_data = []
        
        if len(feedback_data) < MIN_FEEDBACK_FOR_RETRAIN:
            print(json.dumps({
                'success': False,
                'error': f'Not enough feedback data. Need at least {MIN_FEEDBACK_FOR_RETRAIN} entries, have {len(feedback_data)}'
            }))
            sys.exit(1)
        
        # Organize feedback by correct class for dataset creation
        class_mapping = {
            'not_ready': 'unripe',
            'ready': 'ripe',
            'spoilt': 'spoilt'
        }
        
        # Create dataset structure from feedback
        dataset_dir = Path('learning_data/retraining_dataset')
        dataset_dir.mkdir(exist_ok=True)
        
        for class_name in ['not_ready', 'ready', 'spoilt']:
            class_dir = dataset_dir / class_mapping.get(class_name, class_name)
            class_dir.mkdir(exist_ok=True)
        
        # Copy images to appropriate class folders
        copied_count = 0
        for entry in feedback_data:
            if entry['predicted_class'] != entry['correct_class']:  # Only use incorrect predictions
                source_path = Path(entry['image_path'])
                if source_path.exists():
                    target_class = class_mapping.get(entry['correct_class'], entry['correct_class'])
                    target_dir = dataset_dir / target_class
                    target_path = target_dir / source_path.name
                    
                    # Copy file
                    import shutil
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
        
        print(json.dumps({
            'success': True,
            'message': f'Dataset prepared with {copied_count} images. Ready for retraining.',
            'dataset_path': str(dataset_dir),
            'images_copied': copied_count
        }))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Error preparing retraining dataset: {str(e)}'
        }))
        sys.exit(1)

def get_recent_images(limit=10):
    """Get list of recent test images with prediction metadata"""
    try:
        images = []
        
        # Load image metadata if available
        metadata = {}
        if os.path.exists(IMAGE_METADATA_FILE):
            try:
                with open(IMAGE_METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        if os.path.exists(LEARNING_IMAGES_DIR):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in glob.glob(os.path.join(LEARNING_IMAGES_DIR, '**', ext), recursive=True):
                    stat = os.stat(img_path)
                    img_data = {
                        'path': img_path,
                        'name': os.path.basename(img_path),
                        'timestamp': stat.st_mtime,
                        'size': stat.st_size
                    }
                    
                    # Add prediction metadata if available
                    if img_path in metadata:
                        img_data['predicted_class'] = metadata[img_path].get('predicted_class', '')
                        img_data['confidence'] = metadata[img_path].get('confidence', 0.0)
                    
                    images.append(img_data)
        
        # Sort by timestamp (newest first) and limit
        images.sort(key=lambda x: x['timestamp'], reverse=True)
        images = images[:limit]
        
        # Convert timestamp to readable format
        for img in images:
            img['date'] = datetime.fromtimestamp(img['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        return images
    except Exception as e:
        return []

def save_image_metadata(image_path, predicted_class=None, confidence=None):
    """Save prediction metadata for an image"""
    ensure_data_dir()
    
    try:
        if os.path.exists(IMAGE_METADATA_FILE):
            with open(IMAGE_METADATA_FILE, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {}
        else:
            metadata = {}
        
        if image_path not in metadata:
            metadata[image_path] = {}
        
        if predicted_class is not None:
            metadata[image_path]['predicted_class'] = predicted_class
        if confidence is not None:
            metadata[image_path]['confidence'] = float(confidence)
        metadata[image_path]['last_updated'] = datetime.now().isoformat()
        
        with open(IMAGE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        # Non-fatal error, just log it
        pass

def delete_image(image_path):
    """Delete an image and its metadata"""
    try:
        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            return {'success': False, 'error': 'Image file not found'}
        
        # Remove from metadata
        if os.path.exists(IMAGE_METADATA_FILE):
            try:
                with open(IMAGE_METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {}
            
            if image_path in metadata:
                del metadata[image_path]
                with open(IMAGE_METADATA_FILE, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return {'success': True, 'message': 'Image and metadata deleted successfully'}
        
    except Exception as e:
        return {'success': False, 'error': f'Error deleting image: {str(e)}'}

def main():
    parser = argparse.ArgumentParser(description='Continuous Learning Handler')
    parser.add_argument('--action', required=True, choices=['feedback', 'stats', 'retrain', 'recent_images', 'save_metadata', 'delete_image'], help='Action to perform')
    
    # Feedback arguments
    parser.add_argument('--image', help='Path to image')
    parser.add_argument('--predicted', help='Predicted class')
    parser.add_argument('--correct', help='Correct class')
    parser.add_argument('--confidence', type=float, help='Confidence score')
    parser.add_argument('--limit', type=int, default=10, help='Limit for recent images')
    
    args = parser.parse_args()
    
    if args.action == 'feedback':
        if not all([args.image, args.predicted, args.correct, args.confidence is not None]):
            print(json.dumps({'success': False, 'error': 'Missing arguments for feedback action'}))
            sys.exit(1)
        save_feedback(args)
    elif args.action == 'stats':
        get_stats()
    elif args.action == 'retrain':
        retrain_model()
    elif args.action == 'recent_images':
        images = get_recent_images(args.limit)
        print(json.dumps({'success': True, 'images': images}))
    elif args.action == 'save_metadata':
        if args.image and args.predicted is not None:
            save_image_metadata(args.image, args.predicted, args.confidence)
            print(json.dumps({'success': True, 'message': 'Metadata saved'}))
        else:
            print(json.dumps({'success': False, 'error': 'Missing image or predicted class'}))
            sys.exit(1)
    elif args.action == 'delete_image':
        if not args.image:
            print(json.dumps({'success': False, 'error': 'Missing image path'}))
            sys.exit(1)
        result = delete_image(args.image)
        print(json.dumps(result))
        if not result.get('success'):
            sys.exit(1)

if __name__ == "__main__":
    main()
