#!/usr/bin/env python3
"""
Continuous Learning Handler
===========================
Manages feedback data and learning statistics.
"""

import argparse
import json
import os
from datetime import datetime

FEEDBACK_FILE = 'learning_data/feedback.json'

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
            
        print("✅ Feedback saved successfully")
        
    except Exception as e:
        print(f"❌ Error saving feedback: {str(e)}")
        sys.exit(1)

def get_stats():
    ensure_data_dir()
    
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        
        total_samples = len(data)
        if total_samples == 0:
            print("Total Samples: 0")
            print("Accuracy: 0%")
            print("Last Update: Never")
            return

        correct_predictions = sum(1 for item in data if item['predicted_class'] == item['correct_class'])
        accuracy = (correct_predictions / total_samples) * 100
        last_update = data[-1]['timestamp']
        
        print(f"Total Samples: {total_samples}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Last Update: {last_update}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Continuous Learning Handler')
    parser.add_argument('--action', required=True, choices=['feedback', 'stats'], help='Action to perform')
    
    # Feedback arguments
    parser.add_argument('--image', help='Path to image')
    parser.add_argument('--predicted', help='Predicted class')
    parser.add_argument('--correct', help='Correct class')
    parser.add_argument('--confidence', type=float, help='Confidence score')
    
    args = parser.parse_args()
    
    if args.action == 'feedback':
        if not all([args.image, args.predicted, args.correct, args.confidence]):
            print("❌ Missing arguments for feedback action")
            sys.exit(1)
        save_feedback(args)
    elif args.action == 'stats':
        get_stats()

if __name__ == "__main__":
    main()
