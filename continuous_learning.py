#!/usr/bin/env python3
"""
Continuous Learning System for Tomato Classifier
Automatically learns from uploaded test images to improve the model
"""
import os
import sys
import json
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

# Define the ResNet-based model (must match the training script)
class AutoCropClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AutoCropClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ContinuousLearner:
    def __init__(self, model_path, metadata_path, learning_data_path="learning_data"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.learning_data_path = learning_data_path
        self.setup_learning_environment()
        
    def setup_learning_environment(self):
        """Setup directories for continuous learning"""
        os.makedirs(self.learning_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.learning_data_path, "new_images"), exist_ok=True)
        os.makedirs(os.path.join(self.learning_data_path, "feedback"), exist_ok=True)
        
        # Create class directories
        for class_name in ["not_ready", "ready", "spoilt"]:
            os.makedirs(os.path.join(self.learning_data_path, "new_images", class_name), exist_ok=True)
    
    def save_feedback(self, image_path, predicted_class, user_feedback, confidence):
        """Save user feedback for continuous learning"""
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "predicted_class": predicted_class,
            "user_feedback": user_feedback,
            "confidence": confidence,
            "needs_retraining": user_feedback != predicted_class
        }
        
        feedback_file = os.path.join(self.learning_data_path, "feedback", f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print(f"💾 Feedback saved: {feedback_file}")
        return feedback_file
    
    def add_image_to_learning_set(self, image_path, correct_class):
        """Add image to learning dataset with correct class"""
        # Copy image to appropriate class directory
        class_dir = os.path.join(self.learning_data_path, "new_images", correct_class)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"learn_{timestamp}_{os.path.basename(image_path)}"
        dest_path = os.path.join(class_dir, filename)
        
        shutil.copy2(image_path, dest_path)
        print(f"📚 Added to learning set: {dest_path}")
        return dest_path
    
    def get_learning_statistics(self):
        """Get statistics about the learning data"""
        stats = {
            "total_feedback": 0,
            "incorrect_predictions": 0,
            "learning_images": {"not_ready": 0, "ready": 0, "spoilt": 0},
            "retraining_needed": False
        }
        
        # Count feedback files
        feedback_dir = os.path.join(self.learning_data_path, "feedback")
        if os.path.exists(feedback_dir):
            feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
            stats["total_feedback"] = len(feedback_files)
            
            # Count incorrect predictions
            for feedback_file in feedback_files:
                with open(os.path.join(feedback_dir, feedback_file), 'r') as f:
                    feedback_data = json.load(f)
                    if feedback_data.get("needs_retraining", False):
                        stats["incorrect_predictions"] += 1
                        stats["retraining_needed"] = True
        
        # Count learning images
        learning_dir = os.path.join(self.learning_data_path, "new_images")
        for class_name in ["not_ready", "ready", "spoilt"]:
            class_dir = os.path.join(learning_dir, class_name)
            if os.path.exists(class_dir):
                stats["learning_images"][class_name] = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return stats
    
    def should_retrain(self, min_images=10, min_incorrect=5):
        """Check if model should be retrained"""
        stats = self.get_learning_statistics()
        
        total_learning_images = sum(stats["learning_images"].values())
        
        return (
            stats["retraining_needed"] and 
            (total_learning_images >= min_images or stats["incorrect_predictions"] >= min_incorrect)
        )
    
    def prepare_retraining_data(self):
        """Prepare data for retraining by combining original and learning data"""
        print("🔄 Preparing retraining data...")
        
        # Create retraining directory structure
        retrain_dir = os.path.join(self.learning_data_path, "retrain_data")
        os.makedirs(retrain_dir, exist_ok=True)
        os.makedirs(os.path.join(retrain_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(retrain_dir, "val"), exist_ok=True)
        
        # Copy original dataset
        original_dataset = "datasets/tomato"
        if os.path.exists(original_dataset):
            print("📚 Copying original dataset...")
            for split in ["train", "val"]:
                original_split = os.path.join(original_dataset, split)
                if os.path.exists(original_split):
                    dest_split = os.path.join(retrain_dir, split)
                    if os.path.exists(dest_split):
                        shutil.rmtree(dest_split)
                    shutil.copytree(original_split, dest_split)
        
        # Add learning images
        learning_dir = os.path.join(self.learning_data_path, "new_images")
        for class_name in ["not_ready", "ready", "spoilt"]:
            class_dir = os.path.join(learning_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"📚 Adding {len(images)} learning images for {class_name}...")
                    
                    # Add to training set (80% train, 20% val)
                    train_class_dir = os.path.join(retrain_dir, "train", class_name)
                    val_class_dir = os.path.join(retrain_dir, "val", class_name)
                    os.makedirs(train_class_dir, exist_ok=True)
                    os.makedirs(val_class_dir, exist_ok=True)
                    
                    for i, image in enumerate(images):
                        src = os.path.join(class_dir, image)
                        if i < len(images) * 0.8:  # 80% to train
                            dst = os.path.join(train_class_dir, image)
                        else:  # 20% to val
                            dst = os.path.join(val_class_dir, image)
                        shutil.copy2(src, dst)
        
        return retrain_dir
    
    def retrain_model(self, retrain_data_path, epochs=10, learning_rate=0.001):
        """Retrain the model with new data"""
        print("🔄 Starting continuous learning retraining...")
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create data.yaml for retraining
        data_yaml_path = os.path.join(retrain_data_path, "data.yaml")
        data_config = {
            'path': os.path.abspath(retrain_data_path),
            'train': 'train',
            'val': 'val',
            'nc': metadata['num_classes'],
            'names': metadata['class_names']
        }
        
        with open(data_yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_config, f, sort_keys=False)
        
        # Run retraining using the existing auto_train.py
        retrain_script = "auto_train.py"
        if os.path.exists(retrain_script):
            import subprocess
            cmd = [
                sys.executable, retrain_script,
                "--dataset", retrain_data_path,
                "--epochs", str(epochs),
                "--output_model", "continuous_learning_model.pth"
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Continuous learning retraining completed!")
                return True
            else:
                print(f"❌ Retraining failed: {result.stderr}")
                return False
        else:
            print("❌ auto_train.py not found!")
            return False

def main():
    parser = argparse.ArgumentParser(description="Continuous Learning System")
    parser.add_argument("--action", choices=["feedback", "stats", "retrain", "check"], 
                       default="check", help="Action to perform")
    parser.add_argument("--image", type=str, help="Image path for feedback")
    parser.add_argument("--predicted", type=str, help="Predicted class")
    parser.add_argument("--correct", type=str, help="Correct class")
    parser.add_argument("--confidence", type=float, help="Prediction confidence")
    args = parser.parse_args()
    
    # Initialize continuous learner
    model_path = "models/tomato/best_model.pth"
    metadata_path = "models/tomato/training_metadata.json"
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("❌ Model or metadata not found!")
        return
    
    learner = ContinuousLearner(model_path, metadata_path)
    
    if args.action == "feedback":
        if not all([args.image, args.predicted, args.correct, args.confidence]):
            print("❌ Missing required arguments for feedback")
            return
        
        # Save feedback
        learner.save_feedback(args.image, args.predicted, args.correct, args.confidence)
        
        # Add to learning set if prediction was wrong
        if args.predicted != args.correct:
            learner.add_image_to_learning_set(args.image, args.correct)
            print("📚 Image added to learning set for future retraining")
    
    elif args.action == "stats":
        stats = learner.get_learning_statistics()
        print("📊 Continuous Learning Statistics:")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Incorrect predictions: {stats['incorrect_predictions']}")
        print(f"   Learning images: {stats['learning_images']}")
        print(f"   Retraining needed: {stats['retraining_needed']}")
    
    elif args.action == "retrain":
        if learner.should_retrain():
            print("🔄 Retraining triggered...")
            retrain_data_path = learner.prepare_retraining_data()
            success = learner.retrain_model(retrain_data_path)
            if success:
                print("✅ Model retrained successfully!")
            else:
                print("❌ Retraining failed!")
        else:
            print("ℹ️  Not enough data for retraining yet")
    
    elif args.action == "check":
        stats = learner.get_learning_statistics()
        print("🔍 Continuous Learning Status:")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Incorrect predictions: {stats['incorrect_predictions']}")
        print(f"   Learning images: {sum(stats['learning_images'].values())}")
        
        if learner.should_retrain():
            print("🔄 Retraining recommended!")
        else:
            print("ℹ️  Continue collecting feedback...")

if __name__ == "__main__":
    main()
