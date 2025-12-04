#!/usr/bin/env python3
"""
Auto Train Wrapper
==================
Bridges the Web Interface calls to the actual training script.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Auto Train Wrapper')
    parser.add_argument('--dataset_path', required=True, help='Path to dataset')
    parser.add_argument('--crop_name', required=True, help='Name of the crop')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"🔄 Starting Auto Training for {args.crop_name}")
    print(f"📂 Dataset: {args.dataset_path}")
    
    # Construct command for actual training script
    # We need to map arguments from web interface to train_tomato_classifier.py
    
    cmd = [
        sys.executable, 'train_tomato_classifier.py',
        '--dataset', args.dataset_path,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.learning_rate)
    ]
    
    print(f"🚀 Executing: {' '.join(cmd)}")
    
    try:
        # Run the training script and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output to stdout so web interface can capture it
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Auto Training Wrapper Completed Successfully")
        else:
            print(f"\n❌ Auto Training Wrapper Failed with code {process.returncode}")
            sys.exit(process.returncode)
            
    except Exception as e:
        print(f"\n❌ Error in wrapper: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
