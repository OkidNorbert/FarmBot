#!/usr/bin/env python3
"""
AI Tomato Sorter - Model Export Script
Exports trained YOLOv8 models to ONNX and TFLite formats for Raspberry Pi deployment
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
import onnx
import tensorflow as tf
from tqdm import tqdm

def export_to_onnx(model_path, output_dir="exported_models", imgsz=640):
    """Export YOLOv8 model to ONNX format"""
    print("🔄 Exporting to ONNX format...")
    
    # Load model
    model = YOLO(model_path)
    
    # Export to ONNX
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        optimize=True,
        simplify=True,
        opset=11
    )
    
    # Move to output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    final_path = output_path / "tomato_sorter.onnx"
    if os.path.exists(onnx_path):
        os.rename(onnx_path, final_path)
        print(f"✅ ONNX model saved to: {final_path}")
        
        # Validate ONNX model
        try:
            onnx_model = onnx.load(str(final_path))
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation passed")
        except Exception as e:
            print(f"⚠️  ONNX validation warning: {e}")
        
        return final_path
    else:
        print(f"❌ ONNX export failed")
        return None

def export_to_tflite(model_path, output_dir="exported_models", imgsz=640):
    """Export YOLOv8 model to TFLite format"""
    print("🔄 Exporting to TFLite format...")
    
    # Load model
    model = YOLO(model_path)
    
    # Export to TFLite
    tflite_path = model.export(
        format='tflite',
        imgsz=imgsz,
        int8=True,  # Enable quantization
        data=None,  # Will use default quantization
        nms=True
    )
    
    # Move to output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    final_path = output_path / "tomato_sorter.tflite"
    if os.path.exists(tflite_path):
        os.rename(tflite_path, final_path)
        print(f"✅ TFLite model saved to: {final_path}")
        return final_path
    else:
        print(f"❌ TFLite export failed")
        return None

def create_quantized_tflite(model_path, representative_data, output_dir="exported_models"):
    """Create quantized TFLite model with representative dataset"""
    print("🔄 Creating quantized TFLite model...")
    
    try:
        # Load the original model
        model = YOLO(model_path)
        
        # Export with quantization using representative data
        tflite_path = model.export(
            format='tflite',
            int8=True,
            data=representative_data,
            imgsz=640
        )
        
        # Move to output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        final_path = output_path / "tomato_sorter_quantized.tflite"
        if os.path.exists(tflite_path):
            os.rename(tflite_path, final_path)
            print(f"✅ Quantized TFLite model saved to: {final_path}")
            return final_path
        else:
            print(f"❌ Quantized TFLite export failed")
            return None
            
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return None

def benchmark_model(model_path, input_shape=(1, 3, 640, 640), num_runs=100):
    """Benchmark model inference speed"""
    print(f"⚡ Benchmarking model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Warmup runs
    print("   Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark runs
    print(f"   Running {num_runs} inference cycles...")
    times = []
    
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    print(f"📊 Benchmark Results:")
    print(f"   Average inference time: {avg_time*1000:.2f} ms")
    print(f"   Standard deviation: {std_time*1000:.2f} ms")
    print(f"   Min time: {min_time*1000:.2f} ms")
    print(f"   Max time: {max_time*1000:.2f} ms")
    print(f"   FPS: {fps:.2f}")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps
    }

def compare_model_sizes(model_paths):
    """Compare file sizes of different model formats"""
    print("📏 Comparing model sizes...")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   {Path(model_path).name}: {size_mb:.2f} MB")
        else:
            print(f"   {Path(model_path).name}: Not found")

def create_representative_dataset(data_yaml, num_samples=100):
    """Create representative dataset for quantization"""
    print("📊 Creating representative dataset for quantization...")
    
    import yaml
    from PIL import Image
    import cv2
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    train_path = dataset_path / config['train']
    
    # Get sample images
    image_files = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    sample_files = image_files[:num_samples]
    
    print(f"   Using {len(sample_files)} images for quantization")
    
    # Create data.yaml for quantization
    quant_data = {
        'path': str(dataset_path),
        'train': config['train'],
        'val': config['val'],
        'test': config['test'],
        'names': config['names'],
        'nc': config['nc']
    }
    
    quant_yaml_path = "quant_data.yaml"
    with open(quant_yaml_path, 'w') as f:
        yaml.dump(quant_data, f)
    
    return quant_yaml_path

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 models for deployment')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--output_dir', type=str, default='exported_models', help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for export')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'tflite'], 
                       choices=['onnx', 'tflite'], help='Export formats')
    parser.add_argument('--quantize', action='store_true', help='Create quantized TFLite model')
    parser.add_argument('--data_yaml', type=str, help='Data YAML for quantization')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model performance')
    parser.add_argument('--compare_sizes', action='store_true', help='Compare model file sizes')
    
    args = parser.parse_args()
    
    print("🍅 AI Tomato Sorter - Model Export")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    exported_models = []
    
    # Export to ONNX
    if 'onnx' in args.formats:
        onnx_path = export_to_onnx(args.model, args.output_dir, args.imgsz)
        if onnx_path:
            exported_models.append(onnx_path)
    
    # Export to TFLite
    if 'tflite' in args.formats:
        tflite_path = export_to_tflite(args.model, args.output_dir, args.imgsz)
        if tflite_path:
            exported_models.append(tflite_path)
    
    # Create quantized TFLite if requested
    if args.quantize and args.data_yaml:
        quant_data_yaml = create_representative_dataset(args.data_yaml)
        quant_path = create_quantized_tflite(args.model, quant_data_yaml, args.output_dir)
        if quant_path:
            exported_models.append(quant_path)
    
    # Benchmark models
    if args.benchmark:
        print("\n⚡ Benchmarking models...")
        for model_path in exported_models:
            if model_path.endswith('.pt'):
                benchmark_model(model_path)
    
    # Compare model sizes
    if args.compare_sizes:
        print("\n📏 Model size comparison:")
        compare_model_sizes(exported_models)
    
    print(f"\n✅ Export completed!")
    print(f"📁 Exported models saved to: {args.output_dir}")
    for model_path in exported_models:
        print(f"   - {Path(model_path).name}")

if __name__ == "__main__":
    main()
