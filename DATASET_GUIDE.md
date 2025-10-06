# 🍅 **Tomato Dataset Guide - Updated Structure**

## ✅ **Your Dataset is Ready!**

Your dataset has been successfully analyzed and configured for training.

### **📊 Dataset Statistics:**
- **Total Images**: 6,487 images
- **Training**: 5,832 images (90%)
- **Validation**: 655 images (10%)

### **🏷️ Class Distribution:**

#### **Training Set:**
- **Unripe**: 1,276 images → **not_ready** (Class 0)
- **Ripe**: 1,975 images → **ready** (Class 1)
- **Old**: 1,992 images → **spoilt** (Class 2)
- **Damaged**: 589 images → **spoilt** (Class 2)

#### **Validation Set:**
- **Unripe**: 143 images → **not_ready** (Class 0)
- **Ripe**: 220 images → **ready** (Class 1)
- **Old**: 222 images → **spoilt** (Class 2)
- **Damaged**: 70 images → **spoilt** (Class 2)

## 🚀 **Quick Start Training**

### **Step 1: Install Dependencies**
```bash
# Activate virtual environment
source tomato_sorter_env/bin/activate

# Install PyTorch (if not already installed)
pip install torch torchvision
```

### **Step 2: Train Your Model**
```bash
# Quick training (50 epochs)
python train_tomato_classifier.py --epochs 50 --batch_size 32

# Extended training (100 epochs)
python train_tomato_classifier.py --epochs 100 --batch_size 16

# Custom training
python train_tomato_classifier.py --epochs 80 --batch_size 32 --lr 0.001
```

### **Step 3: Test Your Model**
```bash
# Test with camera
python inference_classifier.py --source 0

# Test with single image
python inference_classifier.py --image path/to/tomato.jpg
```

## 🎯 **Class Mapping Explained**

### **Original Dataset Classes → Project Classes:**
- **Unripe** (1,419 total) → **not_ready** (Class 0)
- **Ripe** (2,195 total) → **ready** (Class 1)
- **Old** (2,214 total) → **spoilt** (Class 2)
- **Damaged** (659 total) → **spoilt** (Class 2)

### **Why This Mapping?**
- **Unripe**: Green tomatoes that need more time → **not_ready**
- **Ripe**: Red tomatoes ready for consumption → **ready**
- **Old + Damaged**: Overripe or spoiled tomatoes → **spoilt**

## 📁 **Dataset Structure**

```
tomato_dataset/
├── train/
│   ├── Unripe/     # 1,276 images → not_ready
│   ├── Ripe/       # 1,975 images → ready
│   ├── Old/        # 1,992 images → spoilt
│   └── Damaged/    # 589 images → spoilt
├── val/
│   ├── Unripe/     # 143 images → not_ready
│   ├── Ripe/       # 220 images → ready
│   ├── Old/        # 222 images → spoilt
│   └── Damaged/    # 70 images → spoilt
└── data.yaml       # Configuration file
```

## 🔧 **Training Configuration**

### **Model Architecture:**
- **Backbone**: ResNet18 (pretrained)
- **Classes**: 3 (not_ready, ready, spoilt)
- **Input Size**: 224x224 pixels
- **Augmentation**: Random flip, rotation, color jitter

### **Training Parameters:**
- **Epochs**: 50-100
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Scheduler**: StepLR (reduce by 0.1 every 20 epochs)

## 📈 **Expected Results**

### **Training Progress:**
- **Epoch 1-10**: Learning basic features
- **Epoch 10-30**: Improving accuracy
- **Epoch 30-50**: Fine-tuning
- **Epoch 50+**: Convergence

### **Target Performance:**
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Inference Speed**: <100ms per image

## 🚀 **Deployment Options**

### **Option 1: Raspberry Pi Deployment**
```bash
# Export to ONNX for faster inference
python export_model.py --model tomato_classifier.pth --format onnx

# Run on Raspberry Pi
python inference_pi.py --model tomato_classifier.onnx --source 0
```

### **Option 2: Web Interface**
```bash
# Start web interface
python web_inference.py --model tomato_classifier.pth --port 5000
```

### **Option 3: Real-time Sorting**
```bash
# Connect to robotic arm
python tomato_sorter.py --model tomato_classifier.pth --camera 0 --serial /dev/ttyUSB0
```

## 🔍 **Model Evaluation**

### **Check Training Progress:**
```bash
# View training curves
python -c "
import matplotlib.pyplot as plt
import numpy as np

# Load training history
# Plot accuracy and loss curves
"
```

### **Test Model Performance:**
```bash
# Test on validation set
python evaluate_model.py --model tomato_classifier.pth --dataset tomato_dataset/val

# Test on single image
python inference_classifier.py --image tomato_dataset/val/Ripe/sample.jpg
```

## 📚 **File Structure**

### **Training Files:**
- `train_tomato_classifier.py` - Main training script
- `inference_classifier.py` - Inference script
- `quick_train.py` - Dataset setup script
- `data.yaml` - Dataset configuration

### **Generated Files:**
- `tomato_classifier.pth` - Trained model
- `training_curves.png` - Training progress plots
- `data.yaml` - Dataset configuration

## 🎯 **Next Steps**

### **1. Start Training:**
```bash
python train_tomato_classifier.py --epochs 50 --batch_size 32
```

### **2. Monitor Progress:**
- Watch training accuracy improve
- Check validation accuracy
- Save best model

### **3. Test Model:**
```bash
python inference_classifier.py --image path/to/test/image.jpg
```

### **4. Deploy:**
- Export to ONNX for Raspberry Pi
- Set up robotic arm control
- Configure camera calibration

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   - Reduce batch size: `--batch_size 16`
   - Use CPU: `--device cpu`

2. **Slow Training:**
   - Use GPU: `--device cuda`
   - Reduce image size: `--imgsz 128`

3. **Poor Accuracy:**
   - Increase epochs: `--epochs 100`
   - Adjust learning rate: `--lr 0.0001`

## 🎉 **Ready to Train!**

Your dataset is perfectly organized and ready for training. The classification approach is much more efficient than object detection for your use case.

**Start training now:**
```bash
python train_tomato_classifier.py --epochs 50 --batch_size 32
```

**Your AI Tomato Sorter will learn to classify tomatoes into the three categories needed for automated sorting! 🍅🤖**
