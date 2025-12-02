# GPU Training → CPU Inference Guide

## ✅ Yes, It Works!

You can **train on GPU (RTX 4080 Super)** and **run inference on CPU laptop** - this is fully supported and very common!

## How It Works

### 1. **Model Portability**
- YOLO models (`.pt` files) are **device-agnostic**
- Same model file works on GPU and CPU
- Ultralytics automatically handles device selection

### 2. **Training on GPU (RTX 4080 Super)**
- **Faster training** (10-50x speedup vs CPU)
- **Larger batch sizes** possible (32-64 vs 8-16)
- **Model file is portable** - no GPU-specific code

### 3. **Inference on CPU (Laptop)**
- **Same model file** works directly
- **Automatic device detection** - uses CPU if no GPU
- **No conversion needed**

## Training Setup (GPU - RTX 4080 Super)

### Prerequisites
```bash
# Install CUDA-enabled PyTorch (if not already)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics
pip install ultralytics

# Verify GPU is detected
python -c "from ultralytics import YOLO; import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Training on GPU
When you train via web interface or `train_yolo.py`:
- **Ultralytics automatically uses GPU** if available
- **No special configuration needed**
- **Model saves as `.pt` file** (works on any device)

### Recommended Settings for RTX 4080 Super
```
Model Size: Small (s) or Medium (m)
Epochs: 100-200
Batch Size: 32-64 (GPU can handle large batches)
Image Size: 640
```

**Training Time Estimates (RTX 4080 Super):**
- Nano (n), 100 epochs: ~5-10 minutes
- Small (s), 100 epochs: ~10-20 minutes
- Medium (m), 100 epochs: ~20-40 minutes

## Inference Setup (CPU Laptop)

### Transfer Model File
After training on GPU, copy the model file to your laptop:
```bash
# On GPU machine, model is saved at:
runs/detect/tomato_detector/weights/best.pt

# Copy to laptop (via USB, network, cloud, etc.)
# Place in one of these locations:
models/tomato/best.pt
# OR
models/tomato/yolov8_tomato.pt
```

### Running on CPU Laptop
```bash
# Install Ultralytics (CPU version is fine)
pip install ultralytics

# Start web interface
python web_interface.py
```

**The system will:**
1. ✅ Automatically detect CPU
2. ✅ Load the model (trained on GPU)
3. ✅ Run inference on CPU
4. ✅ Work perfectly (just slower than GPU)

### Performance on CPU Laptop

**Expected Inference Speed:**
- **Nano (n)**: 30-60 FPS (CPU)
- **Small (s)**: 15-30 FPS (CPU)
- **Medium (m)**: 8-15 FPS (CPU)

**For Real-time Detection:**
- ✅ **Nano or Small** work well on CPU
- ⚠️ **Medium** may be slow for real-time
- ❌ **Large/XLarge** too slow for real-time on CPU

## Complete Workflow

### Step 1: Train on GPU (RTX 4080 Super)
```bash
# On GPU machine
cd /path/to/project

# Start web interface
python web_interface.py

# Go to Training Dashboard
# Select: YOLO, Small (s), 100 epochs, batch 32
# Start training

# After training, model saved at:
# runs/detect/tomato_detector/weights/best.pt
```

### Step 2: Copy Model to Laptop
```bash
# Option 1: USB drive
cp runs/detect/tomato_detector/weights/best.pt /media/usb/
# Then copy to laptop

# Option 2: Network (SCP)
scp runs/detect/tomato_detector/weights/best.pt user@laptop:/path/to/project/models/tomato/

# Option 3: Cloud (Google Drive, Dropbox, etc.)
# Upload best.pt, download on laptop
```

### Step 3: Run on CPU Laptop
```bash
# On laptop
cd /path/to/project

# Place model file:
# models/tomato/best.pt

# Start web interface
python web_interface.py

# System automatically uses the model!
```

## Technical Details

### How Ultralytics Handles Devices

**Training (GPU):**
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
# Automatically uses GPU if available
results = model.train(...)
# Model saved as .pt file (device-agnostic)
```

**Inference (CPU):**
```python
from ultralytics import YOLO
model = YOLO('best.pt')  # Model trained on GPU
# Automatically uses CPU if no GPU
results = model.predict(image)
# Works perfectly!
```

### Model File Format
- **`.pt` files** are PyTorch checkpoints
- **Contain model weights** (not device-specific)
- **Ultralytics handles device mapping** automatically
- **No conversion needed** between GPU/CPU

## Best Practices

### 1. **Model Size Selection**
- **Train on GPU**: Use Small (s) or Medium (m) for best accuracy
- **Run on CPU**: Small (s) is best balance, Nano (n) if speed critical

### 2. **Batch Size**
- **Training (GPU)**: 32-64 (RTX 4080 Super can handle it)
- **Inference (CPU)**: Always 1 (single image at a time)

### 3. **Model Transfer**
- ✅ **Same model file** works everywhere
- ✅ **No conversion needed**
- ✅ **Automatic device detection**

### 4. **Performance Optimization for CPU**
- Use **Nano (n)** or **Small (s)** models
- Reduce image size if needed (default 640 is fine)
- Close other applications to free CPU

## Troubleshooting

### "CUDA out of memory" during training
**Solution:**
- Reduce batch size (32 → 16 → 8)
- Use smaller model (Medium → Small → Nano)
- Close other GPU applications

### "Model not found" on laptop
**Solution:**
- Check model file location: `models/tomato/best.pt`
- Verify file was copied correctly
- Check file permissions

### "Slow inference on CPU"
**Solution:**
- Use Nano (n) or Small (s) model
- This is normal - CPU is slower than GPU
- 15-30 FPS is acceptable for real-time detection

### "Model works on GPU but not CPU"
**This shouldn't happen**, but if it does:
```python
# Force CPU usage
from ultralytics import YOLO
model = YOLO('best.pt')
model.to('cpu')  # Explicitly set to CPU
```

## Summary

✅ **Train on RTX 4080 Super** (fast training)  
✅ **Copy `.pt` model file** to laptop  
✅ **Run on CPU laptop** (works automatically)  
✅ **No conversion needed**  
✅ **Same model file** works on both  

**Recommended Workflow:**
1. Train **Small (s)** model on GPU (batch 32-64, 100-150 epochs)
2. Copy `best.pt` to laptop
3. Place in `models/tomato/best.pt`
4. Run web interface on laptop
5. System automatically uses CPU for inference

**Performance:**
- **Training (GPU)**: 10-40 minutes
- **Inference (CPU)**: 15-30 FPS (Small model)
- **Perfect for real-time detection!**

## Quick Reference

| Task | Device | Model Size | Batch Size | Time |
|------|--------|------------|------------|------|
| **Training** | RTX 4080 Super | Small (s) | 32-64 | 10-20 min |
| **Inference** | CPU Laptop | Small (s) | 1 | 15-30 FPS |

**You're all set! Train on GPU, run on CPU - it just works!** 🚀

