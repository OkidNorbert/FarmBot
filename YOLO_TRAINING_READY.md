# YOLO Training Now Available in Web Interface! ✅

## What's New

The training page now supports **YOLO training** in addition to ResNet training!

## How to Use

### 1. Access Training Page
- Go to the Training Dashboard
- Select a dataset
- Click "Start Training"

### 2. Choose Model Type
In the training modal, you'll see:
- **Model Type** dropdown:
  - **ResNet (Classification)** - Original method, classifies pre-cropped images
  - **YOLO (Detection + Classification)** - New! Detects and classifies in one model

### 3. YOLO-Specific Options
When you select YOLO:
- **Model Size** dropdown appears:
  - **Nano (n)** - Fastest, smallest (recommended for testing)
  - **Small (s)** - Good balance
  - **Medium (m)** - Better accuracy
  - **Large (l)** - High accuracy
  - **XLarge (x)** - Best accuracy, slowest

### 4. Training Process

**For YOLO:**
1. System checks if `ultralytics` is installed
2. Automatically converts your classification dataset to YOLO format
3. Trains YOLOv8 model
4. Saves model to `runs/detect/tomato_detector/weights/best.pt`

**For ResNet:**
1. Uses existing `auto_train.py` script
2. Trains ResNet18 classifier
3. Saves model to `models/tomato/best_model.pth`

## Requirements

### For YOLO Training:
```bash
pip install ultralytics
```

### For ResNet Training:
- PyTorch (already installed)

## Dataset Format

### Current Format (Classification):
```
datasets/tomato/
  train/
    Ripe/
    Unripe/
    Old/
    Damaged/
  val/
    Ripe/
    Unripe/
    Old/
    Damaged/
```

### YOLO Format (Auto-converted):
```
datasets/tomato_yolo/
  images/
    train/
    val/
  labels/
    train/  (.txt files with bounding boxes)
    val/
  data.yaml
```

**Note**: Auto-conversion creates placeholder labels (whole image as bounding box). For best results, annotate images with actual bounding boxes using LabelImg.

## Training Parameters

### ResNet:
- **Epochs**: Number of training iterations (default: 30)
- **Batch Size**: Images per batch (default: 32)
- **Learning Rate**: Training speed (default: 0.001)

### YOLO:
- **Epochs**: Number of training iterations (default: 30)
- **Batch Size**: Images per batch (default: 32)
- **Model Size**: n, s, m, l, or x (default: n)
- **Learning Rate**: Auto-configured by YOLO

## Model Output

### YOLO Models:
- **Best Model**: `runs/detect/tomato_detector/weights/best.pt`
- **Last Model**: `runs/detect/tomato_detector/weights/last.pt`
- System automatically finds and uses these models

### ResNet Models:
- **Best Model**: `models/tomato/best_model.pth`
- System uses this for classification

## Benefits of YOLO

✅ **Single Model**: Detection + classification in one
✅ **Better Accuracy**: More accurate bounding boxes
✅ **Faster Inference**: Optimized for real-time
✅ **Better Multi-Tomato**: Handles multiple tomatoes better
✅ **End-to-End**: Trained together, not separately

## Troubleshooting

### "Ultralytics not installed"
```bash
pip install ultralytics
```

### "Dataset conversion failed"
- Check dataset structure is correct
- Ensure images are in train/val folders with class subfolders

### "Training failed"
- Check logs in training status
- Verify dataset has enough images (50+ per class recommended)
- Check disk space

## Next Steps

1. **Install Ultralytics** (when you have internet):
   ```bash
   pip install ultralytics
   ```

2. **Train YOLO Model**:
   - Go to Training Dashboard
   - Select dataset
   - Choose "YOLO (Detection + Classification)"
   - Select model size (start with "nano")
   - Start training

3. **For Best Results**:
   - Annotate images with bounding boxes using LabelImg
   - Use more images (100+ per class)
   - Train for more epochs (100+)

4. **Use Trained Model**:
   - System automatically detects YOLO model
   - No manual configuration needed
   - Just train and use!

## Summary

✅ **Training page now supports YOLO**
✅ **Automatic dataset conversion**
✅ **Easy model selection in UI**
✅ **Automatic model detection after training**

**Ready to train YOLO models through the web interface!**

