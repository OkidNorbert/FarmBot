# Complete Guide: Training YOLO via Web Interface

## Prerequisites

### 1. Install Ultralytics (Required)
```bash
pip install ultralytics
```

**Check if installed:**
```bash
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed')"
```

### 2. Prepare Your Dataset

Your dataset should be in **classification format** (folders by class):
```
datasets/tomato/
  train/
    Ripe/        (or ripe/)
    Unripe/      (or unripe/)
    Old/         (or old/)
    Damaged/     (or damaged/)
  val/
    Ripe/
    Unripe/
    Old/
    Damaged/
```

**Minimum Requirements:**
- At least 50 images per class (recommended: 100+)
- Images in JPG, PNG, or JPEG format
- Train/val split (80/20 recommended)

## Step-by-Step Training Guide

### Step 1: Access Training Dashboard

1. **Start the web interface:**
   ```bash
   python web_interface.py
   ```

2. **Open in browser:**
   - Go to: `http://localhost:5000`
   - Click on **"Training"** tab or navigate to `/training`

### Step 2: Verify Dataset

1. **Check your dataset appears:**
   - Look in the "Datasets" section
   - Your dataset should be listed (e.g., "tomato")
   - Verify it has train/val folders with class subfolders

2. **If dataset not visible:**
   - Click "Create New Dataset" button
   - Or manually create folder structure:
     ```
     datasets/your_dataset_name/
       train/
         Ripe/
         Unripe/
         Old/
       val/
         Ripe/
         Unripe/
         Old/
     ```

### Step 3: Start Training

1. **Click "Start Training" button** next to your dataset

2. **Training Modal Opens** - Fill in the form:

   **Model Type:**
   - Select: **"YOLO (Detection + Classification)"**
   - This enables YOLO training

   **YOLO Model Size:**
   - **Nano (n)** - Fastest, smallest (recommended for testing)
   - **Small (s)** - Good balance
   - **Medium (m)** - Better accuracy
   - **Large (l)** - High accuracy
   - **XLarge (x)** - Best accuracy, slowest
   - **Recommendation**: Start with **Nano (n)** for first training

   **Training Parameters:**
   - **Epochs**: Number of training iterations
     - Start with: **50-100** epochs
     - More epochs = better accuracy (but slower)
   - **Batch Size**: Images per batch
     - Start with: **16** (reduce to 8 if out of memory)
     - Increase if you have GPU: **32-64**

3. **Click "Start Training"**

### Step 4: Monitor Training

**Training Status Card appears** showing:

1. **Progress Bar:**
   - Shows training progress (0-100%)
   - Updates in real-time

2. **Status Message:**
   - Current training phase
   - Epoch progress
   - Any errors or warnings

3. **Real-Time Logs:**
   - Color-coded log output:
     - 🟢 **YOLO messages** - Cyan
     - 🟡 **Epoch progress** - Yellow
     - 🔵 **Metrics (mAP, precision, recall)** - Blue
     - 🔴 **Errors** - Red
     - 🟢 **Success messages** - Green
   - Auto-scrolls to latest entries
   - Updates every 2 seconds

**What to Watch For:**
- ✅ "Converting dataset to YOLO format..." - Normal
- ✅ "Epoch 1/100..." - Training started
- ✅ "mAP50: 0.85" - Good metrics
- ⚠️ "ERROR: Ultralytics not installed" - Need to install
- ⚠️ "Dataset conversion failed" - Check dataset structure

### Step 5: Training Process

**What Happens Automatically:**

1. **Dataset Conversion:**
   - System converts classification format to YOLO format
   - Creates placeholder labels (whole image as bounding box)
   - Saves to: `datasets/tomato_yolo/`

2. **YOLO Training:**
   - Loads pretrained YOLOv8 model
   - Trains on your dataset
   - Saves checkpoints every epoch
   - Tracks metrics (loss, mAP, precision, recall)

3. **Model Saving:**
   - Best model: `runs/detect/tomato_detector/weights/best.pt`
   - Last model: `runs/detect/tomato_detector/weights/last.pt`

4. **Charts Generation:**
   - Automatically creates training curves
   - Saves to: `runs/detect/tomato_detector/training_curves.png`

### Step 6: View Results

**After Training Completes:**

1. **Training Status:**
   - Status changes to "Training completed successfully!"
   - Progress bar shows 100%

2. **Training Charts:**
   - "Training Results & Charts" section appears
   - Shows 4-panel chart with:
     - Box Loss (train vs validation)
     - Total Loss
     - Precision & Recall
     - mAP (Mean Average Precision)
   - Download button available

3. **Model Files:**
   - Best model saved automatically
   - System will automatically use it for detection

### Step 7: Test Your Model

1. **Go to Models section:**
   - Your trained model should appear
   - Shows model name and metadata

2. **Test the model:**
   - Click "Test Model" button
   - Upload a test image
   - See detection results
   - Check if it uses YOLO (look for "detection_method": "YOLO" in response)

## Training Parameters Explained

### Epochs
- **What it is**: Number of times model sees entire dataset
- **Recommended**: 
  - Start: 50-100 epochs
  - Production: 100-200 epochs
  - More = better accuracy (but diminishing returns)

### Batch Size
- **What it is**: Number of images processed together
- **Recommended**:
  - CPU: 8-16
  - GPU: 16-64
  - Larger = faster training (if memory allows)

### Model Size
- **Nano (n)**: 
  - Fastest inference
  - Smallest file size
  - Good for testing
  - Lower accuracy
- **Small (s)**: 
  - Good balance
  - Recommended for most use cases
- **Medium (m)**: 
  - Better accuracy
  - Slower inference
- **Large (l)**: 
  - High accuracy
  - Slow inference
- **XLarge (x)**: 
  - Best accuracy
  - Very slow
  - Large file size

## Understanding Training Metrics

### Loss Values
- **Lower is better**
- **Box Loss**: Bounding box prediction accuracy
- **Class Loss**: Classification accuracy
- **Total Loss**: Combined (Box + Class + DFL)

### mAP (Mean Average Precision)
- **Higher is better** (0-1 scale)
- **mAP@0.5**: IoU threshold 0.5
- **mAP@0.5:0.95**: Average across IoU 0.5-0.95
- **Good values**: >0.7 mAP@0.5

### Precision & Recall
- **Precision**: How many detections are correct
- **Recall**: How many tomatoes are found
- **Higher is better** (0-1 scale)
- **Good values**: >0.8 precision, >0.8 recall

## Troubleshooting

### "Ultralytics not installed"
**Solution:**
```bash
pip install ultralytics
```

### "Dataset conversion failed"
**Check:**
- Dataset folder structure is correct
- Images are in train/val folders
- Class folders exist (Ripe, Unripe, Old, Damaged)
- Images are valid (JPG, PNG, JPEG)

### "Training failed"
**Check logs for:**
- Out of memory → Reduce batch size
- Invalid dataset → Check dataset structure
- Missing images → Verify all images load correctly

### "No YOLO model found after training"
**Check:**
- Training completed successfully
- Model saved to: `runs/detect/tomato_detector/weights/best.pt`
- Copy to: `models/tomato/best.pt` (optional, system finds it automatically)

### Training is very slow
**Solutions:**
- Use smaller model size (nano instead of large)
- Reduce batch size
- Use GPU if available
- Reduce number of epochs for testing

### Low accuracy (mAP < 0.5)
**Solutions:**
- Train for more epochs
- Use larger model size
- Add more training images
- Annotate images with proper bounding boxes (not placeholder)

## Best Practices

### 1. Start Small
- First training: Use Nano model, 50 epochs, small batch
- Test if everything works
- Then scale up

### 2. Monitor Training
- Watch logs for errors
- Check metrics are improving
- Stop early if loss not decreasing

### 3. Dataset Quality
- More images = better accuracy
- Variety is important (different lighting, angles, backgrounds)
- Balanced classes (similar number per class)

### 4. Annotate Properly (For Best Results)
- Current system uses placeholder labels (whole image)
- For production: Use LabelImg to annotate bounding boxes
- Proper annotations = much better accuracy

### 5. Save Checkpoints
- System saves best model automatically
- Can resume training if interrupted
- Best model is always saved

## Quick Reference

### Training Checklist
- [ ] Ultralytics installed
- [ ] Dataset prepared (train/val split)
- [ ] At least 50 images per class
- [ ] Web interface running
- [ ] Navigate to Training Dashboard
- [ ] Select dataset
- [ ] Choose "YOLO (Detection + Classification)"
- [ ] Select model size (start with Nano)
- [ ] Set epochs (start with 50-100)
- [ ] Set batch size (start with 16)
- [ ] Click "Start Training"
- [ ] Monitor logs and progress
- [ ] Wait for completion
- [ ] View charts and results
- [ ] Test model

### File Locations After Training
- **Best Model**: `runs/detect/tomato_detector/weights/best.pt`
- **Last Model**: `runs/detect/tomato_detector/weights/last.pt`
- **Training Charts**: `runs/detect/tomato_detector/training_curves.png`
- **Metrics JSON**: `runs/detect/tomato_detector/training_metrics.json`
- **Results CSV**: `runs/detect/tomato_detector/results.csv`

## Next Steps After Training

1. **Test the Model:**
   - Use "Test Model" feature
   - Upload test images
   - Verify detections are accurate

2. **Use in Production:**
   - Model is automatically detected by system
   - All detection functions will use YOLO
   - No manual configuration needed

3. **Improve Accuracy:**
   - Train for more epochs
   - Use larger model size
   - Add more training data
   - Annotate with proper bounding boxes

4. **Monitor Performance:**
   - Check detection accuracy in real use
   - Collect feedback
   - Retrain with new data if needed

## Summary

✅ **Simple Process:**
1. Install ultralytics
2. Prepare dataset
3. Go to Training Dashboard
4. Select YOLO model type
5. Start training
6. Monitor logs
7. View results

✅ **Automatic:**
- Dataset conversion
- Model saving
- Chart generation
- Model detection

✅ **No Conflicts:**
- System uses YOLO automatically
- ResNet only as fallback
- Clean priority system

**You're ready to train YOLO via the web interface!**

