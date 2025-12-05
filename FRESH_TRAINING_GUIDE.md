# Fresh Training Setup Guide
## AI Tomato Sorter - Clean Start

**Date:** December 5, 2025  
**Status:** ✅ System cleaned and ready for new training

---

## 🎯 What Was Done

### 1. **Backup Created**
- All old models, datasets, and training runs backed up to: `backup_20251205_172707/`
- You can restore from backup if needed

### 2. **Cleaned Directories**
- ✅ Removed old YOLO model: `runs/detect/tomato_detector/`
- ✅ Removed old ResNet model: `models/tomato/`
- ✅ Removed unreliable datasets: `datasets/tomato/` and `datasets/tomato_yolo/`
- ✅ Fresh directories created and ready

### 3. **System Status**
- ✅ Server restarted with clean state
- ✅ No cached models in memory
- ✅ Ready for new dataset upload

---

## 📊 Why the Old Model Failed

The previous model was predicting "spoilt" for everything because:

**Imbalanced Training Data:**
- Class 0 (not_ready): 1,298 instances (22%)
- Class 1 (ready): 1,975 instances (33%)  
- Class 2 (spoilt): 2,613 instances (45%)

The model learned to favor "spoilt" predictions because it saw almost **twice as many spoilt examples** as not_ready examples.

---

## 🎓 Best Practices for New Training Data

### 1. **Balanced Dataset**
Aim for roughly equal numbers of each class:
- **not_ready** (unripe/green): ~33% of images
- **ready** (ripe/red): ~33% of images
- **spoilt** (overripe/damaged): ~33% of images

**Recommended minimum:** 200-300 images per class (600-900 total)

### 2. **Image Quality Guidelines**

#### ✅ Good Images:
- Clear, well-lit tomatoes
- Variety of angles and backgrounds
- Different lighting conditions (natural, artificial)
- Various tomato sizes
- Consistent labeling criteria

#### ❌ Avoid:
- Blurry or out-of-focus images
- Extreme lighting (too dark/bright)
- Inconsistent labeling (same tomato labeled differently)
- Too many images from the same batch

### 3. **Labeling Criteria**

Define clear rules for each class:

**not_ready (Class 0):**
- Mostly green color
- Firm texture
- No red/orange coloring

**ready (Class 1):**
- Mostly red/orange color
- Firm but slightly soft
- Ready for harvest

**spoilt (Class 2):**
- Overripe, mushy, or damaged
- Brown spots or mold
- Cracked or rotting

### 4. **Dataset Structure**

For **YOLO training** (recommended):
```
datasets/
└── tomato_yolo/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

For **ResNet training**:
```
datasets/
└── tomato/
    ├── train/
    │   ├── not_ready/
    │   ├── ready/
    │   └── spoilt/
    ├── val/
    │   ├── not_ready/
    │   ├── ready/
    │   └── spoilt/
    └── test/
        ├── not_ready/
        ├── ready/
        └── spoilt/
```

---

## 🚀 Next Steps

### Step 1: Collect New Images
1. Take photos of tomatoes in all three categories
2. Ensure balanced distribution (equal numbers per class)
3. Use consistent lighting and backgrounds
4. Aim for 200-300 images per class minimum

### Step 2: Organize Dataset
1. Create dataset folder structure
2. Split images: 70% train, 20% val, 10% test
3. Label images correctly according to defined criteria

### Step 3: Upload to System
1. Navigate to http://localhost:5000/training
2. Click "Create New Dataset"
3. Upload images to appropriate class folders
4. Verify dataset balance before training

### Step 4: Train Model
1. Select dataset from training page
2. Choose model type:
   - **YOLO** (recommended): Detects and classifies in one model
   - **ResNet**: Classification only, needs color detection
3. Configure training parameters:
   - Epochs: 50-100 (more for better accuracy)
   - Batch size: 16-32 (adjust based on memory)
4. Start training and monitor progress

### Step 5: Validate Model
1. Test with new images (not in training set)
2. Check confusion matrix for balanced predictions
3. Verify all three classes are predicted correctly
4. If biased, collect more data for underrepresented classes

---

## 📝 Training Configuration Recommendations

### For YOLO:
```yaml
model_type: yolo
model_size: n  # Start with nano for speed
epochs: 100
batch_size: 16
confidence_threshold: 0.5
```

### For ResNet:
```yaml
model_type: resnet
epochs: 50
batch_size: 32
learning_rate: 0.001
```

---

## 🔧 System Configuration

### Current State:
- ✅ Server running on http://localhost:5000
- ✅ Camera detected (index 0, 640x480)
- ✅ Hardware controller available (simulation mode)
- ✅ YOLO support enabled (will activate when model is trained)
- ✅ Thread-safe YOLO inference (segfault issue fixed)

### Fixed Issues:
- ✅ Segmentation fault during YOLO inference (SocketIO threading mode fixed)
- ✅ Thread-safe YOLO detector initialization
- ✅ Proper error handling for model testing

---

## 📚 Additional Resources

### Data Collection Tips:
1. Use natural lighting when possible
2. Capture tomatoes from multiple angles
3. Include variety in size and shape
4. Document your labeling criteria
5. Review and validate labels before training

### Annotation Tools (for YOLO):
- **LabelImg**: https://github.com/heartexlabs/labelImg
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/ (online, easy to use)

### Training Monitoring:
- Watch training logs in real-time on the web interface
- Check mAP (mean Average Precision) - aim for >90%
- Monitor loss curves - should decrease steadily
- Validate on test set after training

---

## ⚠️ Important Notes

1. **Don't rush the data collection** - quality over quantity
2. **Balance is critical** - equal representation of all classes
3. **Consistent labeling** - use the same criteria throughout
4. **Test thoroughly** - validate on real-world images
5. **Iterate if needed** - retrain with more data if predictions are poor

---

## 🎉 You're Ready!

Your system is now clean and ready for fresh training with reliable data. Take your time collecting good quality, balanced training data, and you'll get much better results!

**Good luck with your new training! 🍅🤖**
