# 🍅 Multi-Tomato Training Guide

## Why Train on Multiple Tomatoes?

**YES, you should include images with multiple tomatoes in your training dataset!**

### Benefits:

1. **Matches Real-World Scenarios**: Your system will encounter multiple tomatoes in the same frame during operation
2. **Better Generalization**: The model learns to classify tomatoes in various contexts (cluttered backgrounds, partial occlusions)
3. **Aligns with Inference**: Since your inference system crops individual tomatoes and classifies them, training on cropped tomatoes makes sense
4. **Improved Robustness**: Handles edge cases better (overlapping tomatoes, different sizes, various lighting)

## Recommended Approach

Since your inference system:
1. Detects multiple tomatoes using color-based detection
2. Crops each tomato individually
3. Classifies each crop separately

**Your training dataset should follow the same pattern:**
- Include cropped individual tomatoes (from both single and multi-tomato images)
- Each crop should be labeled with its correct class
- This matches what the model sees during inference

## Dataset Preparation Options

### Option 1: Automatic Preparation (Recommended)

Use the provided script to automatically crop tomatoes from multi-tomato images:

```bash
python prepare_multi_tomato_dataset.py \
    --source /path/to/your/source/dataset \
    --output /path/to/prepared/dataset \
    --all-splits
```

**What it does:**
- Scans all images in your source dataset
- Detects tomatoes using the same color-based detection as inference
- For single-tomato images: Copies them as-is
- For multi-tomato images: Crops each tomato and saves them individually
- Organizes everything into proper class folders

### Option 2: Manual Preparation

1. **Keep single-tomato images** as they are
2. **For multi-tomato images:**
   - Use an image annotation tool (like LabelImg) to draw bounding boxes
   - Crop each tomato individually
   - Label each crop with the correct class
   - Save to appropriate class folders

### Option 3: Hybrid Approach

- Use single-tomato images directly
- Use the automatic script for multi-tomato images
- Manually review and correct any mis-detected crops

## Dataset Structure

After preparation, your dataset should look like:

```
tomato_dataset/
├── train/
│   ├── not_ready/
│   │   ├── single_tomato_001.jpg
│   │   ├── multi_tomato_001_tomato_1.jpg  (cropped)
│   │   ├── multi_tomato_001_tomato_2.jpg  (cropped)
│   │   └── ...
│   ├── ready/
│   │   ├── single_tomato_002.jpg
│   │   ├── multi_tomato_002_tomato_1.jpg  (cropped)
│   │   └── ...
│   └── spoilt/
│       └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Best Practices

### 1. **Balance Your Dataset**
- Aim for roughly equal numbers of each class
- Include both single and multi-tomato crops
- Mix of sizes, lighting conditions, angles

### 2. **Quality Over Quantity**
- Ensure crops are clear and in-focus
- Remove crops that are too small (< 50x50 pixels)
- Remove crops that are mostly background

### 3. **Review Auto-Cropped Images**
- The automatic script may miss some tomatoes or include false positives
- Manually review and clean up the dataset
- Add missing tomatoes manually if needed

### 4. **Augmentation**
- The training script already includes augmentation (rotation, flip, color jitter)
- This helps the model generalize better
- No need for additional augmentation

### 5. **Validation Split**
- Include multi-tomato images in validation set too
- This ensures the model is tested on realistic scenarios

## Training Command

After preparing your dataset:

```bash
python train_tomato_classifier.py \
    --dataset /path/to/prepared/dataset \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

## Expected Results

Training on cropped individual tomatoes (from both single and multi-tomato images) should:

1. **Improve accuracy** on individual tomato classification
2. **Better handle** real-world scenarios with multiple tomatoes
3. **Match inference behavior** (classifying cropped tomatoes)
4. **Reduce false positives** from background clutter

## Troubleshooting

### Issue: Too many false detections in auto-cropping
**Solution**: Manually review and remove incorrect crops, or adjust detection thresholds in the script

### Issue: Some tomatoes not detected
**Solution**: Manually crop and add missing tomatoes to the dataset

### Issue: Dataset becomes too large
**Solution**: 
- Use a subset of multi-tomato images
- Focus on high-quality crops
- Balance classes appropriately

## Summary

✅ **YES, train on multiple tomatoes!**
- Use the automatic preparation script
- Crop individual tomatoes from multi-tomato images
- Label each crop correctly
- This matches your inference pipeline and improves real-world performance

