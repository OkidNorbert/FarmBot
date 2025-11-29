# 🍅 Ugandan Tomato Model Fine-Tuning Guide

## 📸 Photo Requirements

### **Minimum Requirements (Quick Fine-Tuning)**
For fine-tuning the existing model with Ugandan tomatoes, you need:

- **Total: 150-300 photos minimum**
  - **Not Ready (Green/Unripe)**: 50-100 photos
  - **Ready (Ripe)**: 60-120 photos  
  - **Spoilt (Overripe/Damaged)**: 40-80 photos

### **Recommended (Better Results)**
For better accuracy with Ugandan tomatoes:

- **Total: 300-600 photos**
  - **Not Ready**: 100-200 photos
  - **Ready**: 120-240 photos
  - **Spoilt**: 80-160 photos

### **Optimal (Production Quality)**
For best results matching the original training:

- **Total: 600-1000 photos**
  - **Not Ready**: 200-350 photos
  - **Ready**: 250-400 photos
  - **Spoilt**: 150-250 photos

## 📷 Photo Collection Guidelines

### **What to Capture:**

1. **Variety of Ugandan Tomato Types**
   - Different sizes (small, medium, large)
   - Different shapes (round, oval, slightly irregular)
   - Different colors (various shades of green, red, orange)

2. **Lighting Conditions**
   - Natural sunlight (morning, noon, evening)
   - Shade/indoor lighting
   - Different angles (top-down, side view, 45° angle)

3. **Backgrounds**
   - Plain backgrounds (white, black, wood)
   - Natural backgrounds (leaves, soil, crates)
   - Conveyor belt or sorting area

4. **Conditions**
   - Single tomatoes
   - Clusters (2-3 tomatoes together)
   - Partially visible tomatoes
   - Different ripeness stages

5. **Quality**
   - Clear, in-focus images
   - Good resolution (at least 640x640 pixels)
   - JPG or PNG format

## 📁 Dataset Structure

Create this folder structure:

```
ugandan_tomato_dataset/
├── train/
│   ├── not_ready/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── ready/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── spoilt/
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── val/
    ├── not_ready/
    ├── ready/
    └── spoilt/
```

**Split Ratio:**
- **80% for training** (train folder)
- **20% for validation** (val folder)

## 🎯 Quick Start (Minimum Dataset)

If you're in a hurry, you can start with:

- **50 photos per class** (150 total)
- Focus on **ready** class (most important for picking)
- Use data augmentation to multiply your dataset

## 🔄 Data Augmentation

The training script automatically augments your images:
- Random flips
- Rotation (±10°)
- Brightness/contrast adjustments
- Color jitter

This means **50 photos can become 200+ training samples**!

## 📊 Current Model Stats

Your current model was trained with:
- **328 training samples**
- **41 validation samples**
- **99.85% accuracy**

For Ugandan tomatoes, you can achieve similar results with:
- **150-300 photos** (fine-tuning)
- **300-600 photos** (better accuracy)
- **600-1000 photos** (production quality)

## 🚀 Training Command

Once you have your photos organized:

```bash
# Activate environment
source farmbot_env/bin/activate

# Train with your Ugandan tomato dataset
python train_tomato_classifier.py \
    --dataset ugandan_tomato_dataset \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.0001
```

## 💡 Tips for Better Results

1. **Start Small**: Begin with 150 photos, test, then add more if needed
2. **Focus on Ready Class**: Most important for automatic picking
3. **Capture Real Conditions**: Use the same camera/lighting as production
4. **Balance Classes**: Try to have similar numbers in each class
5. **Quality over Quantity**: 100 good photos > 500 blurry photos

## 📈 Expected Results

- **150 photos**: 70-80% accuracy (good for testing)
- **300 photos**: 85-90% accuracy (good for production)
- **600+ photos**: 90-95% accuracy (excellent for production)

## 🎬 Next Steps

1. Collect photos using your camera/phone
2. Organize into the folder structure above
3. Run the training script
4. Test the new model
5. Deploy to your system

---

**Remember**: You can always start with fewer photos and add more later if needed!

