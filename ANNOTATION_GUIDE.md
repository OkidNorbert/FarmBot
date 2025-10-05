# 🏷️ AI Tomato Sorter - Annotation Guide

## ✅ **Annotation Tools Ready!**

Your annotation setup is complete with multiple options for annotating your tomato dataset.

## 🚀 **Quick Start Annotation**

### **Option 1: Simple Annotation Tool (Recommended)**
```bash
# Activate virtual environment
source tomato_sorter_env/bin/activate

# Start simple annotation tool
python start_annotation.py
```

### **Option 2: Direct Annotation Tool**
```bash
# Activate virtual environment
source tomato_sorter_env/bin/activate

# Start annotation tool directly
python simple_annotator.py --images tomato_dataset/images/train --labels tomato_dataset/labels/train
```

### **2. Configure LabelImg for YOLO Format**
1. **Open LabelImg** (should open automatically)
2. **Go to**: View → Auto Save Mode (check this)
3. **Go to**: View → Single Class Mode (check this for faster annotation)
4. **Go to**: Change Save Dir → Select your labels folder

### **3. Set Up Class Labels**
Before starting annotation, set up your class labels:
1. **Go to**: View → Auto Save Mode
2. **Create classes file**: Create `classes.txt` in your project root with:
   ```
   not_ready
   ready
   spoilt
   ```

## 📁 **Dataset Structure for Annotation**

Your dataset structure:
```
tomato_dataset/
├── images/
│   ├── train/     # 6,500 training images
│   ├── val/       # 724 validation images
│   └── test/      # Ready for test split
├── labels/
│   ├── train/     # Will contain .txt files
│   ├── val/       # Will contain .txt files
│   └── test/      # Will contain .txt files
└── data.yaml      # Configuration file
```

## 🎯 **Annotation Process**

### **Step 1: Annotate Training Images**
```bash
# Start LabelImg
labelImg

# In LabelImg:
# 1. Open Dir: tomato_dataset/images/train
# 2. Change Save Dir: tomato_dataset/labels/train
# 3. Start annotating images
```

### **Step 2: Annotate Validation Images**
```bash
# In LabelImg:
# 1. Open Dir: tomato_dataset/images/val
# 2. Change Save Dir: tomato_dataset/labels/val
# 3. Continue annotating
```

### **Step 3: Create Test Split (Optional)**
```bash
# Move some validation images to test
mkdir -p tomato_dataset/images/test
mkdir -p tomato_dataset/labels/test

# Move some images from val to test
# Then annotate test images
```

## 🏷️ **Annotation Guidelines**

### **Class Definitions:**
- **0: not_ready** - Unripe, green tomatoes
- **1: ready** - Ripe, red tomatoes ready for consumption
- **2: spoilt** - Rotten, damaged, or overripe tomatoes

### **Annotation Rules:**
1. **Draw bounding boxes** around each tomato
2. **Label correctly** based on ripeness/condition
3. **Include partially visible** tomatoes
4. **Avoid overlapping** boxes when possible
5. **Be consistent** with labeling criteria

### **Quality Guidelines:**
- **Accurate boxes**: Tight fit around tomatoes
- **Complete coverage**: Include all visible tomatoes
- **Consistent labeling**: Same criteria throughout
- **Quality over speed**: Better to annotate fewer images well

## ⚡ **Efficient Annotation Workflow**

### **Keyboard Shortcuts:**
- **W**: Create bounding box
- **D**: Next image
- **A**: Previous image
- **Del**: Delete selected box
- **Ctrl+S**: Save current image
- **Space**: Flag current image

### **Batch Processing:**
1. **Start with training set**: Focus on high-quality images
2. **Annotate in batches**: 50-100 images per session
3. **Review periodically**: Check annotation quality
4. **Save frequently**: Use Ctrl+S regularly

## 📊 **Annotation Progress Tracking**

### **Check Progress:**
```bash
# Count annotated images
find tomato_dataset/labels/train -name "*.txt" | wc -l
find tomato_dataset/labels/val -name "*.txt" | wc -l

# Check annotation quality
python -c "
import os
train_labels = len([f for f in os.listdir('tomato_dataset/labels/train') if f.endswith('.txt')])
val_labels = len([f for f in os.listdir('tomato_dataset/labels/val') if f.endswith('.txt')])
print(f'Training annotations: {train_labels}')
print(f'Validation annotations: {val_labels}')
"
```

### **Target Numbers:**
- **Training**: Aim for 1,000-2,000 annotated images
- **Validation**: Aim for 200-400 annotated images
- **Test**: Aim for 100-200 annotated images

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **LabelImg won't start:**
   ```bash
   # Check if virtual environment is activated
   source tomato_sorter_env/bin/activate
   labelImg
   ```

2. **Can't save labels:**
   - Check if labels directory exists
   - Ensure write permissions
   - Check disk space

3. **Labels not in YOLO format:**
   - Go to View → Auto Save Mode
   - Check "PascalVOC" is not selected
   - Should show "YOLO" in status bar

4. **Slow performance:**
   - Close other applications
   - Use smaller image batches
   - Check system resources

## 📈 **Quality Assurance**

### **Validation Steps:**
1. **Spot check**: Review random annotations
2. **Consistency check**: Ensure same labeling criteria
3. **Coverage check**: Verify all tomatoes are labeled
4. **Format check**: Verify .txt files are in YOLO format

### **Sample Validation:**
```bash
# Check YOLO format
head -1 tomato_dataset/labels/train/*.txt | head -5

# Should show format like:
# 0 0.5 0.5 0.2 0.2
# 1 0.3 0.3 0.1 0.1
```

## 🚀 **Next Steps After Annotation**

### **1. Validate Dataset:**
```bash
# Check dataset structure
python train/data_preparation.py \
    --source_images tomato_dataset/images \
    --source_labels tomato_dataset/labels \
    --output tomato_dataset \
    --validate --analyze --visualize
```

### **2. Train Model:**
```bash
# Quick training
python train.py --data data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0

# Comprehensive training
python train/train_tomato_detector.py --data data.yaml --epochs 100 --imgsz 640 --batch 16 --plot
```

## 🎯 **Annotation Tips**

### **Efficiency Tips:**
1. **Start with clear images**: Easy cases first
2. **Use consistent criteria**: Same standards throughout
3. **Batch similar images**: Group by lighting/conditions
4. **Take breaks**: Avoid annotation fatigue
5. **Review regularly**: Check quality periodically

### **Quality Tips:**
1. **Tight bounding boxes**: Close fit around tomatoes
2. **Include all tomatoes**: Don't miss any visible ones
3. **Consistent labeling**: Same criteria for all images
4. **Handle edge cases**: Partially visible, overlapping tomatoes
5. **Document criteria**: Write down labeling rules

## 📚 **Resources**

- **LabelImg Documentation**: https://github.com/heartexlabs/labelImg
- **YOLO Format Guide**: https://roboflow.com/formats/yolo-darknet-txt
- **Annotation Best Practices**: https://roboflow.com/annotate

---

**Happy Annotating! 🍅🏷️**
