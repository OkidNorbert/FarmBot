# 🍅 **Tomato Dataset Annotation Solution**

## ✅ **Problem Solved!**

Since LabelImg has compatibility issues with Python 3.13, I've created a **custom annotation tool** that works perfectly with your setup.

## 🚀 **Quick Start (3 Steps)**

### **Step 1: Start Annotation**
```bash
# Activate environment
source tomato_sorter_env/bin/activate

# Start annotation tool
python start_annotation.py
```

### **Step 2: Annotate Images**
- **Mouse**: Draw bounding boxes around tomatoes
- **Keys 0,1,2**: Select class (not_ready, ready, spoilt)
- **Key n**: Next image
- **Key s**: Save annotations
- **Key q**: Quit

### **Step 3: Check Progress**
```bash
# Count annotated images
find tomato_dataset/labels/train -name "*.txt" | wc -l
```

## 🎯 **Annotation Workflow**

### **Your Dataset:**
- **Total Images**: 6,500 training images
- **Classes**: 3 (not_ready, ready, spoilt)
- **Format**: YOLO format (.txt files)

### **Annotation Process:**
1. **Start with training images** (tomato_dataset/images/train)
2. **Draw boxes** around each tomato
3. **Label correctly** based on ripeness
4. **Save frequently** (press 's')
5. **Move to next image** (press 'n')

## 🏷️ **Class Definitions**

| Class | ID | Description | Visual Cues |
|-------|----|--------------|-------------|
| **not_ready** | 0 | Green, unripe tomatoes | Green color, firm texture |
| **ready** | 1 | Red, ripe tomatoes | Red color, ready to eat |
| **spoilt** | 2 | Rotten, damaged tomatoes | Brown/black spots, soft texture |

## 📊 **Annotation Targets**

### **Recommended Numbers:**
- **Training**: 1,000-2,000 annotated images
- **Validation**: 200-400 annotated images
- **Test**: 100-200 annotated images

### **Quality Guidelines:**
- **Tight bounding boxes** around each tomato
- **Include all visible tomatoes** in the image
- **Consistent labeling** throughout the dataset
- **Handle edge cases** (partially visible, overlapping)

## 🔧 **Annotation Tool Features**

### **Controls:**
- **Mouse**: Draw bounding boxes
- **0,1,2**: Select class (not_ready, ready, spoilt)
- **n**: Next image
- **p**: Previous image
- **s**: Save annotations
- **d**: Delete last box
- **c**: Clear all boxes
- **q**: Quit

### **Features:**
- **Auto-save**: Annotations saved automatically
- **YOLO format**: Compatible with YOLOv8
- **Visual feedback**: Color-coded boxes
- **Progress tracking**: Shows current image number
- **Load existing**: Continues from where you left off

## 📈 **Progress Tracking**

### **Check Your Progress:**
```bash
# Count training annotations
find tomato_dataset/labels/train -name "*.txt" | wc -l

# Count validation annotations
find tomato_dataset/labels/val -name "*.txt" | wc -l

# Check annotation quality
python -c "
import os
train_count = len([f for f in os.listdir('tomato_dataset/labels/train') if f.endswith('.txt')])
val_count = len([f for f in os.listdir('tomato_dataset/labels/val') if f.endswith('.txt')])
print(f'Training annotations: {train_count}')
print(f'Validation annotations: {val_count}')
print(f'Progress: {train_count/6500*100:.1f}% of training images')
"
```

## 🚀 **After Annotation**

### **1. Validate Dataset:**
```bash
# Check dataset structure
python -c "
import os
from pathlib import Path

# Check if all required files exist
required_dirs = ['tomato_dataset/images/train', 'tomato_dataset/images/val', 
                 'tomato_dataset/labels/train', 'tomato_dataset/labels/val']

for dir_path in required_dirs:
    if Path(dir_path).exists():
        file_count = len(list(Path(dir_path).glob('*')))
        print(f'✅ {dir_path}: {file_count} files')
    else:
        print(f'❌ {dir_path}: Missing')
"
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

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **Tool won't start:**
   ```bash
   # Check virtual environment
   source tomato_sorter_env/bin/activate
   python --version
   ```

2. **Can't save annotations:**
   - Check if labels directory exists
   - Ensure write permissions
   - Check disk space

3. **Images not loading:**
   - Check image format (JPG, PNG)
   - Verify image paths
   - Check file permissions

4. **Slow performance:**
   - Close other applications
   - Use smaller image batches
   - Check system resources

## 📚 **Resources**

- **YOLO Format Guide**: https://roboflow.com/formats/yolo-darknet-txt
- **Annotation Best Practices**: https://roboflow.com/annotate
- **Dataset Validation**: Use the provided validation scripts

---

**Ready to start annotating! 🍅🏷️**

Your custom annotation tool is ready to use. Start with `python start_annotation.py` and begin annotating your tomato dataset!
