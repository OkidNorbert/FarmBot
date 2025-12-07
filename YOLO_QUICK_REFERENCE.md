# YOLO Dataset Quick Reference Cheat Sheet

## 📁 Directory Structure
```
datasets/tomato_yolo/
├── data.yaml              ← Configuration file
├── images/
│   ├── train/            ← 70% of images
│   ├── val/              ← 20% of images
│   └── test/             ← 10% of images
└── labels/
    ├── train/            ← Training labels (.txt)
    ├── val/              ← Validation labels (.txt)
    └── test/             ← Test labels (.txt)
```

## 📝 Label Format
```
class x_center y_center width height
```
- All coordinates: **0.0 to 1.0** (normalized)
- Class: **0-indexed** (0, 1, 2, ...)
- One line per object
- One `.txt` file per image

## 🎯 Example Label File

**Image:** `tomato_001.jpg` (800x600 pixels)  
**Objects:** 2 tomatoes

**tomato_001.txt:**
```
1 0.375 0.500 0.150 0.200
2 0.750 0.300 0.125 0.167
```

Line 1: ready tomato (class 1) at center (0.375, 0.500), size (0.150, 0.200)  
Line 2: spoilt tomato (class 2) at center (0.750, 0.300), size (0.125, 0.167)

## 📄 data.yaml Template

### 3-Class System:
```yaml
path: /home/okidi6/Documents/GitHub/emebeded/datasets/tomato_yolo
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: not_ready
  1: ready
  2: spoilt
```

### 2-Class System (Simplified):
```yaml
path: /home/okidi6/Documents/GitHub/emebeded/datasets/tomato_2class
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: good
  1: bad
```

## 🔢 Coordinate Conversion

### From Pixels to Normalized:
```python
# Given pixel coordinates
x_min, y_min = 100, 150  # Top-left corner
box_width, box_height = 120, 100
img_width, img_height = 640, 480

# Calculate normalized coordinates
x_center = (x_min + box_width/2) / img_width
y_center = (y_min + box_height/2) / img_height
width = box_width / img_width
height = box_height / img_height

# Result: 0.250 0.417 0.188 0.208
```

### From Normalized to Pixels:
```python
# Given normalized coordinates
x_center, y_center = 0.250, 0.417
width, height = 0.188, 0.208
img_width, img_height = 640, 480

# Calculate pixel coordinates
x_min = int((x_center - width/2) * img_width)
y_min = int((y_center - height/2) * img_height)
box_width = int(width * img_width)
box_height = int(height * img_height)

# Result: 100, 150, 120, 100
```

## 🛠️ Quick Setup Commands

```bash
# Create dataset structure
mkdir -p datasets/tomato_yolo/{images,labels}/{train,val,test}

# Install annotation tool
pip install labelImg

# Launch LabelImg
labelImg

# Verify dataset
python -c "
import os
import yaml

with open('datasets/tomato_yolo/data.yaml') as f:
    data = yaml.safe_load(f)
    
for split in ['train', 'val', 'test']:
    img_dir = f\"{data['path']}/images/{split}\"
    lbl_dir = f\"{data['path']}/labels/{split}\"
    if os.path.exists(img_dir):
        imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))])
        lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        print(f'{split}: {imgs} images, {lbls} labels')
"
```

## ✅ Validation Checklist

- [ ] Directory structure matches YOLO format
- [ ] data.yaml exists and is correctly formatted
- [ ] Image and label filenames match (except extension)
- [ ] All coordinates are between 0.0 and 1.0
- [ ] Class IDs start from 0
- [ ] Each image has corresponding .txt file (or no file if empty)
- [ ] No empty .txt files
- [ ] Balanced dataset (similar number of images per class)
- [ ] Train/val/test split is 70/20/10

## 🚀 Training Commands

### Via Command Line:
```bash
yolo detect train data=datasets/tomato_yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### Via Python:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='datasets/tomato_yolo/data.yaml', epochs=100, imgsz=640)
```

### Via Web Interface:
1. Go to http://localhost:5000/training
2. Click "Start Training"
3. Select YOLO model
4. Configure parameters
5. Start training

## 🎯 Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model Size | yolov8n.pt | Nano - fastest, good for testing |
| Epochs | 50-100 | More epochs = better accuracy |
| Image Size | 640 | Standard YOLO input size |
| Batch Size | 16 | Adjust based on GPU memory |
| Confidence | 0.5 | Threshold for detections |

## 📊 Expected Results

| Metric | Good | Excellent |
|--------|------|-----------|
| mAP50 | >0.85 | >0.95 |
| Precision | >0.80 | >0.90 |
| Recall | >0.75 | >0.85 |

## 🔗 Quick Links

- **Full Guide:** `YOLO_DATASET_GUIDE.md`
- **Training Guide:** `FRESH_TRAINING_GUIDE.md`
- **Ultralytics Docs:** https://docs.ultralytics.com/datasets/detect/
- **LabelImg:** https://github.com/HumanSignal/labelImg
- **Roboflow:** https://roboflow.com/
