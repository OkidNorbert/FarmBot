# YOLO Dataset Preparation Guide for Tomato Detection
## Complete Guide Based on Ultralytics Documentation

**Date:** December 6, 2025  
**Reference:** https://docs.ultralytics.com/datasets/detect/

---

## 📋 Table of Contents
1. [YOLO Dataset Format Overview](#yolo-dataset-format-overview)
2. [Directory Structure](#directory-structure)
3. [Label Format Explained](#label-format-explained)
4. [Creating Your Tomato Dataset](#creating-your-tomato-dataset)
5. [Annotation Tools](#annotation-tools)
6. [Step-by-Step Tutorial](#step-by-step-tutorial)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## 🎯 YOLO Dataset Format Overview

The Ultralytics YOLO format requires:
1. **Images** organized in train/val/test folders
2. **Labels** in `.txt` files (one per image) with bounding box annotations
3. **data.yaml** configuration file defining paths and class names

### Key Requirements:
- ✅ One `.txt` label file per image (same filename, different extension)
- ✅ Bounding boxes in **normalized coordinates** (0 to 1)
- ✅ Format: `class x_center y_center width height`
- ✅ Class IDs are **zero-indexed** (start from 0)
- ✅ If an image has no objects, no `.txt` file is needed

---

## 📁 Directory Structure

### Recommended Structure for Tomato Dataset:

```
datasets/
└── tomato_yolo/
    ├── data.yaml              # Dataset configuration file
    ├── images/
    │   ├── train/             # Training images
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   ├── val/               # Validation images
    │   │   ├── img101.jpg
    │   │   ├── img102.jpg
    │   │   └── ...
    │   └── test/              # Test images (optional)
    │       ├── img201.jpg
    │       └── ...
    └── labels/
        ├── train/             # Training labels
        │   ├── img001.txt
        │   ├── img002.txt
        │   └── ...
        ├── val/               # Validation labels
        │   ├── img101.txt
        │   ├── img102.txt
        │   └── ...
        └── test/              # Test labels (optional)
            ├── img201.txt
            └── ...
```

**Important:** 
- Image and label filenames must match (except extension)
- `img001.jpg` → `img001.txt`
- Labels go in `labels/` folder, images in `images/` folder

---

## 📝 Label Format Explained

### Format: `class x_center y_center width height`

Each line in a `.txt` file represents one object (tomato) in the image.

### Coordinate System:
- **Normalized coordinates**: All values are between 0 and 1
- **x_center**: Horizontal center of bounding box (0 = left edge, 1 = right edge)
- **y_center**: Vertical center of bounding box (0 = top edge, 1 = bottom edge)
- **width**: Width of bounding box relative to image width
- **height**: Height of bounding box relative to image height

### Example:

**Image:** `tomato_001.jpg` (640x480 pixels)

**Visual:**
```
┌─────────────────────────────────┐
│                                 │
│     ┌──────────┐                │  ← Image (640x480)
│     │  Tomato  │                │
│     │  (ripe)  │                │
│     └──────────┘                │
│                                 │
└─────────────────────────────────┘
```

**Pixel coordinates:**
- Bounding box: x=100, y=150, width=120, height=100

**Normalized coordinates:**
- x_center = (100 + 120/2) / 640 = 0.250
- y_center = (150 + 100/2) / 480 = 0.417
- width = 120 / 640 = 0.188
- height = 100 / 480 = 0.208

**Label file** (`tomato_001.txt`):
```
1 0.250 0.417 0.188 0.208
```
(Class 1 = "ready" tomato)

### Multiple Objects Example:

**Image with 3 tomatoes:**

**Label file** (`tomato_002.txt`):
```
0 0.150 0.300 0.120 0.150
1 0.500 0.450 0.180 0.200
2 0.800 0.600 0.140 0.160
```
- Line 1: not_ready tomato (class 0)
- Line 2: ready tomato (class 1)
- Line 3: spoilt tomato (class 2)

---

## 🍅 Creating Your Tomato Dataset

### Option 1: Three Classes (Production System)

**data.yaml:**
```yaml
# Tomato Detection Dataset - 3 Classes
# Path to dataset root (can be absolute or relative)
path: /home/okidi6/Documents/GitHub/emebeded/datasets/tomato_yolo

# Paths to train/val/test images (relative to 'path')
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 3

# Class names (must be in order, starting from 0)
names:
  0: not_ready  # Unripe/green tomatoes
  1: ready      # Ripe/red tomatoes ready for harvest
  2: spoilt     # Overripe/damaged tomatoes
```

### Option 2: Two Classes (Recommended for Testing)

**data.yaml:**
```yaml
# Tomato Detection Dataset - 2 Classes (Simplified)
path: /home/okidi6/Documents/GitHub/emebeded/datasets/tomato_2class

train: images/train
val: images/val
test: images/test  # optional

nc: 2

names:
  0: good  # Good quality tomatoes (ready for sale)
  1: bad   # Bad quality tomatoes (reject/discard)
```

---

## 🛠️ Annotation Tools

### Recommended Tools for Creating YOLO Labels:

### 1. **LabelImg** (Desktop, Free) ⭐ RECOMMENDED
- **Download:** https://github.com/HumanSignal/labelImg
- **Format:** Supports YOLO format directly
- **Pros:** Easy to use, keyboard shortcuts, works offline
- **Cons:** Manual installation required

**Installation:**
```bash
pip install labelImg
labelImg
```

**Usage:**
1. Open directory with images
2. Draw bounding boxes around tomatoes
3. Assign class labels
4. Save in YOLO format
5. Labels automatically saved as `.txt` files

### 2. **Roboflow** (Online, Free Tier) ⭐ EASIEST
- **Website:** https://roboflow.com/
- **Format:** Exports to YOLO format
- **Pros:** Web-based, auto-annotation, augmentation tools
- **Cons:** Requires internet, limited free tier

**Workflow:**
1. Upload images
2. Draw bounding boxes
3. Label classes
4. Export in "YOLO v5 PyTorch" format
5. Download and extract to your dataset folder

### 3. **CVAT** (Online/Self-hosted, Free)
- **Website:** https://cvat.ai/
- **Format:** Exports to YOLO format
- **Pros:** Professional tool, team collaboration
- **Cons:** More complex, steeper learning curve

### 4. **Makesense.ai** (Online, Free)
- **Website:** https://www.makesense.ai/
- **Format:** Exports to YOLO format
- **Pros:** Simple, no registration needed
- **Cons:** Basic features only

---

## 📖 Step-by-Step Tutorial

### Step 1: Collect Images

1. **Take photos** of tomatoes in different conditions:
   - Various lighting (natural, artificial)
   - Different angles
   - Multiple tomatoes per image
   - Single tomatoes
   - Different backgrounds

2. **Organize images** by split:
   - **70% for training** (e.g., 700 images)
   - **20% for validation** (e.g., 200 images)
   - **10% for testing** (e.g., 100 images)

3. **Naming convention:**
   - Use consistent naming: `tomato_001.jpg`, `tomato_002.jpg`, etc.
   - Avoid spaces or special characters
   - Use lowercase

### Step 2: Create Directory Structure

```bash
cd /home/okidi6/Documents/GitHub/emebeded

# Create dataset directories
mkdir -p datasets/tomato_yolo/images/{train,val,test}
mkdir -p datasets/tomato_yolo/labels/{train,val,test}
```

### Step 3: Move Images to Folders

```bash
# Move training images
mv /path/to/your/images/train/*.jpg datasets/tomato_yolo/images/train/

# Move validation images
mv /path/to/your/images/val/*.jpg datasets/tomato_yolo/images/val/

# Move test images (optional)
mv /path/to/your/images/test/*.jpg datasets/tomato_yolo/images/test/
```

### Step 4: Annotate Images

**Using LabelImg:**

1. **Install and launch:**
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Configure:**
   - Click "Open Dir" → Select `datasets/tomato_yolo/images/train/`
   - Click "Change Save Dir" → Select `datasets/tomato_yolo/labels/train/`
   - Click "PascalVOC" button → Change to "YOLO"

3. **Annotate:**
   - Press `W` to create bounding box
   - Draw box around tomato
   - Select class (not_ready, ready, or spoilt)
   - Press `Ctrl+S` to save
   - Press `D` to go to next image

4. **Repeat** for validation and test sets

**Using Roboflow:**

1. **Create project** at https://roboflow.com/
2. **Upload images** (drag and drop)
3. **Annotate:**
   - Click "Start Annotating"
   - Draw bounding boxes
   - Assign classes
4. **Generate dataset:**
   - Click "Generate"
   - Select "YOLO v5 PyTorch" format
   - Download ZIP file
5. **Extract** to `datasets/tomato_yolo/`

### Step 5: Create data.yaml

Create `datasets/tomato_yolo/data.yaml`:

```yaml
# Tomato Detection Dataset
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

### Step 6: Verify Dataset

**Check structure:**
```bash
cd datasets/tomato_yolo
tree -L 3
```

**Expected output:**
```
.
├── data.yaml
├── images
│   ├── train
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val
│   │   └── ...
│   └── test
│       └── ...
└── labels
    ├── train
    │   ├── img001.txt
    │   └── ...
    ├── val
    │   └── ...
    └── test
        └── ...
```

**Verify label files:**
```bash
# Check if each image has a corresponding label
cd datasets/tomato_yolo
for img in images/train/*.jpg; do
    label="labels/train/$(basename ${img%.jpg}.txt)"
    if [ ! -f "$label" ]; then
        echo "Missing label: $label"
    fi
done
```

### Step 7: Train Model

**Via Web Interface:**
1. Navigate to http://localhost:5000/training
2. Dataset should appear in the list
3. Click "Start Training"
4. Select "YOLO" model type
5. Configure epochs (50-100)
6. Start training

**Via Command Line:**
```bash
cd /home/okidi6/Documents/GitHub/emebeded
source farmbot_env/bin/activate

# Train YOLO model
yolo detect train data=datasets/tomato_yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

**Via Python Script:**
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='datasets/tomato_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='tomato_detector'
)
```

---

## ⚠️ Common Mistakes to Avoid

### 1. **Incorrect Coordinate Format**
❌ **Wrong:** Using pixel coordinates
```
1 100 150 120 100  # WRONG - these are pixels!
```

✅ **Correct:** Using normalized coordinates (0-1)
```
1 0.250 0.417 0.188 0.208  # CORRECT - normalized
```

### 2. **Mismatched Filenames**
❌ **Wrong:**
```
images/train/tomato_001.jpg
labels/train/tomato_1.txt     # Filename doesn't match!
```

✅ **Correct:**
```
images/train/tomato_001.jpg
labels/train/tomato_001.txt   # Filenames match
```

### 3. **Wrong Class IDs**
❌ **Wrong:** Starting class IDs from 1
```yaml
names:
  1: not_ready  # WRONG - should start from 0
  2: ready
  3: spoilt
```

✅ **Correct:** Starting from 0
```yaml
names:
  0: not_ready  # CORRECT - zero-indexed
  1: ready
  2: spoilt
```

### 4. **Absolute Paths in data.yaml**
❌ **Wrong:** Using absolute paths for train/val
```yaml
path: /home/okidi6/datasets/tomato_yolo
train: /home/okidi6/datasets/tomato_yolo/images/train  # WRONG
val: /home/okidi6/datasets/tomato_yolo/images/val      # WRONG
```

✅ **Correct:** Using relative paths
```yaml
path: /home/okidi6/datasets/tomato_yolo
train: images/train  # CORRECT - relative to 'path'
val: images/val      # CORRECT - relative to 'path'
```

### 5. **Empty Label Files**
❌ **Wrong:** Creating empty `.txt` files for images with no objects
```
# img_no_tomato.txt (empty file)

```

✅ **Correct:** No `.txt` file if image has no objects
```
# If img_no_tomato.jpg has no tomatoes, don't create img_no_tomato.txt
```

### 6. **Coordinates Out of Range**
❌ **Wrong:** Coordinates > 1.0 or < 0.0
```
1 1.5 0.5 0.2 0.3  # WRONG - x_center > 1.0
```

✅ **Correct:** All coordinates between 0 and 1
```
1 0.75 0.5 0.2 0.3  # CORRECT - all values 0-1
```

---

## 🎓 Best Practices

### 1. **Data Quality**
- ✅ Use high-resolution images (at least 640x640)
- ✅ Ensure good lighting and focus
- ✅ Include variety (angles, backgrounds, lighting)
- ✅ Remove blurry or low-quality images

### 2. **Balanced Dataset**
- ✅ Equal number of images per class
- ✅ Variety of object sizes (small, medium, large tomatoes)
- ✅ Multiple objects per image when possible

### 3. **Annotation Accuracy**
- ✅ Tight bounding boxes (minimal background)
- ✅ Include entire object (don't cut off parts)
- ✅ Consistent labeling criteria
- ✅ Double-check annotations before training

### 4. **Dataset Splits**
- ✅ 70% training, 20% validation, 10% test
- ✅ Ensure no overlap between splits
- ✅ Random distribution across splits

---

## 🧪 Testing Your Dataset

### Quick Validation Script:

```python
import os
import yaml

# Load data.yaml
with open('datasets/tomato_yolo/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(f"Dataset: {data['path']}")
print(f"Classes: {data['nc']}")
print(f"Names: {data['names']}")

# Count images and labels
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(data['path'], 'images', split)
    lbl_dir = os.path.join(data['path'], 'labels', split)
    
    if os.path.exists(img_dir):
        num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        num_labels = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
        print(f"  Match: {'✅' if num_images >= num_labels else '❌'}")
```

---

## 📚 Additional Resources

- **Ultralytics Docs:** https://docs.ultralytics.com/datasets/detect/
- **YOLO Format Spec:** https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
- **LabelImg Tutorial:** https://github.com/HumanSignal/labelImg#usage
- **Roboflow Guide:** https://docs.roboflow.com/
- **YOLO Training Tips:** https://docs.ultralytics.com/modes/train/

---

## 🎉 You're Ready!

Follow this guide to create a properly formatted YOLO dataset for your tomato detection system. Take your time with annotation - quality over quantity!

**Good luck with your dataset preparation! 🍅🤖**
