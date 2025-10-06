# 🏷️ **What Are Annotations? - Complete Guide**

## 🤔 **What Are Annotations?**

Annotations are **labels** that teach your AI model what objects are in each image. Think of them as "training labels" that tell the computer:

1. **WHERE** each tomato is located (bounding boxes)
2. **WHAT TYPE** each tomato is (class labels)

## 🍅 **Why Do You Need Annotations for Your Tomato Sorter?**

### **The Problem:**
Your AI model needs to learn how to:
- **Find tomatoes** in camera images
- **Classify tomatoes** as ready, not ready, or spoilt
- **Control the robotic arm** to sort them correctly

### **The Solution:**
Annotations provide the "ground truth" data that teaches your AI model to recognize and classify tomatoes.

## 🎯 **How Annotations Work in Your Project:**

### **1. The Learning Process:**
```
Human Annotator → Draws Boxes → Labels Classes → AI Model Learns → Sorts Tomatoes
```

### **2. What You're Teaching the AI:**
- **Object Detection**: "This is a tomato" (drawing boxes)
- **Classification**: "This tomato is ready/not ready/spoilt" (labeling)
- **Pattern Recognition**: Learning visual patterns for each class

### **3. The Result:**
After training on thousands of annotated images, your AI can:
- **Automatically detect** tomatoes in new images
- **Classify them** correctly
- **Control the robotic arm** to sort them

## 🏷️ **Your 3 Classes Explained:**

| Class | ID | Description | Visual Cues | Sorting Action |
|-------|----|--------------|-------------|----------------|
| **not_ready** | 0 | Green, unripe tomatoes | Green color, firm texture | Sort to "not ready" bin |
| **ready** | 1 | Red, ripe tomatoes | Red color, ready to eat | Sort to "ready" bin |
| **spoilt** | 2 | Rotten, damaged tomatoes | Brown/black spots, soft | Sort to "spoilt" bin |

## 🚀 **Your Web-Based Annotation Tool is Ready!**

### **Access Your Tool:**
1. **Open your browser**
2. **Go to**: `http://localhost:5000`
3. **Start annotating** your tomato dataset

### **How to Use the Web Tool:**

#### **Step 1: Load an Image**
- The tool automatically loads the first image
- You'll see progress information at the top

#### **Step 2: Select a Class**
- Click on **0: not_ready**, **1: ready**, or **2: spoilt**
- The selected class will be highlighted

#### **Step 3: Draw Bounding Boxes**
- **Click and drag** to draw a box around each tomato
- **Release** to complete the box
- The box will be colored based on the selected class

#### **Step 4: Save and Continue**
- **Click "Save"** to save your annotations
- **Click "Next"** to move to the next image
- **Repeat** for all images

## 🎮 **Controls & Shortcuts:**

### **Mouse Controls:**
- **Click and drag**: Draw bounding box around tomato
- **Release**: Complete the box

### **Button Controls:**
- **← Previous**: Go to previous image
- **Next →**: Go to next image
- **💾 Save**: Save current annotations
- **🗑️ Clear**: Clear all boxes on current image

### **Class Selection:**
- **0: not_ready**: Green, unripe tomatoes
- **1: ready**: Red, ripe tomatoes
- **2: spoilt**: Rotten, damaged tomatoes

### **Keyboard Shortcuts:**
- **0, 1, 2**: Select class
- **n**: Next image
- **p**: Previous image
- **s**: Save annotations
- **c**: Clear boxes

## 📊 **Progress Tracking:**

### **What You'll See:**
- **Current Image**: Which image you're on
- **Total Images**: How many images total
- **Annotated**: How many you've completed
- **Progress Bar**: Visual progress indicator

### **Your Target:**
- **Training Images**: 1,000-2,000 annotated images
- **Validation Images**: 200-400 annotated images
- **Total**: Aim for 1,200-2,400 annotated images

## 🔧 **Annotation Best Practices:**

### **Quality Guidelines:**
1. **Tight Boxes**: Draw boxes close to the tomato edges
2. **Complete Coverage**: Include all visible tomatoes
3. **Consistent Labeling**: Use the same criteria throughout
4. **Handle Edge Cases**: Partially visible, overlapping tomatoes

### **Efficiency Tips:**
1. **Start with Clear Images**: Easy cases first
2. **Batch Similar Images**: Group by lighting/conditions
3. **Take Breaks**: Avoid annotation fatigue
4. **Review Regularly**: Check your work quality

## 🚀 **After Annotating:**

### **1. Validate Your Dataset:**
```bash
# Check annotation count
find tomato_dataset/labels/train -name "*.txt" | wc -l

# Check progress
python -c "
import os
annotated = len([f for f in os.listdir('tomato_dataset/labels/train') if f.endswith('.txt')])
total = 6500
print(f'Annotated: {annotated}/{total} ({annotated/total*100:.1f}%)')
"
```

### **2. Train Your Model:**
```bash
# Quick training
python train.py --data data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0

# Comprehensive training
python train/train_tomato_detector.py --data data.yaml --epochs 100 --imgsz 640 --batch 16 --plot
```

### **3. Deploy on Raspberry Pi:**
```bash
# Run inference
python inference_pi.py --model best.pt --source 0 --serial /dev/ttyUSB0
```

## 🎯 **The Complete Workflow:**

### **Phase 1: Data Preparation**
1. **Extract dataset** from archive
2. **Organize images** into train/val/test splits
3. **Create data.yaml** configuration

### **Phase 2: Annotation**
1. **Start web tool**: `http://localhost:5000`
2. **Annotate images**: Draw boxes and label classes
3. **Save frequently**: Don't lose your work
4. **Track progress**: Aim for 1,000+ annotations

### **Phase 3: Training**
1. **Validate dataset**: Check annotation quality
2. **Train model**: Use YOLOv8 to learn from annotations
3. **Export model**: Convert to ONNX/TFLite for Raspberry Pi

### **Phase 4: Deployment**
1. **Setup hardware**: Raspberry Pi + Arduino + Camera
2. **Calibrate system**: Camera calibration and IK setup
3. **Run inference**: Real-time tomato sorting

## 🔍 **Understanding the YOLO Format:**

### **What Gets Saved:**
Each annotation is saved as a `.txt` file with lines like:
```
0 0.5 0.5 0.2 0.2    # Class 0, center at (0.5,0.5), size 0.2x0.2
1 0.3 0.3 0.1 0.1    # Class 1, center at (0.3,0.3), size 0.1x0.1
```

### **What Each Number Means:**
- **First number**: Class ID (0=not_ready, 1=ready, 2=spoilt)
- **Next 4 numbers**: Bounding box coordinates (normalized 0-1)

## 🎉 **You're Ready to Start!**

### **Your Web Tool is Running:**
- **URL**: `http://localhost:5000`
- **Dataset**: 6,500 images ready for annotation
- **Classes**: 3 classes defined
- **Format**: YOLO format for YOLOv8 training

### **Start Annotating:**
1. **Open browser** → `http://localhost:5000`
2. **Select class** → Click 0, 1, or 2
3. **Draw boxes** → Click and drag around tomatoes
4. **Save & continue** → Click Save, then Next
5. **Repeat** → Until you have 1,000+ annotations

**Happy Annotating! 🍅🏷️**

Your AI Tomato Sorter will learn from every annotation you create!
