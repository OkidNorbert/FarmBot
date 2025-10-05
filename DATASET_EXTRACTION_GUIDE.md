# 📦 AI Tomato Sorter - Dataset Extraction Guide

## ✅ **YES, I CAN EXTRACT DATASETS FROM ZIP OR TAR!**

The AI Tomato Sorter system now includes a powerful dataset extraction utility that can handle various archive formats and automatically organize your data.

## 🚀 **Quick Usage**

### **Extract ZIP Archive:**
```bash
python extract_dataset.py your_dataset.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup
```

### **Extract TAR Archive:**
```bash
python extract_dataset.py your_dataset.tar.gz \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup
```

## 📋 **Supported Archive Formats**

- ✅ **ZIP**: `.zip`
- ✅ **TAR**: `.tar`
- ✅ **TAR.GZ**: `.tar.gz`, `.tgz`
- ✅ **TAR.BZ2**: `.tar.bz2`, `.tbz2`

## 🔧 **Command Options**

```bash
python extract_dataset.py <archive_path> [options]

Options:
  --extract_to EXTRACT_TO    Directory to extract to (default: extracted_dataset)
  --organize_to ORGANIZE_TO  Directory to organize dataset to (default: tomato_dataset)
  --create_yaml             Create data.yaml file automatically
  --validate                Validate organized dataset structure
  --cleanup                 Clean up extracted files after organization
```

## 🎯 **What the Extraction Utility Does**

### **1. Archive Extraction**
- **Automatically detects** archive format (ZIP, TAR, TAR.GZ, etc.)
- **Extracts** to temporary directory
- **Handles** nested directory structures
- **Preserves** file permissions and timestamps

### **2. Dataset Organization**
- **Creates** proper YOLO dataset structure:
  ```
  tomato_dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
  ```

### **3. Smart File Detection**
- **Finds images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Finds labels**: `.txt` files (YOLO format)
- **Matches** images with corresponding labels
- **Organizes** by existing train/val/test structure

### **4. Automatic Configuration**
- **Creates** `data.yaml` with proper paths
- **Counts** images in each split
- **Sets** class names: `not_ready`, `ready`, `spoilt`
- **Validates** dataset structure

### **5. Validation & Cleanup**
- **Checks** directory structure
- **Verifies** image/label pairs
- **Reports** statistics and issues
- **Cleans up** temporary files (optional)

## 📊 **Example Usage Scenarios**

### **Scenario 1: Complete Dataset Archive**
```bash
# You have a complete dataset in a ZIP file
python extract_dataset.py complete_tomato_dataset.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate \
    --cleanup

# Result: Ready-to-use dataset with data.yaml
```

### **Scenario 2: Raw Images Archive**
```bash
# You have raw images that need organization
python extract_dataset.py raw_images.zip \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate

# Result: Organized structure (you'll need to add labels)
```

### **Scenario 3: Partial Dataset**
```bash
# You have some images and labels mixed together
python extract_dataset.py mixed_data.tar.gz \
    --organize_to tomato_dataset \
    --create_yaml \
    --validate

# Result: Organized structure with matched files
```

## 🔍 **Dataset Structure Requirements**

### **Supported Input Structures:**
```
your_archive.zip
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml (optional)
```

### **Or Mixed Structure:**
```
your_archive.zip
├── train/
│   ├── image1.jpg
│   ├── image1.txt
│   └── ...
├── val/
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
└── test/
    ├── image3.jpg
    ├── image3.txt
    └── ...
```

### **Or Flat Structure:**
```
your_archive.zip
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

## 📈 **Output Structure**

After extraction, you get:
```
tomato_dataset/
├── images/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images
├── labels/
│   ├── train/     # Training labels (.txt)
│   ├── val/       # Validation labels (.txt)
│   └── test/      # Test labels (.txt)
└── data.yaml      # Configuration file (if --create_yaml)
```

## 🎯 **Integration with Training**

### **Quick Path:**
```bash
# Extract dataset
python extract_dataset.py dataset.zip --organize_to tomato_dataset --create_yaml --validate --cleanup

# Train model
python train.py --data data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0
```

### **Comprehensive Path:**
```bash
# Extract dataset
python extract_dataset.py dataset.zip --organize_to tomato_dataset --create_yaml --validate --cleanup

# Advanced training
python train/train_tomato_detector.py --data data.yaml --epochs 100 --imgsz 640 --batch 16 --plot
```

## 🛠️ **Troubleshooting**

### **Common Issues:**

1. **No images found:**
   - Check if your archive contains image files
   - Verify image file extensions (.jpg, .png, etc.)
   - Check if images are in subdirectories

2. **No labels found:**
   - Check if your archive contains .txt files
   - Verify label files are in correct format
   - Check if labels are in subdirectories

3. **Archive format not supported:**
   - Use supported formats: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2
   - Re-compress your archive in supported format

4. **Dataset validation fails:**
   - Check directory structure
   - Verify image/label pairs
   - Ensure proper file permissions

### **Debug Commands:**
```bash
# Check archive contents
unzip -l your_dataset.zip
tar -tzf your_dataset.tar.gz

# Check extracted files
ls -la extracted_dataset/

# Check organized dataset
ls -la tomato_dataset/
```

## 🎉 **Benefits**

- ✅ **One-command setup**: Extract and organize in one step
- ✅ **Format flexibility**: Supports multiple archive formats
- ✅ **Smart organization**: Automatically detects and organizes files
- ✅ **Validation**: Ensures dataset integrity
- ✅ **Integration**: Works seamlessly with training pipeline
- ✅ **Cleanup**: Optional cleanup of temporary files

## 🚀 **Next Steps After Extraction**

1. **Review dataset**: Check organized structure
2. **Validate labels**: Ensure proper YOLO format
3. **Train model**: Use with training pipeline
4. **Deploy system**: Use with inference pipeline

**Your dataset extraction is now fully automated! 🍅📦**
