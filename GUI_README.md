# 🍅 **AI Tomato Sorter - GUI Application**

## 🎯 **Easy-to-Use Graphical Interface**

This GUI application makes it incredibly easy to train, test, and deploy your tomato classification models without any command-line knowledge!

## 🚀 **Quick Start**

### **1. Launch the GUI:**
```bash
# Activate virtual environment
source tomato_sorter_env/bin/activate

# Launch GUI
python launch_gui.py
```

### **2. Or run directly:**
```bash
python tomato_gui.py
```

## 📱 **GUI Features**

### **📁 Dataset Tab**
- **Browse Dataset**: Select your tomato dataset directory
- **Analyze Dataset**: View dataset structure and class distribution
- **Verify Structure**: Ensure dataset is properly organized
- **Class Mapping**: See how classes are mapped (Unripe→not_ready, Ripe→ready, Old+Damaged→spoilt)

### **🚀 Training Tab**
- **Training Parameters**: 
  - Epochs (10-200)
  - Batch Size (8-128)
  - Learning Rate (0.0001-0.01)
  - Device Selection (Auto/CPU/CUDA)
- **Training Controls**: Start/Stop training with one click
- **Real-time Progress**: Live training logs and progress
- **Training Curves**: Visual progress tracking

### **🔍 Inference Tab**
- **Model Selection**: Choose your trained model
- **Test Options**: Camera or single image testing
- **Results Display**: Classification results with confidence scores
- **Real-time Testing**: Live camera inference

### **🚀 Deployment Tab**
- **Model Export**: Convert to ONNX/TFLite formats
- **Raspberry Pi Deployment**: Deploy to edge devices
- **Web Interface**: Start web server for remote access
- **Connection Testing**: Verify Pi connectivity

### **❓ Help Tab**
- **Complete Guide**: Step-by-step instructions
- **Troubleshooting**: Common issues and solutions
- **Quick Start**: Get up and running fast

## 🎮 **How to Use**

### **Step 1: Setup Dataset**
1. Open the **Dataset** tab
2. Click **Browse** and select your `tomato_dataset` folder
3. Click **Analyze Dataset** to verify structure
4. Check that all classes are detected correctly

### **Step 2: Train Model**
1. Go to the **Training** tab
2. Adjust parameters if needed (defaults are usually good)
3. Click **Start Training**
4. Monitor progress in the log window
5. Wait for training to complete

### **Step 3: Test Model**
1. Switch to the **Inference** tab
2. Select your trained model file
3. Choose **Camera** or **Image File** testing
4. Click **Start Inference**
5. View results and confidence scores

### **Step 4: Deploy Model**
1. Go to the **Deployment** tab
2. Export model to desired format
3. Deploy to Raspberry Pi or start web server
4. Test deployment

## 🔧 **Requirements**

### **System Requirements:**
- Python 3.8+
- Virtual environment activated
- Required packages installed

### **Install Dependencies:**
```bash
# Activate environment
source tomato_sorter_env/bin/activate

# Install GUI dependencies
pip install opencv-python pillow numpy

# Install training dependencies (if not already installed)
pip install torch torchvision matplotlib scikit-learn
```

## 📊 **Dataset Structure Expected**

```
tomato_dataset/
├── train/
│   ├── Unripe/     # → not_ready (Class 0)
│   ├── Ripe/       # → ready (Class 1)
│   ├── Old/        # → spoilt (Class 2)
│   └── Damaged/    # → spoilt (Class 2)
├── val/
│   ├── Unripe/     # → not_ready (Class 0)
│   ├── Ripe/       # → ready (Class 1)
│   ├── Old/        # → spoilt (Class 2)
│   └── Damaged/    # → spoilt (Class 2)
└── data.yaml       # Auto-generated
```

## 🎯 **Training Workflow**

### **1. Dataset Analysis:**
- Verify 6,487 total images
- Check class distribution
- Ensure proper folder structure

### **2. Training Configuration:**
- **Epochs**: 50 (recommended for quick training)
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 0.001 (good starting point)
- **Device**: Auto (will use GPU if available)

### **3. Training Process:**
- Real-time progress monitoring
- Automatic model saving
- Training curve visualization
- Error handling and recovery

## 🔍 **Inference Options**

### **Camera Testing:**
- Real-time classification
- Live confidence scores
- Visual feedback
- Performance monitoring

### **Image Testing:**
- Single image classification
- Batch processing
- Result comparison
- Model validation

## 🚀 **Deployment Options**

### **Model Export:**
- **ONNX**: For cross-platform deployment
- **TFLite**: For mobile/edge devices
- **PyTorch**: For Python applications

### **Raspberry Pi Deployment:**
- Automated deployment scripts
- Hardware compatibility checking
- Performance optimization
- Remote monitoring

### **Web Interface:**
- Browser-based access
- Remote model testing
- API endpoints
- Real-time updates

## 🛠️ **Troubleshooting**

### **Common Issues:**

1. **GUI Won't Start:**
   ```bash
   # Check dependencies
   python launch_gui.py
   
   # Install missing packages
   pip install opencv-python pillow numpy
   ```

2. **Training Fails:**
   - Check dataset path is correct
   - Verify PyTorch is installed
   - Ensure sufficient disk space
   - Check GPU availability

3. **Inference Errors:**
   - Verify model file exists
   - Check image format compatibility
   - Ensure camera is accessible

4. **Deployment Issues:**
   - Verify Pi connection
   - Check model format compatibility
   - Ensure proper permissions

### **Performance Tips:**

1. **Faster Training:**
   - Use GPU if available
   - Increase batch size
   - Reduce image resolution

2. **Better Accuracy:**
   - Increase epochs
   - Adjust learning rate
   - Use data augmentation

3. **Faster Inference:**
   - Use ONNX format
   - Reduce image size
   - Optimize model

## 📚 **File Structure**

```
project/
├── tomato_gui.py          # Main GUI application
├── launch_gui.py         # GUI launcher
├── train_tomato_classifier.py  # Training script
├── inference_classifier.py    # Inference script
├── quick_train.py        # Dataset analysis
├── tomato_dataset/       # Your dataset
└── GUI_README.md         # This guide
```

## 🎉 **Benefits of Using the GUI**

### **✅ User-Friendly:**
- No command-line knowledge required
- Visual interface for all operations
- Real-time feedback and progress

### **✅ Comprehensive:**
- Complete workflow from dataset to deployment
- All features in one application
- Integrated help and troubleshooting

### **✅ Professional:**
- Clean, modern interface
- Error handling and validation
- Progress tracking and logging

### **✅ Efficient:**
- One-click operations
- Automated processes
- Batch operations support

## 🚀 **Ready to Use!**

Your GUI application is ready to make tomato classification training and deployment incredibly easy!

**Launch it now:**
```bash
python launch_gui.py
```

**Start with the Dataset tab to analyze your data, then move to Training to train your model, and finally use Inference to test it! 🍅🤖**
