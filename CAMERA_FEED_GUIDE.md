# 📹 Live Camera Feed Guide

## 🎯 **New Feature: Live Camera Monitoring**

### **What You Can Now Do:**
- **See live camera feed** in your web interface
- **Monitor tomatoes** in real-time
- **Capture images** from the live feed
- **Classify tomatoes** automatically
- **Control camera settings** from the web interface

## 🚀 **How to Access Camera Feed**

### **Step 1: Open Web Interface**
1. Go to **http://localhost:5001**
2. Click **"Live Camera"** in the navigation menu
3. The camera feed will start automatically

### **Step 2: Camera Controls**
- **Start/Stop Camera** - Control the live feed
- **Capture Image** - Take a snapshot
- **Classify Current** - Run AI classification on current frame
- **Auto Capture** - Automatically capture and classify at intervals

## 📹 **Camera Feed Features**

### **Live Video Stream:**
- **Real-time camera feed** from your default camera (usually /dev/video0)
- **High-quality video** with automatic scaling
- **Responsive design** that works on desktop and mobile

### **Camera Controls:**
- **Start/Stop buttons** for camera control
- **Capture button** to take snapshots
- **Classification button** to run AI on current frame
- **Auto-capture settings** for continuous monitoring

### **Live Classification:**
- **Real-time AI classification** of tomatoes
- **Confidence scores** for each prediction
- **Color-coded results** (green=ready, yellow=not_ready, red=spoilt)
- **Automatic capture** and classification at set intervals

## 🔧 **Camera Settings**

### **Detection Overlay:**
- **Show Detection Overlay** - Display bounding boxes and labels
- **Auto Capture on Detection** - Automatically capture when tomatoes are detected
- **Capture Interval** - Set how often to capture (1-60 seconds)

### **Classification Settings:**
- **Live Classification** - Run AI on every captured frame
- **Confidence Threshold** - Set minimum confidence for predictions
- **Class Filtering** - Focus on specific tomato classes

## 🎯 **Production Use Cases**

### **1. Real-Time Monitoring:**
- **Monitor conveyor belt** for incoming tomatoes
- **Check tomato quality** before sorting
- **Verify robotic arm** positioning and movement
- **Debug system issues** with live visual feedback

### **2. Quality Control:**
- **Inspect tomatoes** before they enter the system
- **Verify classification accuracy** in real-time
- **Monitor sorting performance** and accuracy
- **Adjust system parameters** based on live feedback

### **3. System Integration:**
- **Coordinate with robotic arm** for precise positioning
- **Trigger sorting actions** based on live detection
- **Monitor system health** and performance
- **Record sorting operations** for analysis

## 🔧 **Technical Details**

### **Camera Requirements:**
- **USB camera** or **Raspberry Pi camera module**
- **Minimum resolution**: 640x480
- **Recommended**: 1280x720 or higher
- **Frame rate**: 15-30 FPS for smooth operation

### **Browser Compatibility:**
- **Chrome/Chromium** (recommended)
- **Firefox** (good support)
- **Safari** (limited support)
- **Mobile browsers** (responsive design)

### **Network Requirements:**
- **Local network** access (same as web interface)
- **Sufficient bandwidth** for video streaming
- **Low latency** for real-time operation

## 🚀 **Usage Instructions**

### **Basic Camera Monitoring:**
1. **Open web interface** → Click "Live Camera"
2. **Camera starts automatically** (if available)
3. **View live feed** of your camera
4. **Use controls** to capture or classify images

### **Automatic Classification:**
1. **Enable "Auto Capture"** checkbox
2. **Set capture interval** (e.g., 5 seconds)
3. **System automatically** captures and classifies
4. **View results** in the classification panel

### **Manual Classification:**
1. **Click "Capture Image"** to take a snapshot
2. **Click "Classify Current"** to run AI on current frame
3. **View results** with confidence scores
4. **Save results** for analysis

## 🔍 **Troubleshooting**

### **Camera Not Working:**
- **Check camera connection** (USB or camera module)
- **Verify camera permissions** in browser
- **Try different camera** (change camera index)
- **Check camera drivers** and installation

### **Poor Video Quality:**
- **Adjust camera resolution** in system settings
- **Check lighting conditions** in your setup
- **Verify camera focus** and positioning
- **Test with different browsers**

### **Classification Issues:**
- **Ensure good lighting** for clear tomato visibility
- **Position camera** for optimal tomato viewing
- **Check model accuracy** with test images
- **Adjust confidence thresholds** if needed

## 🎉 **Benefits for Production**

### **Real-Time Monitoring:**
- **See exactly what the camera sees** during operation
- **Monitor tomato quality** and positioning
- **Debug issues** with visual feedback
- **Verify system performance** in real-time

### **Quality Assurance:**
- **Inspect tomatoes** before sorting
- **Verify classification accuracy** with live feedback
- **Monitor sorting performance** and adjust as needed
- **Record operations** for analysis and improvement

### **System Integration:**
- **Coordinate with robotic arm** for precise control
- **Trigger sorting actions** based on live detection
- **Monitor system health** and performance
- **Optimize sorting parameters** based on live data

---

## 🎉 **Ready to Use!**

**Your AI Tomato Sorter now has live camera monitoring!**

**Access it at: http://localhost:5001 → Live Camera**

**Features:**
- ✅ **Live camera feed** for real-time monitoring
- ✅ **Automatic classification** of tomatoes
- ✅ **Image capture** and analysis
- ✅ **Production-ready** for robotic sorting
- ✅ **Web-based interface** for easy control

**Perfect for monitoring your robotic tomato sorting system!** 📹🤖🍅✨
