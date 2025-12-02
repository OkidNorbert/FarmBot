# Quick Start: Train YOLO via Web Interface 🚀

## 5-Minute Quick Start

### Step 1: Install Ultralytics (One-time)
```bash
pip install ultralytics
```

### Step 2: Open Web Interface
```bash
python web_interface.py
```
Open browser: `http://localhost:5000/training`

### Step 3: Start Training
1. Find your dataset in the list
2. Click **"Start Training"** button
3. In the modal:
   - **Model Type**: Select **"YOLO (Detection + Classification)"**
   - **Model Size**: Select **"Nano (n)"** (for first time)
   - **Epochs**: Enter **50** (or 100 for better results)
   - **Batch Size**: Enter **16** (or 8 if out of memory)
4. Click **"Start Training"**

### Step 4: Watch Training
- See real-time logs
- Watch progress bar
- Wait for completion (5-30 minutes depending on dataset size)

### Step 5: View Results
- Training charts appear automatically
- Model is saved and ready to use
- Test with "Test Model" button

## That's It! ✅

Your YOLO model is now trained and ready to use. The system will automatically detect and use it for all detection tasks.

## Need More Details?

See `YOLO_WEB_TRAINING_GUIDE.md` for complete guide with troubleshooting and best practices.

