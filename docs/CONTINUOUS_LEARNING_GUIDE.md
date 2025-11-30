# Continuous Learning System Guide

## Overview

The Continuous Learning system allows you to improve your tomato classification model by providing feedback on predictions. This guide explains how the system works and how to use it effectively.

## How It Works

### 1. Prediction Confidence

**What is it?**
- Confidence is a value between 0.0 and 1.0 (0% to 100%)
- It represents how certain the model is about its prediction
- Higher confidence (e.g., 0.95) = model is very sure
- Lower confidence (e.g., 0.60) = model is uncertain

**What does it do?**
- **Tracking**: Stored with each feedback entry for analysis
- **Retraining Strategy**: Used to determine which images to include in retraining:
  - **Incorrect predictions** (wrong class) → Always included
  - **Correct but low confidence** (< 70%) → Also included (helps model learn from uncertainty)
  - **Correct with high confidence** (≥ 70%) → Excluded (model already knows these well)

### 2. Submitting Feedback

**What happens when you submit feedback?**

1. **Data Storage**:
   - Feedback is saved to `learning_data/feedback.json`
   - Each entry contains:
     ```json
     {
       "timestamp": "2024-01-15T10:30:00",
       "image_path": "learning_data/new_images/test_uploads/test_20240115_103000.jpg",
       "predicted_class": "ready",
       "correct_class": "not_ready",
       "confidence": 0.85
     }
     ```

2. **Image Reference**:
   - The `image_path` points to the actual image file
   - The image should already exist in `learning_data/new_images/test_uploads/`
   - When you test an image through the web interface, it's automatically saved there

3. **Statistics Update**:
   - Total feedback count increases
   - Incorrect predictions count updates
   - Retraining status updates if you have enough feedback (≥10 entries)

### 3. Image Storage and Labeling for Retraining

**Where are images stored?**

1. **Test Images** (Original):
   - Location: `learning_data/new_images/test_uploads/`
   - These are the original test images you uploaded
   - They remain in place and are referenced by path

2. **Retraining Dataset** (Created when you click "Retrain Model"):
   - Location: `learning_data/retraining_dataset/`
   - Structure:
     ```
     retraining_dataset/
     ├── unripe/          (from not_ready feedback)
     ├── ripe/            (from ready feedback)
     └── spoilt/          (from spoilt feedback)
     ```

**How are images labeled?**

When you click "Retrain Model", the system:

1. **Reads all feedback entries** from `feedback.json`

2. **Filters images** based on:
   - **Incorrect predictions**: Always included (model was wrong)
   - **Low confidence correct predictions**: Included if confidence < 70% (model was uncertain)

3. **Copies images** to the appropriate class folder:
   - `correct_class = "not_ready"` → `retraining_dataset/unripe/`
   - `correct_class = "ready"` → `retraining_dataset/ripe/`
   - `correct_class = "spoilt"` → `retraining_dataset/spoilt/`

4. **Preserves filenames** (with unique naming if duplicates exist)

**Example Flow:**

```
User tests image → Image saved to: learning_data/new_images/test_uploads/test_001.jpg
User provides feedback:
  - Predicted: "ready" (confidence: 0.65)
  - Correct: "not_ready"
  → Saved to feedback.json

User clicks "Retrain Model":
  → Image copied to: learning_data/retraining_dataset/unripe/test_001.jpg
  → Ready for training!
```

## Using the System

### Step-by-Step Workflow

1. **Test Images**:
   - Upload/test images through the web interface
   - Images are automatically saved to `learning_data/new_images/test_uploads/`
   - Prediction metadata (class, confidence) is saved automatically

2. **Review Predictions**:
   - Go to Continuous Learning page
   - View recent test images with prediction badges
   - Identify incorrect or uncertain predictions

3. **Provide Feedback**:
   - Click on an image to auto-fill the form
   - Select the correct class
   - Submit feedback

4. **Retrain Model**:
   - Wait until you have at least 10 feedback entries
   - Click "Retrain Model" button
   - System prepares dataset in `learning_data/retraining_dataset/`
   - Use this dataset to retrain your model

### Best Practices

1. **Focus on Errors**: Prioritize feedback for incorrect predictions
2. **Include Uncertain Cases**: Even if correct, low confidence (< 70%) helps the model
3. **Balance Classes**: Try to provide feedback for all three classes
4. **Quality over Quantity**: Accurate feedback is more valuable than volume
5. **Regular Retraining**: Retrain periodically as you accumulate feedback

## File Structure

```
learning_data/
├── feedback.json                    # All feedback entries
├── image_metadata.json              # Prediction metadata for images
├── new_images/
│   └── test_uploads/               # Original test images
│       ├── test_20240115_103000.jpg
│       └── ...
└── retraining_dataset/              # Created when retraining
    ├── unripe/
    │   ├── test_001.jpg
    │   └── ...
    ├── ripe/
    │   └── ...
    └── spoilt/
        └── ...
```

## Statistics Explained

- **Total Feedback**: Number of feedback entries submitted
- **Incorrect Predictions**: Count of entries where predicted ≠ correct
- **Learning Images**: Total number of test images in the system
- **Retraining Status**: "Yes" if you have ≥10 feedback entries with incorrect predictions

## Troubleshooting

**Q: Image not found when submitting feedback?**
- Make sure the image path is correct
- Check that the image exists in `learning_data/new_images/test_uploads/`

**Q: No images copied during retraining?**
- Check that you have feedback entries where predicted ≠ correct
- Verify image paths in feedback.json are valid
- Ensure images haven't been deleted

**Q: How do I use the retraining dataset?**
- The dataset is organized by class folders
- Use it with your training script (e.g., `train_tomato_classifier.py`)
- Combine with your original dataset for better results

## Next Steps

After preparing the retraining dataset:

1. **Combine Datasets**: Merge with your original training dataset
2. **Retrain Model**: Use your training script with the combined dataset
3. **Evaluate**: Test the new model and compare performance
4. **Iterate**: Continue providing feedback and retraining

