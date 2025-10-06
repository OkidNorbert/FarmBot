# 🤖 Automated AI Training for Agricultural Crops

## Overview
This system automates the entire process of training AI models for new crops. Just organize your photos and run one command!

## 🚀 Quick Start

### 1. Setup New Crop Dataset
```bash
# Create folder structure for a new crop
python setup_new_crop.py strawberry

# This creates:
# datasets/strawberry/
# ├── ripe/
# ├── unripe/
# ├── overripe/
# └── damaged/
```

### 2. Add Your Images
- Place your crop images in the appropriate class folders
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Recommended: 50-100+ images per class for best results

### 3. Train Automatically
```bash
# Train the model (one command!)
python auto_train.py --dataset_path datasets/strawberry --crop_name strawberry

# With custom parameters
python auto_train.py --dataset_path datasets/strawberry --crop_name strawberry --epochs 50 --batch_size 16
```

### 4. Test Your Model
```bash
# Test with a new image
python models/strawberry/strawberry_inference.py --image path/to/test/strawberry.jpg
```

## 📁 Folder Structure

### Input Dataset Structure
```
your_crop_dataset/
├── ripe/           # Ripe crop images
├── unripe/         # Unripe crop images
├── overripe/       # Overripe crop images
└── damaged/        # Damaged crop images
```

### Output Structure (Auto-Generated)
```
models/
└── your_crop/
    ├── best_model.pth              # Best performing model
    ├── final_model.pth             # Final epoch model
    ├── training_metadata.json      # Training information
    ├── training_history.json       # Detailed training history
    └── your_crop_inference.py     # Ready-to-use inference script
```

## 🎯 Features

### ✅ Fully Automated
- **Dataset Analysis**: Automatically analyzes your dataset structure
- **Data Splitting**: Creates train/validation/test splits automatically
- **Model Training**: Trains the model with optimal settings
- **Inference Script**: Generates a ready-to-use inference script
- **Metadata**: Saves all training information for future reference

### ✅ Flexible Architecture
- **Any Number of Classes**: Works with 2, 3, 4, or more classes
- **Any Crop Type**: Works with any agricultural crop
- **Custom Parameters**: Adjust epochs, batch size, learning rate
- **Device Agnostic**: Works on CPU or GPU

### ✅ Production Ready
- **Best Model Selection**: Automatically saves the best performing model
- **Training History**: Complete training logs for analysis
- **Easy Deployment**: Generated inference script is ready for production
- **Metadata Tracking**: All training details saved for reproducibility

## 📊 Example Usage

### Example 1: Strawberry Classification
```bash
# 1. Setup
python setup_new_crop.py strawberry

# 2. Add images to datasets/strawberry/ripe/, datasets/strawberry/unripe/, etc.

# 3. Train
python auto_train.py --dataset_path datasets/strawberry --crop_name strawberry

# 4. Test
python models/strawberry/strawberry_inference.py --image test_strawberry.jpg
```

### Example 2: Apple Quality Control
```bash
# 1. Setup
python setup_new_crop.py apple

# 2. Add images to datasets/apple/ (fresh/, bruised/, rotten/, etc.)

# 3. Train with custom settings
python auto_train.py --dataset_path datasets/apple --crop_name apple --epochs 50 --batch_size 16

# 4. Test
python models/apple/apple_inference.py --image test_apple.jpg
```

## 🔧 Advanced Usage

### Custom Training Parameters
```bash
python auto_train.py \
    --dataset_path datasets/corn \
    --crop_name corn \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --output_dir custom_models
```

### Batch Processing Multiple Crops
```bash
# Train multiple crops in sequence
for crop in strawberry apple corn tomato; do
    python auto_train.py --dataset_path datasets/$crop --crop_name $crop
done
```

## 📈 Training Output

The system provides detailed feedback during training:

```
🌱 Auto-Training Pipeline for: strawberry
📁 Dataset: datasets/strawberry
💾 Model output: models/strawberry
============================================================

🔍 Analyzing dataset structure...
  📂 ripe: 150 images
  📂 unripe: 120 images
  📂 overripe: 80 images
  📂 damaged: 60 images

📊 Total images: 410
🏷️  Classes found: ['damaged', 'overripe', 'ripe', 'unripe']

📋 Creating data splits (train:0.8, val:0.1, test:0.1)...
  ✅ ripe: 120 train, 15 val, 15 test
  ✅ unripe: 96 train, 12 val, 12 test
  ✅ overripe: 64 train, 8 val, 8 test
  ✅ damaged: 48 train, 6 val, 6 test

📄 Creating data.yaml...
  ✅ Created datasets/strawberry/data.yaml

🚀 Starting training...
   Epochs: 30
   Batch size: 32
   Learning rate: 0.001

📊 Training with 328 images, validating with 41 images
🏷️  Classes: ['damaged', 'overripe', 'ripe', 'unripe']
🖥️  Using device: cpu

Epoch  1/30: Train Loss: 1.2345, Train Acc: 45.67%, Val Loss: 1.1234, Val Acc: 48.78%
Epoch  2/30: Train Loss: 0.9876, Train Acc: 52.34%, Val Loss: 0.9876, Val Acc: 56.10%
...
Epoch 30/30: Train Loss: 0.0123, Train Acc: 98.78%, Val Loss: 0.0456, Val Acc: 95.12%

✅ Training completed!
📊 Best validation accuracy: 95.12%
💾 Model saved to: models/strawberry

📝 Creating inference script...
  ✅ Created models/strawberry/strawberry_inference.py

🎉 AUTOMATED TRAINING COMPLETE!
============================================================
🌱 Crop: strawberry
📊 Classes: ['damaged', 'overripe', 'ripe', 'unripe']
🎯 Best Accuracy: 95.12%
💾 Model Directory: models/strawberry
🔧 Inference Script: models/strawberry/strawberry_inference.py

📋 Next Steps:
   1. Test your model: python models/strawberry/strawberry_inference.py --image path/to/test/image.jpg
   2. Deploy to production using the model in: models/strawberry
   3. Check training_metadata.json for detailed information
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No images found"**
   - Check that your images are in the correct class folders
   - Ensure images are in supported formats (.jpg, .jpeg, .png)
   - Check file permissions

2. **"Very small dataset warning"**
   - Add more images to each class (aim for 50+ per class)
   - Use data augmentation techniques

3. **"Low accuracy"**
   - Increase number of training epochs
   - Add more diverse images to your dataset
   - Check image quality and lighting

4. **"Out of memory"**
   - Reduce batch size (try 16 or 8)
   - Use smaller image sizes
   - Close other applications

### Performance Tips

1. **Dataset Quality**
   - Use high-quality, well-lit images
   - Include variety in angles and conditions
   - Remove blurry or unclear images

2. **Training Parameters**
   - More epochs for complex datasets
   - Larger batch size for faster training (if you have enough memory)
   - Adjust learning rate if training is too slow/fast

3. **Hardware Optimization**
   - Use GPU if available (automatically detected)
   - Close unnecessary applications during training
   - Use SSD storage for faster data loading

## 📚 Generated Files Explained

### `best_model.pth`
- The best performing model during training
- Use this for production deployment
- Selected based on validation accuracy

### `final_model.pth`
- The model from the final training epoch
- May not be the best performing model
- Useful for comparison

### `training_metadata.json`
```json
{
  "crop_name": "strawberry",
  "dataset_path": "/path/to/datasets/strawberry",
  "num_classes": 4,
  "class_names": ["damaged", "overripe", "ripe", "unripe"],
  "training_samples": 328,
  "validation_samples": 41,
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.001,
  "best_val_accuracy": 95.12,
  "final_val_accuracy": 95.12,
  "training_date": "2024-01-15T10:30:00",
  "device": "cpu"
}
```

### `training_history.json`
- Complete training history with loss and accuracy for each epoch
- Useful for analyzing training progress
- Can be used to create training plots

### `{crop_name}_inference.py`
- Ready-to-use inference script
- Automatically loads the correct model and classes
- Simple command-line interface

## 🎯 Best Practices

1. **Dataset Preparation**
   - Organize images by class in separate folders
   - Use descriptive class names (ripe, unripe, damaged, etc.)
   - Ensure balanced dataset (similar number of images per class)

2. **Image Quality**
   - Use consistent lighting conditions
   - Include variety in angles and distances
   - Remove blurry or unclear images

3. **Training Strategy**
   - Start with default parameters
   - Monitor training progress
   - Adjust parameters if needed

4. **Testing**
   - Test with images not seen during training
   - Test with various conditions and lighting
   - Validate performance meets your requirements

## 🚀 Production Deployment

Once training is complete, you can deploy your model:

1. **Use the generated inference script** for simple deployments
2. **Integrate the model** into your existing applications
3. **Deploy to edge devices** (Raspberry Pi, etc.)
4. **Create web APIs** using the trained model

The automated training system makes it easy to create new crop classifiers quickly and efficiently!
