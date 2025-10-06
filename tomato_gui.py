#!/usr/bin/env python3
"""
Tomato Classification GUI Application
Easy-to-use interface for training, testing, and deploying tomato classification models
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import threading
import subprocess
import os
import sys
from pathlib import Path
import json
import time
import cv2
from PIL import Image, ImageTk
import numpy as np

class TomatoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍅 AI Tomato Sorter - Classification GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.dataset_path = tk.StringVar(value="tomato_dataset")
        self.model_path = tk.StringVar(value="tomato_classifier.pth")
        self.epochs = tk.IntVar(value=50)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.device = tk.StringVar(value="auto")
        
        # Training variables
        self.training_active = False
        self.training_process = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_dataset_tab()
        self.setup_training_tab()
        self.setup_inference_tab()
        self.setup_deployment_tab()
        self.setup_help_tab()
        
    def setup_dataset_tab(self):
        """Setup dataset management tab"""
        dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(dataset_frame, text="📁 Dataset")
        
        # Dataset path selection
        ttk.Label(dataset_frame, text="Dataset Path:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        # Dataset analysis
        ttk.Button(dataset_frame, text="🔍 Analyze Dataset", command=self.analyze_dataset, 
                  style='Accent.TButton').grid(row=1, column=0, columnspan=3, pady=10)
        
        # Dataset info display
        self.dataset_info = scrolledtext.ScrolledText(dataset_frame, height=15, width=80)
        self.dataset_info.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        dataset_frame.grid_rowconfigure(2, weight=1)
        dataset_frame.grid_columnconfigure(1, weight=1)
        
    def setup_training_tab(self):
        """Setup training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🚀 Training")
        
        # Training parameters
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters", padding=10)
        params_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Spinbox(params_frame, from_=10, to=200, textvariable=self.epochs, width=10).grid(row=0, column=1, padx=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5)
        ttk.Spinbox(params_frame, from_=8, to=128, textvariable=self.batch_size, width=10).grid(row=0, column=3, padx=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Spinbox(params_frame, from_=0.0001, to=0.01, increment=0.0001, textvariable=self.learning_rate, width=10).grid(row=1, column=1, padx=5)
        
        # Device selection
        ttk.Label(params_frame, text="Device:").grid(row=1, column=2, sticky='w', padx=5)
        device_combo = ttk.Combobox(params_frame, textvariable=self.device, values=["auto", "cpu", "cuda"], width=10)
        device_combo.grid(row=1, column=3, padx=5)
        
        # Training controls
        controls_frame = ttk.Frame(training_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.train_button = ttk.Button(controls_frame, text="🚀 Start Training", command=self.start_training, style='Accent.TButton')
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="⏹️ Stop Training", command=self.stop_training, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Training progress
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        # Training log
        self.training_log = scrolledtext.ScrolledText(training_frame, height=15, width=80)
        self.training_log.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        training_frame.grid_rowconfigure(3, weight=1)
        training_frame.grid_columnconfigure(0, weight=1)
        
    def setup_inference_tab(self):
        """Setup inference tab"""
        inference_frame = ttk.Frame(self.notebook)
        self.notebook.add(inference_frame, text="🔍 Inference")
        
        # Model selection
        model_frame = ttk.LabelFrame(inference_frame, text="Model Selection", padding=10)
        model_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5)
        
        # Inference options
        options_frame = ttk.LabelFrame(inference_frame, text="Inference Options", padding=10)
        options_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        self.inference_source = tk.StringVar(value="camera")
        ttk.Radiobutton(options_frame, text="Camera", variable=self.inference_source, value="camera").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Radiobutton(options_frame, text="Image File", variable=self.inference_source, value="image").grid(row=0, column=1, sticky='w', padx=5)
        
        self.image_path = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.image_path, width=40, state='disabled').grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(options_frame, text="Browse Image", command=self.browse_image).grid(row=1, column=2, padx=5)
        
        # Inference controls
        controls_frame = ttk.Frame(inference_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.inference_button = ttk.Button(controls_frame, text="🔍 Start Inference", command=self.start_inference, style='Accent.TButton')
        self.inference_button.grid(row=0, column=0, padx=5)
        
        self.stop_inference_button = ttk.Button(controls_frame, text="⏹️ Stop Inference", command=self.stop_inference, state='disabled')
        self.stop_inference_button.grid(row=0, column=1, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(inference_frame, text="Results", padding=10)
        results_frame.grid(row=3, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(results_frame, height=10, width=80)
        self.result_text.grid(row=0, column=0, sticky='nsew')
        
        # Configure grid weights
        inference_frame.grid_rowconfigure(3, weight=1)
        inference_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
    def setup_deployment_tab(self):
        """Setup deployment tab"""
        deployment_frame = ttk.Frame(self.notebook)
        self.notebook.add(deployment_frame, text="🚀 Deployment")
        
        # Export options
        export_frame = ttk.LabelFrame(deployment_frame, text="Model Export", padding=10)
        export_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        ttk.Button(export_frame, text="📦 Export to ONNX", command=self.export_onnx).grid(row=0, column=0, padx=5)
        ttk.Button(export_frame, text="📱 Export to TFLite", command=self.export_tflite).grid(row=0, column=1, padx=5)
        
        # Raspberry Pi deployment
        pi_frame = ttk.LabelFrame(deployment_frame, text="Raspberry Pi Deployment", padding=10)
        pi_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        ttk.Button(pi_frame, text="🍓 Deploy to Raspberry Pi", command=self.deploy_to_pi).grid(row=0, column=0, padx=5)
        ttk.Button(pi_frame, text="🔧 Test Pi Connection", command=self.test_pi_connection).grid(row=0, column=1, padx=5)
        
        # Web interface
        web_frame = ttk.LabelFrame(deployment_frame, text="Web Interface", padding=10)
        web_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        ttk.Button(web_frame, text="🌐 Start Web Server", command=self.start_web_server).grid(row=0, column=0, padx=5)
        ttk.Button(web_frame, text="🛑 Stop Web Server", command=self.stop_web_server).grid(row=0, column=1, padx=5)
        
        # Deployment log
        self.deployment_log = scrolledtext.ScrolledText(deployment_frame, height=15, width=80)
        self.deployment_log.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        deployment_frame.grid_rowconfigure(3, weight=1)
        deployment_frame.grid_columnconfigure(0, weight=1)
        
    def setup_help_tab(self):
        """Setup help tab"""
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="❓ Help")
        
        # Help content
        help_text = """
🍅 AI Tomato Sorter - Classification GUI

This application provides an easy-to-use interface for training, testing, and deploying tomato classification models.

📁 DATASET TAB:
- Select your dataset path (should contain train/ and val/ folders)
- Analyze dataset structure and class distribution
- Verify dataset is properly organized

🚀 TRAINING TAB:
- Configure training parameters (epochs, batch size, learning rate)
- Start/stop training process
- Monitor training progress and logs
- View training curves and metrics

🔍 INFERENCE TAB:
- Test trained models on images or camera
- View classification results and confidence scores
- Compare different models

🚀 DEPLOYMENT TAB:
- Export models to different formats (ONNX, TFLite)
- Deploy to Raspberry Pi
- Start web interface for remote access

QUICK START:
1. Set dataset path in Dataset tab
2. Click "Analyze Dataset" to verify structure
3. Go to Training tab and click "Start Training"
4. Wait for training to complete
5. Test your model in Inference tab
6. Deploy using Deployment tab

TROUBLESHOOTING:
- Ensure PyTorch is installed: pip install torch torchvision
- Check dataset structure matches expected format
- Verify model files exist before inference
- Check device compatibility for training

For more help, see the documentation files in the project directory.
        """
        
        help_display = scrolledtext.ScrolledText(help_frame, height=25, width=80)
        help_display.insert('1.0', help_text)
        help_display.config(state='disabled')
        help_display.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        help_frame.grid_rowconfigure(0, weight=1)
        help_frame.grid_columnconfigure(0, weight=1)
        
    def browse_dataset(self):
        """Browse for dataset directory"""
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.dataset_path.set(path)
            
    def browse_model(self):
        """Browse for model file"""
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            
    def browse_image(self):
        """Browse for image file"""
        path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.image_path.set(path)
            
    def analyze_dataset(self):
        """Analyze dataset structure"""
        dataset_path = self.dataset_path.get()
        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", "Dataset path does not exist!")
            return
            
        try:
            # Run dataset analysis
            result = subprocess.run([
                sys.executable, "quick_train.py", 
                "--dataset", dataset_path, 
                "--analyze"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            # Display results
            self.dataset_info.delete('1.0', tk.END)
            self.dataset_info.insert('1.0', result.stdout)
            
            if result.stderr:
                self.dataset_info.insert(tk.END, f"\nErrors:\n{result.stderr}")
                
        except Exception as e:
            # Fallback to manual analysis
            self.manual_dataset_analysis(dataset_path)
            
    def manual_dataset_analysis(self, dataset_path):
        """Manual dataset analysis as fallback"""
        try:
            from pathlib import Path
            import os
            
            analysis_text = "🍅 Dataset Analysis\n" + "="*50 + "\n\n"
            
            # Check if dataset exists
            if not os.path.exists(dataset_path):
                analysis_text += "❌ Dataset path does not exist!\n"
                self.dataset_info.delete('1.0', tk.END)
                self.dataset_info.insert('1.0', analysis_text)
                return
            
            # Analyze train split
            train_path = Path(dataset_path) / "train"
            if train_path.exists():
                analysis_text += "TRAIN Split:\n"
                total_train = 0
                for class_folder in train_path.iterdir():
                    if class_folder.is_dir():
                        count = len(list(class_folder.glob('*.jpg')))
                        total_train += count
                        analysis_text += f"  {class_folder.name}: {count} images\n"
                analysis_text += f"  Total: {total_train} images\n\n"
            
            # Analyze val split
            val_path = Path(dataset_path) / "val"
            if val_path.exists():
                analysis_text += "VAL Split:\n"
                total_val = 0
                for class_folder in val_path.iterdir():
                    if class_folder.is_dir():
                        count = len(list(class_folder.glob('*.jpg')))
                        total_val += count
                        analysis_text += f"  {class_folder.name}: {count} images\n"
                analysis_text += f"  Total: {total_val} images\n\n"
            
            # Class mapping
            analysis_text += "Class Mapping:\n"
            analysis_text += "  Unripe -> not_ready (Class 0)\n"
            analysis_text += "  Ripe -> ready (Class 1)\n"
            analysis_text += "  Old -> spoilt (Class 2)\n"
            analysis_text += "  Damaged -> spoilt (Class 2)\n\n"
            
            # Check data.yaml
            data_yaml_path = Path(dataset_path) / "data.yaml"
            if data_yaml_path.exists():
                analysis_text += "✅ data.yaml found\n"
            else:
                analysis_text += "⚠️ data.yaml not found - will be created during training\n"
            
            analysis_text += f"\nTotal Dataset: {total_train + total_val} images"
            
            self.dataset_info.delete('1.0', tk.END)
            self.dataset_info.insert('1.0', analysis_text)
            
        except Exception as e:
            self.dataset_info.delete('1.0', tk.END)
            self.dataset_info.insert('1.0', f"Error analyzing dataset: {str(e)}")
            
    def start_training(self):
        """Start training process"""
        if self.training_active:
            return
            
        self.training_active = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress.start()
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
        
    def run_training(self):
        """Run training process"""
        try:
            # Check if PyTorch is available
            try:
                import torch
                import torchvision
                self.root.after(0, self.update_training_log, "✅ PyTorch found - starting training...\n")
            except ImportError:
                self.root.after(0, self.update_training_log, "❌ PyTorch not found. Installing...\n")
                # Try to install PyTorch
                install_result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "torch", "torchvision", "--no-cache-dir"
                ], capture_output=True, text=True, timeout=300)
                
                if install_result.returncode != 0:
                    self.root.after(0, self.training_error, "Failed to install PyTorch. Please install manually: pip install torch torchvision")
                    return
            
            cmd = [
                sys.executable, "train_tomato_classifier.py",
                "--dataset", self.dataset_path.get(),
                "--epochs", str(self.epochs.get()),
                "--batch_size", str(self.batch_size.get()),
                "--lr", str(self.learning_rate.get())
            ]
            
            self.root.after(0, self.update_training_log, f"🚀 Starting training with command: {' '.join(cmd)}\n")
            
            self.training_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=os.getcwd()
            )
            
            # Read output in real-time
            for line in iter(self.training_process.stdout.readline, ''):
                if not self.training_active:
                    break
                self.root.after(0, self.update_training_log, line)
                
            self.training_process.wait()
            
        except subprocess.TimeoutExpired:
            self.root.after(0, self.training_error, "Training timeout - PyTorch installation took too long")
        except Exception as e:
            self.root.after(0, self.training_error, str(e))
        finally:
            self.root.after(0, self.training_finished)
            
    def update_training_log(self, line):
        """Update training log display"""
        self.training_log.insert(tk.END, line)
        self.training_log.see(tk.END)
        
    def training_error(self, error):
        """Handle training error"""
        self.training_log.insert(tk.END, f"\nError: {error}\n")
        
    def training_finished(self):
        """Handle training completion"""
        self.training_active = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress.stop()
        
        if self.training_process and self.training_process.returncode == 0:
            messagebox.showinfo("Success", "Training completed successfully!")
        else:
            messagebox.showerror("Error", "Training failed. Check the log for details.")
            
    def stop_training(self):
        """Stop training process"""
        self.training_active = False
        if self.training_process:
            self.training_process.terminate()
            
    def start_inference(self):
        """Start inference process"""
        model_path = self.model_path.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file does not exist!")
            return
            
        if self.inference_source.get() == "image":
            image_path = self.image_path.get()
            if not image_path or not os.path.exists(image_path):
                messagebox.showerror("Error", "Image file does not exist!")
                return
                
        # Start inference in separate thread
        inference_thread = threading.Thread(target=self.run_inference)
        inference_thread.daemon = True
        inference_thread.start()
        
    def run_inference(self):
        """Run inference process"""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path.get()):
                self.root.after(0, self.inference_error, f"Model file not found: {self.model_path.get()}")
                return
            
            # Check if inference script exists
            if not os.path.exists("inference_classifier.py"):
                self.root.after(0, self.inference_error, "inference_classifier.py not found. Please ensure it's in the project directory.")
                return
            
            if self.inference_source.get() == "camera":
                cmd = [sys.executable, "inference_classifier.py", "--model", self.model_path.get(), "--source", "0"]
                self.root.after(0, self.update_inference_results, "🔍 Starting camera inference...\n", "")
            else:
                if not self.image_path.get() or not os.path.exists(self.image_path.get()):
                    self.root.after(0, self.inference_error, "Please select a valid image file.")
                    return
                cmd = [sys.executable, "inference_classifier.py", "--model", self.model_path.get(), "--image", self.image_path.get()]
                self.root.after(0, self.update_inference_results, f"🔍 Analyzing image: {os.path.basename(self.image_path.get())}\n", "")
                
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=30)
            
            self.root.after(0, self.update_inference_results, result.stdout, result.stderr)
            
        except subprocess.TimeoutExpired:
            self.root.after(0, self.inference_error, "Inference timeout - model may be too large or slow")
        except Exception as e:
            self.root.after(0, self.inference_error, str(e))
            
    def update_inference_results(self, stdout, stderr):
        """Update inference results display"""
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', stdout)
        if stderr:
            self.result_text.insert(tk.END, f"\nErrors:\n{stderr}")
            
    def inference_error(self, error):
        """Handle inference error"""
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', f"Error: {error}")
        
    def stop_inference(self):
        """Stop inference process"""
        # Implementation for stopping inference
        pass
        
    def export_onnx(self):
        """Export model to ONNX format"""
        model_path = self.model_path.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found!")
            return
            
        try:
            # Create ONNX export script
            onnx_script = """
import torch
import torch.onnx
from train_tomato_classifier import TomatoClassifier

# Load model
model = TomatoClassifier(num_classes=3)
model.load_state_dict(torch.load('{}', map_location='cpu'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, 'tomato_classifier.onnx', 
                  export_params=True, opset_version=11, 
                  input_names=['input'], output_names=['output'])

print("✅ Model exported to tomato_classifier.onnx")
""".format(model_path)
            
            with open("export_onnx.py", "w") as f:
                f.write(onnx_script)
            
            # Run export
            result = subprocess.run([sys.executable, "export_onnx.py"], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "Model exported to ONNX format!")
                self.deployment_log.insert(tk.END, result.stdout)
            else:
                messagebox.showerror("Error", f"ONNX export failed: {result.stderr}")
                
        except Exception as e:
            messagebox.showerror("Error", f"ONNX export failed: {str(e)}")
        
    def export_tflite(self):
        """Export model to TFLite format"""
        messagebox.showinfo("Info", "TFLite export requires TensorFlow. Install with: pip install tensorflow")
        
    def deploy_to_pi(self):
        """Deploy to Raspberry Pi"""
        # Create deployment script
        deploy_script = """
#!/bin/bash
echo "🍓 Deploying to Raspberry Pi..."

# Check if model exists
if [ ! -f "tomato_classifier.pth" ]; then
    echo "❌ Model file not found!"
    exit 1
fi

# Create deployment package
mkdir -p pi_deployment
cp tomato_classifier.pth pi_deployment/
cp inference_classifier.py pi_deployment/
cp requirements.txt pi_deployment/
cp data.yaml pi_deployment/

# Create Pi startup script
cat > pi_deployment/start_pi.sh << 'EOF'
#!/bin/bash
echo "🍅 Starting Tomato Sorter on Raspberry Pi..."
source tomato_sorter_env/bin/activate
python inference_classifier.py --model tomato_classifier.pth --source 0
EOF

chmod +x pi_deployment/start_pi.sh

echo "✅ Deployment package created in pi_deployment/"
echo "📦 Copy pi_deployment/ to your Raspberry Pi"
echo "🚀 Run: ./start_pi.sh on the Pi"
"""
        
        with open("deploy_to_pi.sh", "w") as f:
            f.write(deploy_script)
        
        os.chmod("deploy_to_pi.sh", 0o755)
        
        # Run deployment
        result = subprocess.run(["./deploy_to_pi.sh"], capture_output=True, text=True, cwd=os.getcwd())
        
        self.deployment_log.insert(tk.END, result.stdout)
        if result.stderr:
            self.deployment_log.insert(tk.END, f"\nErrors:\n{result.stderr}")
        
        messagebox.showinfo("Success", "Deployment package created! Check pi_deployment/ folder.")
        
    def test_pi_connection(self):
        """Test Raspberry Pi connection"""
        # Simple ping test
        pi_ip = tk.simpledialog.askstring("Pi Connection Test", "Enter Raspberry Pi IP address:")
        if pi_ip:
            try:
                result = subprocess.run(["ping", "-c", "3", pi_ip], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    messagebox.showinfo("Success", f"Raspberry Pi at {pi_ip} is reachable!")
                    self.deployment_log.insert(tk.END, f"✅ Pi {pi_ip} is online\n")
                else:
                    messagebox.showerror("Error", f"Cannot reach Raspberry Pi at {pi_ip}")
                    self.deployment_log.insert(tk.END, f"❌ Pi {pi_ip} is offline\n")
            except Exception as e:
                messagebox.showerror("Error", f"Connection test failed: {str(e)}")
        
    def start_web_server(self):
        """Start web server"""
        try:
            # Create simple web server
            web_script = """
from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <html>
    <head><title>🍅 Tomato Sorter</title></head>
    <body>
        <h1>🍅 AI Tomato Sorter</h1>
        <p>Web interface is running!</p>
        <p>Model: tomato_classifier.pth</p>
        <p>Status: Ready for inference</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
"""
            
            with open("web_server.py", "w") as f:
                f.write(web_script)
            
            # Start web server in background
            self.web_process = subprocess.Popen([sys.executable, "web_server.py"], 
                                              cwd=os.getcwd())
            
            messagebox.showinfo("Success", "Web server started at http://localhost:5000")
            self.deployment_log.insert(tk.END, "🌐 Web server started at http://localhost:5000\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start web server: {str(e)}")
        
    def stop_web_server(self):
        """Stop web server"""
        try:
            if hasattr(self, 'web_process') and self.web_process:
                self.web_process.terminate()
                messagebox.showinfo("Success", "Web server stopped")
                self.deployment_log.insert(tk.END, "🛑 Web server stopped\n")
            else:
                messagebox.showinfo("Info", "No web server running")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop web server: {str(e)}")

def main():
    """Main function"""
    root = tk.Tk()
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create custom styles
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    app = TomatoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
