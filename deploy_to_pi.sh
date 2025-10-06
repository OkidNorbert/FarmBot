
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
