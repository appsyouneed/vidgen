#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg wget unzip

# Install Python dependencies
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing Python dependencies..."
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
    python3 -m pip install -r requirements.txt --break-system-packages --ignore-installed
else
    echo "Python dependencies already installed, skipping..."
fi

# Download RIFE model
if [ ! -f "RIFEv4.26_0921.zip" ]; then
    echo "Downloading RIFE model..."
    wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o RIFEv4.26_0921.zip
else
    echo "RIFE model already downloaded, skipping..."
fi

# Download RIFE model directory
if [ ! -d "train_log/model" ]; then
    echo "Downloading RIFE model directory..."
    git clone https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    cp -r /tmp/rife/model train_log/
    rm -rf /tmp/rife
else
    echo "RIFE model directory already exists, skipping..."
fi

echo "=== Setup Complete ==="
echo ""
echo "Killing any existing Python processes..."
pkill -9 python3 2>/dev/null || true
sleep 2

echo "Starting application..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 app.py
echo "Models will be cached in: ~/.cache/huggingface/"
echo "RIFE model will be downloaded to: ./RIFEv4.26_0921.zip"
echo ""
echo "To run the application:"
echo "  python app.py"
echo ""
echo "The app will be accessible at: http://0.0.0.0:7860"
