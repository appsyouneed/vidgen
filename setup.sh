#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg wget unzip

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
python3 -m pip install -r requirements.txt --break-system-packages --break-system-packages --ignore-installed

echo "Downloading RIFE model..."
wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
unzip -o RIFEv4.26_0921.zip

echo "Downloading RIFE model directory..."
git clone https://github.com/hzwer/Practical-RIFE.git /tmp/rife
cp -r /tmp/rife/model train_log/
rm -rf /tmp/rife

echo "=== Setup Complete ==="
echo "Models will be cached in: ~/.cache/huggingface/"
echo "RIFE model will be downloaded to: ./RIFEv4.26_0921.zip"
echo ""
echo "To run the application:"
echo "  python app.py"
echo ""
echo "The app will be accessible at: http://0.0.0.0:7860"
