#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

echo "Creating temp directory..."
mkdir -p /root/vidgen/tmp
chmod 1777 /root/vidgen/tmp

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip python3-venv ffmpeg wget unzip

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

if [ ! -f "RIFEv4.26_0921.zip" ]; then
    echo "Downloading RIFE model..."
    wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o RIFEv4.26_0921.zip
else
    echo "RIFE model already downloaded, skipping..."
fi

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
echo "Enabling vidgen service..."
systemctl daemon-reload
systemctl enable vidgen
systemctl start vidgen

echo ""
echo "Service commands:"
echo "  ./start.sh  - Start vidgen"
echo "  ./stop.sh   - Stop vidgen"
echo "  systemctl status vidgen - Check status"
echo ""
echo "The app will be accessible at: http://0.0.0.0:7860"
