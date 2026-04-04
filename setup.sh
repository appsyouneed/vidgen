#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

echo "Creating temp directory..."
mkdir -p /root/vidgen/tmp
chmod 1777 /root/vidgen/tmp

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg wget unzip git

echo "Upgrading pip..."
pip3 install --upgrade pip --break-system-packages

echo "Installing PyTorch with CUDA 12.4 support..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --break-system-packages --ignore-installed

echo "Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages --ignore-installed

echo "Setting up RIFE interpolation model..."
if [ ! -d "train_log/model" ] || [ ! -f "train_log/RIFE_HDv3.py" ]; then
    echo "Removing incomplete RIFE installation..."
    rm -rf train_log __MACOSX RIFEv4.26_0921.zip
    
    echo "Downloading RIFE model architecture..."
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    mkdir -p train_log
    cp -r /tmp/rife/model train_log/
    
    echo "Downloading RIFE weights..."
    wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o RIFEv4.26_0921.zip
    
    echo "Cleaning up..."
    rm -rf /tmp/rife __MACOSX
    
    echo "RIFE model installed successfully"
else
    echo "RIFE model already installed, skipping..."
fi

echo "=== Setup Complete ==="
echo ""
echo "Setting up systemd service..."
cp vidgen.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable vidgen
systemctl start vidgen

echo ""
echo "Service commands:"
echo "  systemctl start vidgen   - Start vidgen"
echo "  systemctl stop vidgen    - Stop vidgen"
echo "  systemctl status vidgen  - Check status"
echo "  systemctl restart vidgen - Restart vidgen"
echo ""
echo "To run manually: python3 app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
