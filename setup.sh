#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

UBUNTU_VER=$(lsb_release -rs)
if (( $(echo "$UBUNTU_VER < 24" | bc -l) )); then
    echo "Ubuntu $UBUNTU_VER detected: upgrading pip first..."
    pip install --upgrade pip
fi

echo "Creating temp directory..."
mkdir -p /root/vidgen/tmp
chmod 1777 /root/vidgen/tmp

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Creating Python virtual environment..."
python3 -m venv /root/vidgen/venv
source /root/vidgen/venv/bin/activate

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg wget unzip git

# --- CUDA 12.4 Toolkit (if not already installed) ---
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing CUDA 12.4 toolkit..."
    UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
    CUDA_DEB="cuda-keyring_1.1-1_all.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${CUDA_DEB}"
    dpkg -i "$CUDA_DEB"
    apt-get update
    apt-get install -y cuda-toolkit-12-4
    rm -f "$CUDA_DEB"
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /root/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc
    echo "CUDA 12.4 installed."
else
    echo "CUDA already installed: $(nvcc --version | head -1)"
fi

# Upgrade pip only if needed
if python3 -m pip install --upgrade pip --dry-run 2>&1 | grep -q "Would install"; then
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip
else
    echo "pip already up to date, skipping."
fi

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --ignore-installed

echo "Installing Python dependencies..."
pip install -r requirements.txt --ignore-installed

echo "Fixing pyOpenSSL compatibility..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip install --upgrade pyopenssl

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
echo "View live output:"
echo "  tail -f /root/vidgen/vidgen.log"
echo ""
echo "To run manually: python3 app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
