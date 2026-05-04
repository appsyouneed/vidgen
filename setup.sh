#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

echo "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found, installing..."
    apt-get update && apt-get install -y python3-pip
fi

echo "Installing system dependencies..."
apt-get install -y ffmpeg wget unzip git

echo "Creating temp directory..."
mkdir -p "$SCRIPT_DIR/tmp"
chmod 1777 "$SCRIPT_DIR/tmp"

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Detecting CUDA version..."
CUDA_VERSION=""
if [ -f /usr/local/cuda/version.json ]; then
    CUDA_VERSION=$(grep -oP '"cuda".*?"version".*?"\K[0-9]+\.[0-9]+' /usr/local/cuda/version.json | head -1)
elif command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
fi

if [ -n "$CUDA_VERSION" ]; then
    echo "Detected CUDA $CUDA_VERSION"
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]); then
        TORCH_CUDA="cu124"
    else
        TORCH_CUDA="cu121"
    fi
else
    echo "CUDA version not detected, defaulting to cu121"
    TORCH_CUDA="cu121"
fi

if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed and working, skipping..."
else
    echo "Installing PyTorch with $TORCH_CUDA support..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_CUDA --break-system-packages --no-cache-dir
fi

echo "Installing Python dependencies..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" --break-system-packages --ignore-installed typing-extensions --no-cache-dir

echo "Ensuring critical packages are correctly installed..."
pip3 install Pillow "transformers<5" --break-system-packages --no-cache-dir --force-reinstall

echo "Fixing pyOpenSSL compatibility..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip3 install pyopenssl --break-system-packages

echo "Setting up RIFE interpolation model..."
if [ ! -d "$SCRIPT_DIR/train_log/model" ] || [ ! -f "$SCRIPT_DIR/train_log/RIFE_HDv3.py" ]; then
    echo "Removing incomplete RIFE installation..."
    rm -rf "$SCRIPT_DIR/train_log" "$SCRIPT_DIR/__MACOSX" "$SCRIPT_DIR/RIFEv4.26_0921.zip"

    echo "Downloading RIFE model architecture..."
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    mkdir -p "$SCRIPT_DIR/train_log"
    cp -r /tmp/rife/model "$SCRIPT_DIR/train_log/"

    echo "Downloading RIFE weights..."
    wget -q -P "$SCRIPT_DIR" https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o "$SCRIPT_DIR/RIFEv4.26_0921.zip" -d "$SCRIPT_DIR"

    rm -rf /tmp/rife "$SCRIPT_DIR/__MACOSX"
    echo "RIFE model installed successfully"
else
    echo "RIFE model already installed, skipping..."
fi

echo "=== Setup Complete ==="
echo ""
echo "Setting up systemd service..."

cat > /etc/systemd/system/vidgen.service <<EOF
[Unit]
Description=Vidgen Video Generation Application
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment="PYTHONUNBUFFERED=1"
Environment="HF_HOME=/root/.cache/huggingface"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="TMPDIR=$SCRIPT_DIR/tmp"
Environment="TEMP=$SCRIPT_DIR/tmp"
Environment="TMP=$SCRIPT_DIR/tmp"
ExecStartPre=/bin/mkdir -p $SCRIPT_DIR/tmp
ExecStartPre=/bin/chmod 1777 $SCRIPT_DIR/tmp
ExecStart=/usr/bin/python3 $SCRIPT_DIR/app.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/vidgen.log
StandardError=append:$SCRIPT_DIR/vidgen.log

[Install]
WantedBy=multi-user.target
EOF

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
echo "  tail -f $SCRIPT_DIR/vidgen.log"
echo ""
echo "To run manually: python3 $SCRIPT_DIR/app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
