#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

# Detect script directory (works regardless of where it's run from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto sudo if not root
if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

UBUNTU_VER=$(lsb_release -rs)

echo "Installing system dependencies..."
PKGS="python3-pip python3-venv ffmpeg wget unzip git"
if (( $(echo "$UBUNTU_VER < 24" | bc -l) )); then
    echo "Ubuntu $UBUNTU_VER detected: adding python3.10-venv..."
    PKGS="$PKGS python3.10-venv"
fi
apt-get update && apt-get install -y $PKGS

echo "Creating temp directory..."
mkdir -p "$SCRIPT_DIR/tmp"
chmod 1777 "$SCRIPT_DIR/tmp"

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Creating Python virtual environment..."
rm -rf "$SCRIPT_DIR/venv"
python3 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

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
pip install -r "$SCRIPT_DIR/requirements.txt" --ignore-installed

echo "Fixing pyOpenSSL compatibility..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip install --upgrade pyopenssl

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

    echo "Cleaning up..."
    rm -rf /tmp/rife "$SCRIPT_DIR/__MACOSX"

    echo "RIFE model installed successfully"
else
    echo "RIFE model already installed, skipping..."
fi

echo "=== Setup Complete ==="
echo ""
echo "Setting up systemd service..."

# Write service file dynamically with correct paths
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
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/app.py
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
