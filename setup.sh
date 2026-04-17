#!/bin/bash
set -e

echo "=== Vidgen Setup ==="

echo "Creating directories..."
mkdir -p /root/vidgen/tmp
chmod 1777 /root/vidgen/tmp
mkdir -p /root/.cache/vidgen_models

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg git

echo "Installing PyTorch with CUDA 13.0 support (nightly)..."
pip3 install --force-reinstall --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 --break-system-packages

echo "Installing Python dependencies..."
pip3 install -r /root/vidgen/requirements.txt --break-system-packages --ignore-installed

echo "Setting up systemd service..."
cp /root/vidgen/vidgen.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable vidgen

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Service commands:"
echo "  systemctl start vidgen    - Start vidgen"
echo "  systemctl stop vidgen     - Stop vidgen"
echo "  systemctl status vidgen   - Check status"
echo "  systemctl restart vidgen  - Restart vidgen"
echo "  journalctl -u vidgen -f   - Follow logs"
echo ""
echo "To run manually: python3 /root/vidgen/app.py"
echo "App accessible at: http://0.0.0.0:7860"
