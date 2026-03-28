#!/bin/bash
set -e

echo "=== Wan 2.2 14B VPS Setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y ffmpeg

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt

echo "=== Setup Complete ==="
echo "Models will be cached in: ~/.cache/huggingface/"
echo "RIFE model will be downloaded to: ./RIFEv4.26_0921.zip"
echo ""
echo "To run the application:"
echo "  python app.py"
echo ""
echo "The app will be accessible at: http://0.0.0.0:7860"
