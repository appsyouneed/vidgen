#!/bin/bash
set -e

echo "=== Fixing RIFE Model Installation ==="

cd /root/vidgen

echo "Removing old RIFE files..."
rm -rf train_log __MACOSX RIFEv4.26_0921.zip model

echo "Cloning RIFE architecture (model/ goes in working dir)..."
git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
cp -r /tmp/rife/model .
rm -rf /tmp/rife

echo "Downloading RIFE weights into train_log/..."
wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
unzip -o RIFEv4.26_0921.zip
rm -rf RIFEv4.26_0921.zip __MACOSX

echo "=== Fix Complete ==="
echo "Run: python3 app.py"
