#!/bin/bash
set -e

echo "=== Switching from image-editor to vidgen ==="

# Stop and disable image-editor
sudo systemctl stop image-editor.service 2>/dev/null || true
sudo systemctl disable image-editor.service 2>/dev/null || true

# Kill any remaining Python processes
sudo pkill -9 python3 2>/dev/null || true
sudo pkill -9 python 2>/dev/null || true
sleep 2

# Copy vidgen service file
sudo cp /root/vidgen/vidgen.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start vidgen
sudo systemctl enable vidgen.service
sudo systemctl start vidgen.service

echo "✓ Switched to vidgen!"
echo ""
echo "Check status: sudo systemctl status vidgen"
echo "View logs: sudo journalctl -u vidgen -f"
