#!/bin/bash
sudo systemctl stop vidgen.service
sudo systemctl disable vidgen.service
sudo cp /root/image-editor/image-editor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable image-editor.service
sudo systemctl start image-editor.service
echo "Switched from vidgen to image-editor!"
echo "Check status: sudo systemctl status image-editor"
