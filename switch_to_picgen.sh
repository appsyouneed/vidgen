#!/bin/bash
sudo systemctl stop vidgen.service
sudo systemctl disable vidgen.service
sudo cp /root/picgen/picgen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picgen.service
sudo systemctl start picgen.service
echo "Switched from vidgen to picgen!"
echo "Check status: sudo systemctl status picgen"
