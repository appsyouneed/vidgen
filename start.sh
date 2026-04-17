#!/bin/bash
echo "=== Installing and starting vidgen service ==="
cp /root/vidgen/vidgen.service /etc/systemd/system/vidgen.service
systemctl daemon-reload
systemctl enable vidgen
systemctl start vidgen
systemctl status vidgen --no-pager
