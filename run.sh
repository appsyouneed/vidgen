#!/bin/bash

echo "=== Starting vidgen service ==="
systemctl start vidgen
systemctl status vidgen --no-pager

echo ""
echo "Follow logs with:"
echo "  tail -f /root/vidgen/vidgen.log"
