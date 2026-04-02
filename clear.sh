#!/bin/bash
rm -rf /root/vidgen/__pycache__
rm -rf /root/vidgen/*.pyc
rm -rf /root/.cache/torch
find /root/vidgen -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -f "python3.*app.py"
echo "Cache cleared and old processes killed"
