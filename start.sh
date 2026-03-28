#!/bin/bash
set -e

echo "=== Stopping any existing Python processes ==="
pkill -9 python3 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

echo "=== Starting Wan 2.2 Video Generation ==="
export HF_HOME=/root/.cache/huggingface
export PYTHONUNBUFFERED=1
cd /root/vidgen
python3 app.py
