#!/bin/bash

echo "=== Stopping vidgen service ==="
systemctl stop vidgen 2>/dev/null || true

echo "Killing any running app.py processes..."
pkill -f "python3 app.py" 2>/dev/null || true
pkill -f "python app.py" 2>/dev/null || true

echo "Freeing port 7860..."
lsof -ti:7860 | xargs kill -9 2>/dev/null || true

echo "Vidgen stopped."
