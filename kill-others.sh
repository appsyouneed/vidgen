#!/bin/bash
for pid in $(pgrep -f "python3 app.py"); do
    kill -9 "$pid" 2>/dev/null
done
