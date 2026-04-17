#!/bin/bash

echo "=== Starting vidgen service ==="
systemctl start vidgen
systemctl status vidgen --no-pager
