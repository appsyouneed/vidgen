#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

cp "$SCRIPT_DIR/vidgen.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vidgen.service
systemctl restart vidgen.service

echo "✓ vidgen service started!"
echo ""
echo "Check status: systemctl status vidgen"
echo "View logs: tail -f $SCRIPT_DIR/vidgen.log"
