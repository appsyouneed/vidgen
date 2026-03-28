#!/bin/bash
cp vidgen.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable vidgen
systemctl start vidgen
