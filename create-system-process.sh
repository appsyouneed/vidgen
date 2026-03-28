#!/bin/bash
sudo cp vidgen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vidgen
sudo systemctl start vidgen
