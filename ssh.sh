#!/bin/bash
ssh-keygen -f '/root/.ssh/known_hosts' -R '194.93.48.222'
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=3 root@194.93.48.222
echo ""
echo "To connect manually, run:"
echo "ssh root@194.93.48.222"
