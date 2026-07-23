@echo off
REM Windows batch version of ssh.sh
REM Remove known host entry (Windows equivalent)
ssh-keygen -f "%USERPROFILE%\.ssh\known_hosts" -R "194.93.48.12" 2>nul
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=3 root@194.93.48.12
echo.
echo To connect manually, run:
echo ssh root@194.93.48.12
pause