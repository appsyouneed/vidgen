# Vidgen Setup Complete

## Files Created

1. **vidgen.service** - Systemd service for auto-start
2. **commands.txt** - Quick reference for all commands
3. **/root/switch_to_vidgen.sh** - Switch from image-editor to vidgen
4. **start.sh** - Manual start script (updated with cache persistence)

## SSH Broken Pipe Fix

Updated `/root/ssh.sh` with keepalive options:
- ServerAliveInterval=60 (sends keepalive every 60 seconds)
- ServerAliveCountMax=3 (disconnects after 3 failed keepalives)

## Model Persistence

Models are cached in `/root/.cache/huggingface/` and will NOT be re-downloaded:
- Set via `HF_HOME` environment variable
- Configured in both start.sh and vidgen.service
- First download: ~30GB, takes 10-30 minutes
- Subsequent runs: instant (loads from cache)

## Quick Start

### Switch to Vidgen (from image-editor)
```bash
bash /root/switch_to_vidgen.sh
```

### Switch to Image Editor (from vidgen)
```bash
bash /root/switch_to_image_editor.sh
```

### Manual Control
```bash
# Start
sudo systemctl start vidgen

# Stop
sudo systemctl stop vidgen

# Status
sudo systemctl status vidgen

# Logs
sudo journalctl -u vidgen -f
```

## Access

Once running, access at: **http://194.93.48.12:7860**

## Notes

- Only ONE service can run at a time (image-editor OR vidgen)
- Switch scripts automatically stop the other service
- Models persist between switches
- Service auto-starts on boot (whichever was enabled last)
