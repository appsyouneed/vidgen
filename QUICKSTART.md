# Quick Start Guide

## 1. Install Dependencies
```bash
./setup.sh
```

## 2. Run the Application

### Option A: Direct Run
```bash
python app.py
```

### Option B: Run as System Service
```bash
# Copy service file
sudo cp wan-vidgen.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable wan-vidgen
sudo systemctl start wan-vidgen

# Check status
sudo systemctl status wan-vidgen

# View logs
sudo journalctl -u wan-vidgen -f
```

## 3. Access the Interface

Open your browser and navigate to:
```
http://your-vps-ip:7860
```

## 4. First Run

On first run, the application will:
1. Download RIFE model (~200MB) - takes 1-2 minutes
2. Download Wan 2.2 14B model (~28GB) - takes 10-30 minutes depending on connection
3. Load models into GPU memory - takes 2-3 minutes

Total first-run setup: ~15-35 minutes

Subsequent runs will be much faster (30-60 seconds) as models are cached.

## 5. Usage

1. Upload an input image
2. Enter a text prompt (e.g., "make this image come alive, cinematic motion")
3. Adjust settings if desired:
   - Duration: 0.5-10 seconds
   - FPS: 16, 32, 64, or 128 (higher = smoother but slower)
   - Steps: 4-8 recommended for speed
   - Quality: 5-7 recommended
4. Click "Generate Video"

## Firewall Configuration

If you can't access the interface, ensure port 7860 is open:

```bash
# UFW
sudo ufw allow 7860/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 7860 -j ACCEPT
```

## Performance Tips

- First generation will be slower as models compile
- Typical generation time: 30-120 seconds depending on settings
- Higher resolution/duration = longer generation time
- Frame interpolation adds 10-30 seconds

## Stopping the Service

```bash
sudo systemctl stop wan-vidgen
```

## Updating

To update the code:
```bash
# Stop service if running
sudo systemctl stop wan-vidgen

# Update files
# (copy new files here)

# Restart service
sudo systemctl start wan-vidgen
```
