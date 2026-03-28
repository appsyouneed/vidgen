# Wan 2.2 14B Fast Preview - VPS Deployment

Video generation from images using Wan 2.2 14B model with FP8 quantization on Ubuntu VPS with Blackwell 6000 GPU.

## System Requirements

- Ubuntu VPS with NVIDIA Blackwell 6000 GPU
- CUDA 12.4+ drivers installed
- Python 3.10+
- At least 32GB RAM
- At least 100GB free disk space for models

## Installation

1. Clone or upload this directory to your VPS

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Start the application:
```bash
python app.py
```

The application will be accessible at `http://your-vps-ip:7860`

## Models and Files

### Main Model
- **Wan 2.2 14B I2V**: Automatically downloaded from HuggingFace
  - Model ID: Determined by `REPO_ID` env var or auto-selected from `TestOrganizationPleaseIgnore` organization
  - Default: Latest compatible model with `diffusers:WanImageToVideoPipeline`
  - Cache location: `~/.cache/huggingface/hub/`
  - Size: ~28GB (bfloat16)

### Frame Interpolation Model
- **RIFE v4.26**: Downloaded on first run
  - URL: `https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip`
  - Location: `./RIFEv4.26_0921.zip` (extracted to `./train_log/`)
  - Size: ~200MB
  - Used for frame interpolation (2x, 4x, 8x)

### Optimizations Applied
- **FP8 Quantization**: Float8 dynamic activation + Float8 weights on transformers
- **Int8 Quantization**: Int8 weights on text encoder
- **VAE Tiling**: Enabled for memory efficiency
- **VAE Slicing**: Enabled for memory efficiency

### LoRA Models
- Currently no LoRAs configured (empty `LORA_MODELS` list)
- Can be added by editing the `LORA_MODELS` array in `app.py`

## Configuration

### Model Settings
- **Resolution**: 480-832px (auto-adjusted to 16px multiples)
- **FPS**: 16 (base), up to 128 with interpolation
- **Frame Range**: 8-160 frames
- **Duration**: 0.5-10 seconds

### Environment Variables
- `REPO_ID`: Override the model repository ID
- `TOKENIZERS_PARALLELISM`: Set to "true" for parallel tokenization

### Schedulers Available
- FlowMatchEulerDiscrete
- SASolver
- DEISMultistep
- DPMSolverMultistepInverse
- UniPCMultistep (default)
- DPMSolverMultistep
- DPMSolverSinglestep

## Persistence

All downloaded models and files are cached and will persist between runs:

1. **HuggingFace Models**: `~/.cache/huggingface/`
2. **RIFE Model**: `./RIFEv4.26_0921.zip` and `./train_log/`
3. **Generated Videos**: Temporary files (auto-cleaned by Gradio)

The cache deletion code has been disabled to ensure models persist.

## Features

- Image-to-video generation with text prompts
- Optional last frame specification for controlled endings
- Frame interpolation (2x, 4x, 8x) using RIFE
- Frame extraction from generated videos
- Multiple scheduler options
- Adjustable guidance scales for high/low noise stages
- Quality control (1-10)
- Seed control for reproducibility

## GPU Memory Usage

Approximate VRAM usage:
- Model loading: ~16GB
- Inference (832x624, 81 frames): ~20-24GB
- With interpolation: +2-4GB

The Blackwell 6000 GPU should handle this comfortably.

## Troubleshooting

If you encounter CUDA out of memory errors:
1. Reduce resolution (use smaller input images)
2. Reduce duration (fewer frames)
3. Reduce frame multiplier
4. Lower quality setting

## Differences from HuggingFace Spaces Version

- Removed `@spaces.GPU` decorator (not needed for dedicated GPU)
- Removed ZeroGPU AOTI compilation
- Enabled VAE slicing and tiling (commented out in HF version)
- Changed server binding to `0.0.0.0:7860` for network access
- Removed `mcp_server` parameter from Gradio launch
- Models persist in cache (not deleted after use)
