# Migration Summary: HuggingFace Spaces → Ubuntu VPS

## Changes Made

### 1. Removed HuggingFace Spaces Dependencies
- ❌ Removed `import spaces`
- ❌ Removed `@spaces.GPU` decorator
- ❌ Removed `IS_ZERO_GPU` environment variable checks
- ❌ Removed `aoti.py` file (ZeroGPU AOTI compilation)
- ❌ Removed `aoti.aoti_blocks_load()` calls
- ❌ Removed `mcp_server=True` from Gradio launch

### 2. Enabled GPU Optimizations
- ✅ Enabled `pipe.vae.enable_slicing()` (was commented out)
- ✅ Enabled `pipe.vae.enable_tiling()` (was commented out)
- ✅ Kept FP8 quantization on transformers
- ✅ Kept Int8 quantization on text encoder

### 3. Network Configuration
- ✅ Changed Gradio launch to bind to `0.0.0.0:7860` for external access
- ✅ Removed HuggingFace Spaces-specific launch parameters

### 4. Cache Management
- ✅ Cache deletion code already commented out (models will persist)
- ✅ Models cached in `~/.cache/huggingface/`
- ✅ RIFE model cached in `./RIFEv4.26_0921.zip` and `./train_log/`

### 5. Updated Dependencies
- ✅ Removed spaces-specific packages
- ✅ Added explicit `gradio` dependency
- ✅ Kept all model-related dependencies unchanged

## What Stayed the Same

### Models
- ✅ Wan 2.2 14B I2V model (same model ID logic)
- ✅ RIFE v4.26 for frame interpolation
- ✅ Same quantization settings (FP8/Int8)
- ✅ Same LoRA support (currently empty list)

### Settings
- ✅ Resolution: 480-832px
- ✅ FPS: 16 base, up to 128 with interpolation
- ✅ Frame range: 8-160 frames
- ✅ Duration: 0.5-10 seconds
- ✅ All schedulers available
- ✅ Guidance scales
- ✅ Quality settings (1-10)

### Features
- ✅ Image-to-video generation
- ✅ Optional last frame input
- ✅ Frame extraction from generated videos
- ✅ Frame interpolation (2x, 4x, 8x)
- ✅ Seed control
- ✅ All UI components and interactions

### Processing Pipeline
- ✅ Image resizing logic
- ✅ Frame generation
- ✅ RIFE interpolation
- ✅ Video export
- ✅ Progress tracking

## Files Modified

1. **app.py** - Main application file
   - Removed ZeroGPU imports and decorators
   - Enabled VAE optimizations
   - Updated network binding

2. **requirements.txt** - Python dependencies
   - Removed spaces-specific packages
   - Added explicit gradio

3. **aoti.py** - DELETED (ZeroGPU-specific)

## Files Added

1. **setup.sh** - Installation script
2. **README_VPS.md** - VPS deployment documentation
3. **QUICKSTART.md** - Quick start guide
4. **wan-vidgen.service** - Systemd service file
5. **MIGRATION_SUMMARY.md** - This file

## Verification Checklist

- [x] Python syntax valid
- [x] No spaces imports remaining
- [x] No @spaces decorators remaining
- [x] No ZeroGPU-specific code remaining
- [x] Cache persistence enabled
- [x] Network binding configured for VPS
- [x] All model downloads will persist
- [x] VAE optimizations enabled
- [x] All features preserved

## Models and Files Used

### Downloaded on First Run

1. **Wan 2.2 14B Model** (~28GB)
   - Source: HuggingFace Hub
   - Location: `~/.cache/huggingface/hub/models--*/`
   - Format: Diffusers pipeline (bfloat16)
   - Components:
     - Text encoder (Int8 quantized)
     - Transformer (FP8 quantized)
     - Transformer 2 (FP8 quantized)
     - VAE (with slicing and tiling)
     - Scheduler (configurable)

2. **RIFE v4.26** (~200MB)
   - Source: https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
   - Location: `./RIFEv4.26_0921.zip` (extracted to `./train_log/`)
   - Purpose: Frame interpolation
   - Model: RIFE_HDv3

### Included in Repository

1. **Sample Images**
   - kill_bill.jpeg
   - wan22_input_2.jpg
   - wan_i2v_input.JPG

2. **RIFE Model Code**
   - model/RIFE_HDv3.py (will be extracted from zip)
   - model/warplayer.py
   - model/loss.py
   - model/pytorch_msssim/

### Generated at Runtime

1. **Temporary Video Files**
   - Created in system temp directory
   - Auto-cleaned by Gradio
   - Format: MP4 (H.264)

## Expected Behavior

### First Run
1. Downloads RIFE model (1-2 min)
2. Downloads Wan 2.2 model (10-30 min)
3. Loads models to GPU (2-3 min)
4. Ready for inference

### Subsequent Runs
1. Loads cached models (30-60 sec)
2. Ready for inference

### Inference
- 832x624, 81 frames, 4 steps: ~30-60 seconds
- Higher resolution/duration: 60-120 seconds
- With 4x interpolation: +20-40 seconds

## GPU Memory Usage

- Model loading: ~16GB VRAM
- Inference peak: ~20-24GB VRAM
- With interpolation: +2-4GB VRAM
- Total: ~26GB VRAM (well within Blackwell 6000 capacity)

## Success Criteria

✅ Application starts without errors
✅ Models download and cache correctly
✅ GPU is detected and used
✅ Video generation works
✅ Frame interpolation works
✅ Frame extraction works
✅ All UI features functional
✅ Models persist between runs
✅ Accessible from network
