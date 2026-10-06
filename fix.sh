#!/usr/bin/env bash
# ============================================================================
#  fix.sh  —  Post-setup repair for the 16 GB-RAM / shared-Blackwell (sm_120,
#  cu130 torch) VPS where the stock setup.sh assumptions don't hold.
#
#  NOTE (kept for reference; normally you should NOT need to run this anymore):
#  Every fix below is now folded into the app / setup so a fresh install is
#  correct out of the box:
#    - item 1 (torchvision/torchaudio missing): setup.sh installs+ABI-checks
#      them, and its no-system-torch branch installs them into the venv.
#    - item 2 (cudaMallocAsync): app.py no longer sets that allocator backend.
#    - item 3 (low-RAM GPU-direct load): app.py auto-detects low free RAM and
#      streams the NSFW checkpoint tensor-by-tensor (mmap) + loads the base
#      model straight to GPU, via _load_and_merge_nsfw / _available_ram_gb.
#    - item 4 (VAE tiling): app.py's _apply_qwen_vae_memory_policy always tiles.
#    - item 5 (FA3): app.py only enables FA3 if a compatible `kernels` is
#      already importable, else uses SDPA — no hang, no crash.
#  This script is retained as a manual escape hatch only.
#
#  Run this AFTER setup.sh if picgen crashes on startup or on the VAE decode.
#  It is idempotent — safe to re-run; each step checks before changing.
#
#  Usage:
#      cd /root/newgen
#      bash fix.sh              # apply all fixes, then start picgen-only
#      bash fix.sh --no-start   # apply fixes only, don't launch
#
#  What it fixes (everything we worked out during bring-up):
#    1. torchvision/torchaudio missing in the venv (no system torch to symlink)
#    2. cudaMallocAsync allocator backend -> CUDA illegal memory access on sm_120
#    3. Qwen picgen load OOMing 16 GB RAM -> load straight to GPU, low CPU RAM
#    4. VAE full-frame decode -> conv3d CUDA illegal memory access -> force tiling
#    5. FlashAttention-3 kernel fetch hang -> disable FA3 (SDPA fallback)
# ============================================================================
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

APP="$APP_DIR/app.py"
APP_VENV="$APP_DIR/.app-venv"
APP_PY="$APP_VENV/bin/python"
APP_PIP="$APP_VENV/bin/pip"

START_APP=1
[ "${1:-}" = "--no-start" ] && START_APP=0

echo "=== NewGen fix.sh — repairing environment in $APP_DIR ==="

[ -f "$APP" ]     || { echo "ERROR: $APP not found. Run from /root/newgen."; exit 1; }
[ -x "$APP_PY" ]  || { echo "ERROR: venv python not found at $APP_PY. Run setup.sh first."; exit 1; }

# Back up app.py once per run so any patch is reversible.
BACKUP="$APP.fixbak.$(date +%s)"
cp "$APP" "$BACKUP"
echo "[fix] Backed up app.py -> $BACKUP"

# ---------------------------------------------------------------------------
# 0. Stop any running instance first (frees the GPU / avoids double-load OOM).
# ---------------------------------------------------------------------------
echo "[fix] Stopping any running app instance..."
bash "$APP_DIR/run.sh" stop 2>/dev/null || true
pkill -f "app\.py" 2>/dev/null || true
sleep 2

# ---------------------------------------------------------------------------
# 1. Ensure torchvision + torchaudio are really installed in the venv and
#    match the venv's torch build. This box has NO system torch to symlink, so
#    setup.sh's eviction leaves torchvision missing -> transformers'
#    AutoVideoProcessor import crashes the Qwen pipeline load.
# ---------------------------------------------------------------------------
echo "[fix] Checking torch / torchvision in the venv..."
TORCH_VER="$("$APP_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
echo "[fix]   torch = ${TORCH_VER:-<not importable>}"

# Pick the wheel index from torch's build tag (cu130 / cu128 / cpu fallback).
CU_TAG="$(printf '%s' "$TORCH_VER" | sed -n 's/.*+\(cu[0-9]\+\).*/\1/p')"
[ -z "$CU_TAG" ] && CU_TAG="cu130"
INDEX_URL="https://download.pytorch.org/whl/${CU_TAG}"

if "$APP_PY" -c 'import torchvision, torchaudio' 2>/dev/null; then
    echo "[fix]   torchvision + torchaudio already importable — skipping install."
else
    echo "[fix]   Installing torchvision + torchaudio from $INDEX_URL (--no-deps so torch is not disturbed)..."
    "$APP_PIP" install --no-cache-dir --no-deps torchvision torchaudio --index-url "$INDEX_URL" || \
        echo "[fix]   WARNING: cu-tagged install failed; you may need to match versions manually."
fi

# Verify the ABI actually works (torchvision C-ext linked against this torch)
# and that the exact import that crashed before now succeeds.
if "$APP_PY" - <<'PY'
import sys
try:
    import torch, torchvision
    torchvision.ops.nms                      # C-extension ABI check
    from transformers import AutoVideoProcessor  # this is what crashed before
    print("[fix]   OK: torchvision + AutoVideoProcessor import cleanly")
except Exception as e:
    print(f"[fix]   ERROR: torchvision/transformers still broken: {e}")
    sys.exit(1)
PY
then :; else
    echo "[fix]   torchvision still broken — trying a matched (torch+vision+audio) reinstall on cu128 (Blackwell)..."
    "$APP_PIP" install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 || true
fi

# ---------------------------------------------------------------------------
# 2. Remove the cudaMallocAsync allocator backend.
#    On sm_120 (Blackwell) + cu130 torch, backend:cudaMallocAsync produces
#    "CUDA error: an illegal memory access" during model load. Keep
#    expandable_segments:True (helps fragmentation) but drop the async backend.
# ---------------------------------------------------------------------------
echo "[fix] Removing cudaMallocAsync allocator backend..."
if grep -q "backend:cudaMallocAsync" "$APP"; then
    sed -i 's/expandable_segments:True,backend:cudaMallocAsync/expandable_segments:True/' "$APP"
    echo "[fix]   PYTORCH_CUDA_ALLOC_CONF -> expandable_segments:True"
else
    echo "[fix]   already removed."
fi

# ---------------------------------------------------------------------------
# 3. Rewrite the PICGEN load block: load Qwen straight onto the GPU with low
#    CPU RAM. Base model via device_map="cuda"; NSFW weights applied
#    tensor-by-tensor by mmap'ing the safetensors and copying each onto the GPU
#    parameter in place (never holds the whole ~15 GB dict in 16 GB RAM);
#    fp8 cast + VAE policy + optimize on GPU. Idempotent: only patches if the
#    old CPU-load block is still present.
# ---------------------------------------------------------------------------
echo "[fix] Ensuring picgen loads directly onto the GPU (low CPU RAM)..."
"$APP_PY" <<'PYEOF'
p = "app.py"
s = open(p, encoding="utf-8").read()

SENTINEL = "PICGEN MODE: Loading Qwen straight onto the GPU (low CPU RAM)"
if SENTINEL in s:
    print("[fix]   picgen GPU-direct load already applied — skipping.")
    raise SystemExit(0)

start = s.find('    print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")')
if start == -1:
    print("[fix]   WARNING: picgen load anchor not found — leaving as-is "
          "(maybe already customized). Check manually.")
    raise SystemExit(0)
end = s.find('_active_model = "pic"', start)
if end == -1:
    print("[fix]   WARNING: end anchor _active_model=\"pic\" not found — leaving as-is.")
    raise SystemExit(0)
end = s.find("\n", end) + 1

block = '''    print(" PICGEN MODE: Loading Qwen straight onto the GPU (low CPU RAM)...")
    start_qwen = time.time()
    import gc as _gc
    from safetensors import safe_open as _safe_open

    _model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
    if not os.path.exists(_model_index_path):
        print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            cache_dir=BASE_MODEL_LOCAL_PATH,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
    else:
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )

    print("Loading NSFW weights for Qwen...")
    if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
        print("Downloading NSFW weights...")
        os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
        v23_path = hf_hub_download(
            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
            cache_dir=PICGEN_MODELS_DIR,
            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
        )
    else:
        v23_path = NSFW_WEIGHTS_LOCAL_PATH

    # Apply NSFW weights WITHOUT loading the whole file into CPU RAM: mmap the
    # safetensors and copy each tensor directly onto the GPU parameter in place.
    print("Applying NSFW weights tensor-by-tensor onto GPU (low RAM)...")
    def _strip(k):
        for pre in ("model.diffusion_model.", "transformer."):
            if k.startswith(pre):
                return "transformer", k[len(pre):]
        for pre in ("first_stage_model.", "vae."):
            if k.startswith(pre):
                return "vae", k[len(pre):]
        if "conditioner.embedders.0." in k:
            return "text_encoder", k.split("conditioner.embedders.0.", 1)[1]
        if "text_encoder." in k:
            return "text_encoder", k.split("text_encoder.", 1)[1]
        return None, None

    _sub = {"transformer": pic_pipe.transformer, "vae": pic_pipe.vae,
            "text_encoder": pic_pipe.text_encoder}
    _sd = {name: dict(m.named_parameters()) for name, m in _sub.items()}
    _sd_buf = {name: dict(m.named_buffers()) for name, m in _sub.items()}
    _applied = 0
    with _safe_open(v23_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            comp, sub_key = _strip(key)
            if comp is None:
                continue
            tgt = _sd[comp].get(sub_key)
            if tgt is None:
                tgt = _sd_buf[comp].get(sub_key)
            if tgt is None:
                continue
            try:
                val = f.get_tensor(key).to(tgt.device, tgt.dtype, non_blocking=True)
                with torch.no_grad():
                    tgt.copy_(val)
                del val
                _applied += 1
            except Exception as _e:
                print(f"  [nsfw] skip {key}: {_e}")
    print(f"  NSFW weights applied: {_applied} tensors.")
    _gc.collect()
    torch.cuda.synchronize()

    _cast_transformer_to_fp8(pic_pipe)
    _gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    _apply_qwen_vae_memory_policy(pic_pipe, PIC_DEVICE)
    _optimize_qwen_pipe(pic_pipe, PIC_DEVICE)

    qwen_time = time.time() - start_qwen
    print(f" QWEN READY on GPU in {qwen_time:.1f}s - Picgen functional!")
    _active_model = "pic"
'''

s = s[:start] + block + s[end:]
open(p, "w", encoding="utf-8").write(s)
print("[fix]   picgen load block rewritten for GPU-direct low-RAM load.")
PYEOF

# ---------------------------------------------------------------------------
# 4. Force VAE tiling/slicing ALWAYS ON. The >=40 GB "full-frame decode"
#    optimization runs the VAE decoder's conv3d over the whole latent in one
#    shot, which triggers a CUDA illegal memory access on this cu130/Blackwell
#    box. Neuter _apply_qwen_vae_memory_policy so it always enables tiling.
# ---------------------------------------------------------------------------
echo "[fix] Forcing VAE tiling/slicing always ON (safe conv3d decode)..."
"$APP_PY" <<'PYEOF'
p = "app.py"
s = open(p, encoding="utf-8").read()

if "tiling+slicing ON (safe decode)" in s:
    print("[fix]   VAE policy already forced to tiling — skipping.")
    raise SystemExit(0)

marker = "def _apply_qwen_vae_memory_policy(pipe, device):"
i = s.find(marker)
if i == -1:
    print("[fix]   WARNING: VAE policy helper not found — skipping.")
    raise SystemExit(0)
j = s.find("\ndef ", i + 1)          # start of the next top-level def
if j == -1:
    print("[fix]   WARNING: could not find end of VAE policy helper — skipping.")
    raise SystemExit(0)

newfn = (
    'def _apply_qwen_vae_memory_policy(pipe, device):\n'
    '    """Always keep VAE tiling+slicing ON (safe on this cu130/Blackwell box).\n'
    '    Full-frame decode triggered a conv3d CUDA illegal-memory-access, so we\n'
    '    never disable tiling regardless of VRAM. Never raises."""\n'
    '    try:\n'
    '        vae = getattr(pipe, "vae", None)\n'
    '        if vae is None:\n'
    '            return\n'
    '        try:\n'
    '            vae.enable_tiling()\n'
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    '            vae.enable_slicing()\n'
    '        except Exception:\n'
    '            pass\n'
    '        print("    [vae] tiling+slicing ON (safe decode).")\n'
    '    except Exception as _e:\n'
    '        print(f"    [vae] policy skipped (non-fatal): {_e}")\n'
)

s = s[:i] + newfn + s[j + 1:]
open(p, "w", encoding="utf-8").write(s)
print("[fix]   VAE policy forced to tiling+slicing ON.")
PYEOF

# ---------------------------------------------------------------------------
# Syntax check before we try to launch.
# ---------------------------------------------------------------------------
echo "[fix] Syntax-checking patched app.py..."
if "$APP_PY" -c "import py_compile; py_compile.compile('app.py', doraise=True)"; then
    echo "[fix]   syntax OK."
else
    echo "[fix]   SYNTAX ERROR after patching. Restoring backup $BACKUP"
    cp "$BACKUP" "$APP"
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Runtime env for launch: disable FlashAttention-3 (its kernel fetch can
#    hang on a fresh box; SDPA is the exact, safe fallback) and pin the stable
#    allocator. These are exported so the backgrounded app inherits them.
# ---------------------------------------------------------------------------
export NEWGEN_QWEN_FA3=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== fix.sh complete ==="
echo "  Applied: torchvision repair, allocator fix, GPU-direct picgen load,"
echo "           VAE tiling, FA3 disabled."
echo "  Backup of pre-fix app.py: $BACKUP"
echo ""

if [ "$START_APP" = "1" ]; then
    echo "[fix] Launching picgen-only (video tab disabled)..."
    exec bash "$APP_DIR/run.sh" -picgen -novidgen
else
    echo "[fix] --no-start given. To launch:"
    echo "      export NEWGEN_QWEN_FA3=0"
    echo "      bash run.sh -picgen -novidgen"
fi
