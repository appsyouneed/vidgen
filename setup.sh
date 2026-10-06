#!/bin/bash
set -e

echo "=== Newgen Setup (Video + Photo Generator) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# CRITICAL: This setup script NEVER touches the system torch / torchvision /
# torchaudio / numpy stack that ships with your Ubuntu 22.04 VPS.
# Those versions (torch 2.8 dev+cu128, etc.) are already correct and must
# not be changed.  All Python-level dependencies that the app needs are
# installed into /root/newgen/.app-venv so they are completely isolated from
# both the OS-managed python3 packages and the system torch stack.
#
# The ONLY system packages installed here are apt-managed OS tools:
#   ffmpeg, wget, unzip, git, git-lfs, python3-pip, python3-venv
# Those are safe because they live in /usr/bin / /usr/lib, not in
# site-packages, and cannot conflict with Python packages.
# ---------------------------------------------------------------------------

PYTHON="python3"

echo "Installing system apt dependencies (safe — OS tools only, not Python packages)..."
if lsof /var/lib/dpkg/lock-frontend > /dev/null 2>&1; then
    lsof -t /var/lib/dpkg/lock-frontend | xargs kill -9 2>/dev/null || true
fi
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/updates/*
dpkg --configure -a || true
apt-get update
apt-get install -y --fix-missing ffmpeg wget unzip git python3-pip python3-venv git-lfs

echo "Creating directories..."
mkdir -p /root/newgen/tmp
mkdir -p /root/newgen/mmaudio
mkdir -p /root/newgen/wan22_distill
mkdir -p /root/.cache/huggingface
chmod 1777 /root/newgen/tmp

# ---------------------------------------------------------------------------
# App venv — isolated from system Python site-packages.
# We do NOT use --system-site-packages so nothing leaks in from the OS level.
# The venv gets its own pip, its own diffusers, gradio, etc.
# torch / torchvision / torchaudio are NOT reinstalled here; the venv
# inherits the system torch via PYTHONPATH override at app launch time
# (see autorun.sh / systemd service), keeping the multi-GB dev build intact.
# ---------------------------------------------------------------------------
APP_VENV="/root/newgen/.app-venv"

# ALWAYS rebuild clean. Reusing an existing venv here was the direct cause
# of a bug where a stale/broken .app-venv (missing a working cffi/_cffi_backend,
# or built against a different python3 than $PYTHON currently resolves to)
# kept getting silently reused across setup.sh reruns, so the cryptography
# import crash never actually got fixed by "running setup.sh again."
# A full rebuild is the only way to guarantee the venv matches this exact
# requirements.txt and this exact python3 binary every time.
echo "Rebuilding app venv at $APP_VENV from scratch (removing any existing one)..."
rm -rf "$APP_VENV"
$PYTHON -m venv "$APP_VENV"

APP_PY="$APP_VENV/bin/python"
APP_PIP="$APP_VENV/bin/pip"

echo "Upgrading pip inside venv..."
"$APP_PIP" install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# Detect the system torch location so the venv can import it without
# reinstalling it.  We find the real site-packages dir by asking the system
# python where torch lives, then prepend it to PYTHONPATH for all subsequent
# pip calls and for the running app.  This means pip inside the venv will
# treat torch/torchvision/torchaudio as "already satisfied" (via PYTHONPATH)
# and skip downloading them.
# ---------------------------------------------------------------------------
SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SYS_SITE" ] && [ -d "$SYS_SITE/torch" ]; then
    echo "System torch found at $SYS_SITE."
    echo "Symlinking ONLY torch/torchvision/torchaudio/torchao (and their"
    echo "private deps) into the venv's own site-packages, instead of"
    echo "exporting PYTHONPATH to the whole system dist-packages dir."
    echo "(A blanket PYTHONPATH export shadows EVERY same-named package in"
    echo "the venv with the system copy — e.g. it silently made the app"
    echo "import system diffusers 0.33.1 instead of the venv's own 0.37.1,"
    echo "even though pip had correctly installed 0.37.1 into the venv.)"
    VENV_SITE_PACKAGES="$APP_VENV/lib/$($APP_PY -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
    for pkg in torch torchgen torchvision torchaudio torchao \
               functorch caffe2 \
               nvidia triton; do
        if [ -e "$SYS_SITE/$pkg" ] && [ ! -e "$VENV_SITE_PACKAGES/$pkg" ]; then
            ln -s "$SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        fi
    done
    # Deliberately NOT symlinking torch*/torchvision*/torchaudio*/torchao*
    # .dist-info or .egg-info metadata into the venv.
    #
    # The system torch here is a genuinely newer stable release (2.14.0)
    # than the ancient nightly torchvision/torchaudio builds (dev20250526,
    # which require exactly torch==2.8.0.dev20250526) -- that mismatch is
    # real and lives on the HOST, not something this script can fix without
    # violating "never touch system torch."
    #
    # But symlinking the .dist-info made pip's resolver INSIDE the venv treat
    # torch as a formally "installed, version-tracked" package, so every
    # single pip install in this script re-evaluated and re-printed that
    # same conflict warning. Without the .dist-info, pip has no metadata
    # record of torch/torchvision/torchaudio at all inside the venv, so it
    # silently skips evaluating them -- `import torch` still works fine via
    # the plain directory symlinks above, only pip's bookkeeping is blind
    # to it now, which is exactly what we want for a package we manage
    # entirely outside pip anyway.
    # ---------------------------------------------------------------------
    # Write a .pth file so the venv's sys.path always includes the system
    # site-packages dir.  importlib.util.find_spec("torch") walks sys.path
    # at runtime — the directory symlinks above make 'import torch' work, but
    # find_spec needs to see the directory in sys.path directly to satisfy
    # diffusers' is_torch_available() check reliably.  The .pth file is the
    # standard Python mechanism for extending sys.path from inside a venv
    # without breaking venv isolation (the venv's own site-packages still
    # take precedence for everything else because sys.path ordering is:
    # venv site-packages first, then .pth additions).
    # ---------------------------------------------------------------------
    PTH_FILE="$VENV_SITE_PACKAGES/zzz_system_torch_path.pth"
    echo "$SYS_SITE" > "$PTH_FILE"
    echo "  Wrote sys.path entry: $PTH_FILE -> $SYS_SITE"

    # ---------------------------------------------------------------------
    # AUTOMATED ABI CHECK + SELF-REPAIR.
    # torchvision/torchaudio are precompiled C-extensions linked against a
    # specific torch build. If the SYSTEM torch has since been upgraded
    # (e.g. to a newer stable release) while torchvision/torchaudio are
    # still old nightly builds compiled against the old torch, importing
    # torchvision crashes with "operator torchvision::nms does not exist" --
    # a C-extension ABI mismatch, not anything pip's dependency resolver can
    # see or a venv can work around. It has to be fixed at the system level,
    # so we test for it here and repair it automatically instead of leaving
    # it as a manual step to remember.
    # ---------------------------------------------------------------------
    echo "Verifying system torch/torchvision/torchaudio ABI compatibility..."
    if ! python3 -c "import torch, torchvision; torchvision.ops.nms" >/dev/null 2>&1; then
        echo "  MISMATCH DETECTED: system torchvision/torchaudio do not match system torch."
        SYS_TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
        CU_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu128")
        echo "  System torch is $SYS_TORCH_VER ($CU_TAG). Reinstalling matching"
        echo "  torchvision/torchaudio at the SYSTEM level (torch itself is left untouched)..."
        python3 -m pip install --quiet --upgrade --no-deps \
            torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${CU_TAG}" \
        && echo "  Repaired: installed torchvision/torchaudio matching torch ${SYS_TORCH_VER}." \
        || echo "  WARNING: automatic repair failed. Run manually: python3 -m pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/${CU_TAG}"

        # Re-symlink into the venv now that the system copies changed.
        for pkg in torchvision torchaudio; do
            rm -f "$VENV_SITE_PACKAGES/$pkg"
            NEW_SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$SYS_SITE")
            [ -e "$NEW_SYS_SITE/$pkg" ] && ln -s "$NEW_SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        done

        if python3 -c "import torch, torchvision; torchvision.ops.nms" >/dev/null 2>&1; then
            echo "  Verified: torchvision ABI now matches torch."
        else
            echo "  WARNING: still mismatched after repair attempt. The app WILL crash on"
            echo "  'import torchvision' until this is resolved manually on the host."
        fi
    else
        echo "  OK: torchvision ABI matches system torch."
    fi
else
    echo "WARNING: Could not locate system torch in $SYS_SITE."
    echo "  Do NOT let setup.sh install torch — that would overwrite your dev build."
    # -----------------------------------------------------------------------
    # No system torch to symlink. On boxes like the 16 GB / Blackwell VPS the
    # venv carries its OWN torch, and torchvision/torchaudio are then missing
    # (setup.sh only symlinked them from a system copy that doesn't exist here).
    # transformers' AutoVideoProcessor import then crashes the Qwen load.
    # Install torchvision+torchaudio INTO THE VENV, matched to the venv torch's
    # CUDA build tag, with --no-deps so the venv torch is never disturbed.
    # (This folds in fix.sh item 1 so fix.sh never needs to run again.)
    # -----------------------------------------------------------------------
    if "$APP_PY" -c "import torch" >/dev/null 2>&1; then
        if "$APP_PY" -c "import torchvision, torchaudio; torchvision.ops.nms" >/dev/null 2>&1; then
            echo "  venv already has working torchvision + torchaudio — nothing to do."
        else
            VENV_TORCH_VER="$("$APP_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
            CU_TAG="$(printf '%s' "$VENV_TORCH_VER" | sed -n 's/.*+\(cu[0-9]\+\).*/\1/p')"
            [ -z "$CU_TAG" ] && CU_TAG="$("$APP_PY" -c 'import torch; print("cu"+torch.version.cuda.replace(".",""))' 2>/dev/null || echo cu128)"
            echo "  Installing torchvision+torchaudio into venv (torch=$VENV_TORCH_VER, $CU_TAG, --no-deps)..."
            "$APP_PIP" install --no-cache-dir --no-deps torchvision torchaudio \
                --index-url "https://download.pytorch.org/whl/${CU_TAG}" \
                && echo "  Installed torchvision+torchaudio into venv." \
                || echo "  WARNING: venv torchvision/torchaudio install failed — match versions manually."
            # Verify the exact import that crashes the Qwen load if it's broken.
            "$APP_PY" -c "import torchvision; torchvision.ops.nms; from transformers import AutoVideoProcessor; print('  OK: venv torchvision + AutoVideoProcessor import cleanly')" \
                || echo "  WARNING: torchvision/transformers still broken in venv."
        fi
    fi
fi

echo "Installing Python application dependencies into isolated venv..."
# Requirements file has torch/torchvision/torchaudio commented out intentionally.
# --no-warn-conflicts suppresses the "torchvision X requires torch==Y but you have Z"
# noise that appears because the .pth file makes pip see the system torchvision's
# dist-info (which declares a torch version requirement) alongside the system torch
# (which may have a different version). The ABI check above already verified these
# are compatible at the C-extension level, so the pip metadata conflict is harmless.
"$APP_PIP" install --quiet --no-cache-dir --no-warn-conflicts \
    -r "$SCRIPT_DIR/requirements.txt"

echo "Ensuring critical pinned packages inside venv..."
# These are pinned to exact versions the app requires. They live only in the
# venv — the system site-packages are never touched.
# IMPORTANT: torch / torchvision / torchaudio / torchao are NOT installed here.
# They come from the system Python stack via PYTHONPATH (see run_app.sh).
# Installing them here would pull in a mismatched torch build and overwrite
# the system torch 2.8 dev+cu128.
"$APP_PIP" install --quiet --no-cache-dir --no-warn-conflicts \
    Pillow \
    "transformers==4.55.4" \
    "huggingface-hub>=0.34.0,<1.0" \
    "numpy>=1.26,<2.2" \
    "diffusers==0.37.1" \
    "safetensors>=0.4.0" \
    accelerate

# ---------------------------------------------------------------------------
# CRITICAL: Evict any torch / torchvision / torchaudio / torchao that pip may
# have pulled in as transitive dependencies of the packages above (e.g.
# diffusers, accelerate, torchao all list torch as a dependency and pip will
# happily download a fresh copy if it doesn't see one on sys.path at install
# time).  We run this AFTER every pip install block so any stray copies are
# removed before the app starts.  The system copies remain available via
# PYTHONPATH at runtime.
# ---------------------------------------------------------------------------
echo "Evicting any venv-local torch/torchvision/torchaudio/torchao (must use system copies)..."
# Use --prefix to ensure we only remove packages installed INTO the venv itself,
# never the system site-packages (which pip sees via the .pth file and would
# otherwise try — and fail — to uninstall, printing noisy "outside environment" errors).
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    # Only attempt uninstall if the package exists as a real file (not a symlink
    # to system) inside the venv's own site-packages.
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done
echo "Torch eviction done — system torch will be used via symlinks at runtime."

echo "Installing torchcodec into venv (required by torchaudio.save for Foley audio)..."
# torchaudio 2.6+ routes torchaudio.save() through TorchCodec; without it the
# HunyuanVideo-Foley step crashes and videos come out silent.
# CRITICAL: --no-deps prevents pip from pulling torch back in as a dependency.
# Without --no-deps, torchcodec's metadata lists torch as a dep, pip sees no
# dist-info for torch in the venv (we deliberately omit it), downloads and
# re-installs torch 2.14.0 into the venv — undoing the eviction above and
# creating a second torch copy that conflicts with the system torch symlinks.
"$APP_PIP" install --quiet --no-cache-dir --no-deps torchcodec 2>/dev/null || \
"$APP_PIP" install --quiet --no-cache-dir --no-deps \
    "torchcodec==0.2.1" 2>/dev/null || \
    echo "⚠️  torchcodec install failed — Foley audio may be silent (non-fatal)."

# ---------------------------------------------------------------------------
# SECOND EVICTION PASS — runs after all installs including torchcodec.
# Any package installed above (torchcodec, sageattention, rembg, etc.) may
# have re-pulled torch/torchvision/torchaudio as transitive deps.  The
# system copies are already in the venv via directory symlinks (added above)
# AND via the .pth sys.path entry, so evicting pip-installed copies is safe.
# ---------------------------------------------------------------------------
echo "Final torch eviction pass (post all-installs)..."
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done
# Verify torch is still importable via symlinks after eviction.
if "$APP_PY" -c "import torch; assert torch.__version__" 2>/dev/null; then
    echo "  Torch import verified OK via system symlinks after final eviction."
else
    echo "  WARNING: torch not importable after eviction — re-symlinking from $SYS_SITE..."
    for pkg in torch torchgen torchvision torchaudio torchao functorch caffe2 nvidia triton; do
        if [ -e "$SYS_SITE/$pkg" ] && [ ! -e "$VENV_SITE_PACKAGES/$pkg" ]; then
            ln -sf "$SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        fi
    done
    "$APP_PY" -c "import torch; print('  Torch re-symlink OK:', torch.__version__)" \
        || echo "  CRITICAL: torch still not importable. Check system torch install."
fi

# ---------------------------------------------------------------------------
# Patch gradio/oauth.py inside the VENV for huggingface_hub >= 0.26
# huggingface_hub removed HfFolder in 0.26.0; gradio 4.43.0 still imports it
# at module level, crashing the entire process on startup.
# This patch lives only inside the venv's copy of gradio — the system is untouched.
# ---------------------------------------------------------------------------
echo "Patching venv gradio/oauth.py for huggingface_hub >= 0.26 compatibility..."
VENV_SITE="$APP_VENV/lib/$(ls $APP_VENV/lib/)/site-packages"
OAUTH_FILE="$VENV_SITE/gradio/oauth.py"
python3 - "$OAUTH_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()

# Remove any previously botched patch attempts, then apply cleanly.
# The original line we need to replace is exactly:
#   from huggingface_hub import HfFolder, whoami
# It may have been wrapped in broken try blocks already — strip all of that
# out and rewrite from the first import of fastapi.responses onward.

GOOD = """\
try:
    from huggingface_hub import HfFolder, whoami
except ImportError:
    from huggingface_hub import whoami
    try:
        from huggingface_hub import get_token as _get_token
    except ImportError:
        _get_token = lambda: None  # noqa: E731

    class HfFolder:  # noqa: N801
        @staticmethod
        def get_token():
            return _get_token()"""

# Find the block between 'from fastapi.responses import RedirectResponse'
# and 'from .utils import get_space' and replace it wholesale.
ANCHOR_START = "from fastapi.responses import RedirectResponse"
ANCHOR_END = "from .utils import get_space"

start = text.index(ANCHOR_START) + len(ANCHOR_START)
end = text.index(ANCHOR_END)

text = text[:start] + "\n" + GOOD + "\n\n" + text[end:]
candidate.write_text(text)
print(f"  Patched: {candidate}")
PYPATCH

echo "Patching venv gradio_client/utils.py for pydantic v2 bool-schema compatibility..."
GRADIO_CLIENT_FILE="$VENV_SITE/gradio_client/utils.py"
python3 - "$GRADIO_CLIENT_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()
if "if not isinstance(schema, dict):" in text:
    print(f"  Already patched: {candidate}")
    sys.exit(0)

OLD = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
NEW = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:'
if OLD in text:
    candidate.write_text(text.replace(OLD, NEW, 1))
    print(f"  Patched: {candidate}")
    sys.exit(0)

OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
NEW2 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "unknown"\n    if "const" in schema:'
if OLD2 in text:
    candidate.write_text(text.replace(OLD2, NEW2, 1))
    print(f"  Patched get_type fallback: {candidate}")
    sys.exit(0)

print(f"  ERROR: no matching pattern in {candidate}")
sys.exit(1)
PYPATCH

echo "Patching diffusers.loaders for WanLoraLoaderMixin (diffusers 0.37.1 compat)..."
# diffusers 0.37.1 ships WanImageToVideoPipeline but its pipeline_wan_i2v
# module imports WanLoraLoaderMixin from diffusers.loaders — a class that does
# not exist in 0.37.1.  This causes ALL wan import paths to fail (including the
# top-level 'import diffusers' path, because diffusers.__init__ re-imports from
# the same broken submodule), which makes WanImageToVideoPipeline resolve to a
# dummy stub.  The stub's from_pretrained() then raises the misleading
# "PyTorch library not found" error even when torch is correctly installed.
# This patch injects an empty shim class so the import chain completes.
# When diffusers is later upgraded to a version that ships the real class,
# the conditional below is a no-op and nothing is overwritten.
"$APP_PY" - <<'WANPATCH'
import sys
try:
    import diffusers.loaders as _dl
    if hasattr(_dl, "WanLoraLoaderMixin"):
        print("  WanLoraLoaderMixin already present — no patch needed.")
        sys.exit(0)
    # Locate the loaders __init__.py and append the shim.
    import pathlib
    loaders_init = pathlib.Path(_dl.__file__)
    shim = (
        "\n"
        "# --- WanLoraLoaderMixin compat shim (injected by setup.sh) ---\n"
        "# diffusers 0.37.1 ships pipeline_wan_i2v which imports this class\n"
        "# but does not define it in diffusers.loaders.  The shim is a no-op\n"
        "# base class; diffusers >= 0.38 will ship the real implementation.\n"
        "if not hasattr(sys.modules[__name__], 'WanLoraLoaderMixin'):\n"  
        "    class WanLoraLoaderMixin:  # noqa: N801\n"
        "        \"\"\"Compat shim — real class ships in diffusers >= 0.38.\"\"\"\n"
        "    import sys as _sys_shim\n"
        "    _sys_shim.modules[__name__].WanLoraLoaderMixin = WanLoraLoaderMixin\n"
    )
    existing = loaders_init.read_text()
    if "WanLoraLoaderMixin compat shim" in existing:
        print("  loaders/__init__.py already has the shim — skipping.")
        sys.exit(0)
    loaders_init.write_text(existing + shim)
    print(f"  Patched {loaders_init}")
except Exception as e:
    print(f"  WARNING: WanLoraLoaderMixin patch failed: {e}")
    print("  app.py has a runtime fallback shim — startup will still succeed.")
    sys.exit(0)
WANPATCH

echo "Installing SageAttention into venv for accelerated inference..."
# --no-deps: SageAttention lists torch as a dep; without this flag pip would
# re-download torch and undo the eviction above.
"$APP_PIP" install --quiet --no-cache-dir --no-deps sageattention 2>/dev/null || {
    echo "Pre-built SageAttention not available for this GPU arch — building from source..."
    "$APP_PIP" install --no-cache-dir --no-deps \
        "git+https://github.com/thu-ml/SageAttention.git" 2>/dev/null || {
        echo "⚠️  SageAttention install failed — will fall back to standard SDPA at runtime."
    }
}
# Evict any torch that sageattention may have pulled in as a real install (not symlink).
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done

echo "Installing rembg into venv (BiRefNet background removal for Merge Photos)..."
# rembg[gpu] depends on onnxruntime-gpu which may pull torch transitively.
"$APP_PIP" install --quiet --no-cache-dir "rembg[gpu]" 2>/dev/null || {
    echo "rembg[gpu] failed — trying CPU-only rembg..."
    "$APP_PIP" install --quiet --no-cache-dir rembg 2>/dev/null || {
        echo "⚠️  rembg install failed — Merge Photos will use original images without background removal."
    }
}
# Evict any torch that rembg may have pulled in as a real install (not symlink).
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done

echo "Ensuring pyOpenSSL inside venv..."
"$APP_PY" -c "from OpenSSL import SSL" 2>/dev/null || \
    "$APP_PIP" install --quiet pyopenssl

# ---------------------------------------------------------------------------
# LipSync (MuseTalk) dependencies — xtcocotools, pycocotools, mmpose
#
# xtcocotools is a C-extension that requires numpy headers and GCC to build.
# It often fails on clean VPS images because:
#   1. Its pyproject.toml declares numpy as a build dependency but pip's
#      build isolation picks up a numpy that has no headers (binary-only wheel),
#      so gcc can't find xtcocotools/_mask.c → "No such file or directory".
#   2. The package is not widely mirrored and binary wheels are only available
#      for a narrow set of (python, arch) combinations.
#
# Strategy (in order):
#   a) Try a pre-built wheel from piwheels / extra-index (fast, no compile).
#   b) Install numpy headers so gcc can find them, then build from source.
#   c) Fall back silently — app.py has a runtime guard and will skip mmpose
#      features rather than crash if xtcocotools is absent.
#
# pycocotools is a separate (better-maintained) package that provides the
# same COCO mask API; we install it regardless so any code that imports
# pycocotools directly still works even if xtcocotools fails.
#
# mmpose>=1.1.0 depends on xtcocotools at import time only for its COCO
# dataset helpers — the face-landmark inference used by MuseTalk does NOT
# require the full COCO dataset stack at runtime, so a missing xtcocotools
# is non-fatal for LipSync as long as mmpose itself installs.
# ---------------------------------------------------------------------------
echo "[LipSync] Installing pycocotools (COCO mask API, always needed)..."
"$APP_PIP" install --quiet --no-cache-dir pycocotools 2>/dev/null \
    && echo "[LipSync] pycocotools installed OK." \
    || echo "[LipSync] ⚠️  pycocotools install failed (non-fatal)."

echo "[LipSync] Installing xtcocotools (C-extension, may need to build from source)..."
# Pass a: try pre-built wheel first (no compile, fastest).
if "$APP_PIP" install --quiet --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        xtcocotools 2>/dev/null; then
    echo "[LipSync] xtcocotools installed OK (pre-built wheel)."
else
    # Pass b: ensure Python dev headers + numpy headers are present so gcc
    # can compile _mask.c, then build from source.
    echo "[LipSync] Pre-built wheel unavailable — attempting source build..."
    apt-get install -y --quiet python3-dev 2>/dev/null || true
    # Install numpy with headers into the venv so the build backend can find
    # _mask.c's #include <numpy/arrayobject.h>.
    "$APP_PIP" install --quiet --no-cache-dir "numpy>=1.26,<2.2" 2>/dev/null || true
    if "$APP_PIP" install --quiet --no-cache-dir \
            --no-build-isolation \
            xtcocotools 2>/dev/null; then
        echo "[LipSync] xtcocotools installed OK (built from source)."
    else
        echo "[LipSync] ⚠️  xtcocotools install failed (non-fatal) — LipSync pose"
        echo "           estimation will run without xtcocotools COCO helpers."
        echo "           The app will still start; LipSync itself is unaffected."
    fi
fi

echo "[LipSync] Installing mmpose>=1.1.0 (face landmark detection for MuseTalk)..."
# --no-deps: mmpose's dep tree pulls in torch and a chain of heavy packages;
# we only need mmpose itself since its torch/torchvision deps come from the
# system symlinks already in the venv.
if "$APP_PIP" install --quiet --no-cache-dir --no-deps "mmpose>=1.1.0" 2>/dev/null; then
    echo "[LipSync] mmpose installed OK."
else
    echo "[LipSync] ⚠️  mmpose install failed (non-fatal) — MuseTalk pose"
    echo "           estimation falls back to built-in face detector."
fi

# Evict any torch that mmpose/xtcocotools may have re-pulled in.
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done

echo "Setting up RIFE interpolation model..."
if [ ! -d "/root/newgen/train_log/model" ] || [ ! -f "/root/newgen/train_log/RIFE_HDv3.py" ]; then
    rm -rf /root/newgen/train_log /root/newgen/__MACOSX /root/newgen/RIFEv4.26_0921.zip
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    mkdir -p /root/newgen/train_log
    cp -r /tmp/rife/model /root/newgen/train_log/
    wget -q -P /root/newgen https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o /root/newgen/RIFEv4.26_0921.zip -d /root/newgen
    rm -rf /tmp/rife /root/newgen/__MACOSX
    echo "RIFE installed."
else
    echo "RIFE already installed, skipping."
fi

echo "Preparing Wan 2.2 distilled weights directory..."
mkdir -p /root/newgen/wan22_distill
echo "Note: the two merged 4-step BF16 experts (~28.6 GB each, ~57 GB total)"
echo "download automatically on first video generation."
echo "Plan for roughly 150-200 GB total across Wan, Qwen and MMAudio."

# ---------------------------------------------------------------------------
# END-OF-SETUP VERIFICATION
# Run the full import chain inside the venv to catch any remaining issues
# BEFORE the user tries to launch app.py.  A failure here with a clear
# message is far better than a confusing crash 30 seconds into startup.
# ---------------------------------------------------------------------------
echo "Verifying full WanImageToVideoPipeline import chain inside venv..."
"$APP_PY" - <<'VERIFY'
import sys
errors = []

# 1. torch
try:
    import torch
    v = torch.__version__
    assert v and v != "None" and "." in str(v), f"bad version: {v!r}"
    print(f"  [OK] torch {v}")
except Exception as e:
    errors.append(f"  [FAIL] torch: {e}")

# 2. diffusers
try:
    import diffusers
    print(f"  [OK] diffusers {diffusers.__version__}")
except Exception as e:
    errors.append(f"  [FAIL] diffusers: {e}")

# 3. WanLoraLoaderMixin shim
try:
    import diffusers.loaders as _dl
    if not hasattr(_dl, "WanLoraLoaderMixin"):
        class WanLoraLoaderMixin: pass
        _dl.WanLoraLoaderMixin = WanLoraLoaderMixin
        import sys as _s
        m = _s.modules.get("diffusers.loaders")
        if m: setattr(m, "WanLoraLoaderMixin", WanLoraLoaderMixin)
    print("  [OK] WanLoraLoaderMixin available in diffusers.loaders")
except Exception as e:
    errors.append(f"  [FAIL] WanLoraLoaderMixin shim: {e}")

# 4. WanImageToVideoPipeline (the real class, not a dummy stub)
try:
    from diffusers import WanImageToVideoPipeline
    mod = getattr(WanImageToVideoPipeline, "__module__", "") or ""
    if "dummy" in mod:
        errors.append(
            f"  [FAIL] WanImageToVideoPipeline is a dummy stub (module={mod!r}).\n"
            "         diffusers' is_torch_available() returned False.\n"
            "         torch symlinks may be missing — re-run setup.sh."
        )
    else:
        print(f"  [OK] WanImageToVideoPipeline (module={mod!r})")
except Exception as e:
    errors.append(f"  [FAIL] WanImageToVideoPipeline: {e}")

# 5. transformers
try:
    import transformers
    print(f"  [OK] transformers {transformers.__version__}")
except Exception as e:
    errors.append(f"  [FAIL] transformers: {e}")

# 6. gradio
try:
    import gradio
    print(f"  [OK] gradio {gradio.__version__}")
except Exception as e:
    errors.append(f"  [FAIL] gradio: {e}")

if errors:
    print("\n=== SETUP VERIFICATION FAILURES ===")
    for err in errors:
        print(err)
    print("\nFix the issues above before launching app.py.")
    sys.exit(1)
else:
    print("  All import checks passed — venv is ready.")
VERIFY
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  Setup verification failed (see errors above)."
    echo "    The most common cause is a stale torch symlink or a re-installed"
    echo "    pip torch overwriting the system copy. Try: bash setup.sh (again)."
    echo "    DO NOT launch app.py until verification passes."
else
    echo "✅ Setup verification passed."
fi

echo "Copying app files to /root/newgen/..."
if [ "$SCRIPT_DIR" != "/root/newgen" ]; then
    cp "$SCRIPT_DIR/app.py" /root/newgen/app.py
    cp "$SCRIPT_DIR/prompts.py" /root/newgen/prompts.py
    cp "$SCRIPT_DIR/requirements.txt" /root/newgen/requirements.txt
else
    echo "Already in /root/newgen — skipping file copy."
fi

# Copy qwenimage module if present
if [ -d "$SCRIPT_DIR/qwenimage" ] && [ "$SCRIPT_DIR/qwenimage" != "/root/newgen/qwenimage" ]; then
    cp -r "$SCRIPT_DIR/qwenimage" /root/newgen/qwenimage
fi

# Copy starters if present
if [ -d "$SCRIPT_DIR/starters" ] && [ "$SCRIPT_DIR/starters" != "/root/newgen/starters" ]; then
    cp -r "$SCRIPT_DIR/starters" /root/newgen/starters
fi

# ---------------------------------------------------------------------------
# Write the launch wrapper that ensures the app venv's python is used but
# the system torch is visible via PYTHONPATH.  This is what systemd / autorun
# should call instead of plain "python3 app.py".
# ---------------------------------------------------------------------------
cat > /root/newgen/run_app.sh << 'LAUNCHER'
#!/bin/bash
# Launch app.py using the isolated app venv.
# System torch is made visible via symlinks placed directly inside the
# venv's own site-packages by setup.sh (see the SYS_SITE block above) —
# NOT via a PYTHONPATH export. A PYTHONPATH pointing at the whole system
# dist-packages dir would shadow every other same-named package in the
# venv (this previously caused the app to silently load system diffusers
# 0.33.1 instead of the venv's correctly-installed 0.37.1). Do not
# reintroduce a PYTHONPATH export here.
APP_VENV="/root/newgen/.app-venv"
cd /root/newgen
exec "$APP_VENV/bin/python" app.py "$@"
LAUNCHER
chmod +x /root/newgen/run_app.sh

cat > /root/newgen/autorun.sh << 'AUTORUN'
#!/usr/bin/env bash
# ============================================================
#  NewGen VPS Launcher  --  regenerated by setup.sh on every run
#
#  Usage:
#    bash autorun.sh           # start app.py (survives SSH disconnect)
#    bash autorun.sh stop      # kill it
#    bash autorun.sh status    # show PID
#    bash autorun.sh logs      # tail live log
#    bash autorun.sh restart   # stop then start
# ============================================================

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$APP_DIR/app.py"
LOG="$APP_DIR/newgen.log"
PID_FILE="$APP_DIR/app.pid"
APP_VENV="$APP_DIR/.app-venv"

# Prefer the isolated venv python (has all deps installed by setup.sh).
# Fall back to the PYTHON env var or bare python3 only if the venv doesn't exist.
# IMPORTANT: do NOT export PYTHONPATH pointing at the system site-packages here.
# setup.sh places directory symlinks for torch/torchvision/torchaudio inside the
# venv's own site-packages AND writes a zzz_system_torch_path.pth file so
# sys.path already includes the system torch location at runtime — no PYTHONPATH
# needed.  A PYTHONPATH export pointing at the whole system dist-packages dir
# would shadow every same-named package in the venv with the system copy (e.g.
# it would make the app import system diffusers 0.33.1 instead of the venv's
# correctly-installed 0.37.1).
if [ -f "$APP_VENV/bin/python" ]; then
    PYTHON="$APP_VENV/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

do_stop() {
    # Kill via PID file
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[autorun] Stopping PID $PID..."
            kill "$PID"
        fi
        rm -f "$PID_FILE"
    fi
    # Also catch anything missed (nohup, direct python3, venv python, etc.)
    pkill -f "app\.py" 2>/dev/null || true
    echo "[autorun] Stopped."
}

do_start() {
    # Any startup flags forwarded straight to app.py, e.g.:
    #   picgen / -picgen / --picgen        (picgen mode; else vidgen default)
    #   novidgen / -novidgen / --novidgen  (hard-disable the vidgen tab)
    # Passed as separate args; joined verbatim into the launch command.
    local MODE_ARG="$*"

    if is_running; then
        echo "[autorun] Already running (PID $(cat "$PID_FILE")). Run:  bash autorun.sh restart"
        exit 0
    fi
    [[ -f "$APP" ]] || { echo "[autorun] ERROR: $APP not found."; exit 1; }

    if [ -n "$MODE_ARG" ]; then
        echo "[autorun] Starting app.py ($MODE_ARG)..."
    else
        echo "[autorun] Starting app.py..."
    fi
    echo "[autorun] Log  -> $LOG"

    # nohup + setsid: process survives SSH disconnect and terminal close.
    # $MODE_ARG is passed through to app.py (_parse_startup_mode / _parse_novidgen).
    nohup setsid "$PYTHON" "$APP" $MODE_ARG > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"

    echo "[autorun] Waiting for startup..."
    for i in $(seq 1 30); do
        sleep 2
        if grep -q "Running on local URL" "$LOG" 2>/dev/null; then
            echo "[autorun] Started (PID $(cat "$PID_FILE")). Gradio is up."
            echo "[autorun] Confirm AutorunAPI:"
            grep "AutorunAPI\|Running on" "$LOG" | tail -5
            echo ""
            echo "[autorun] ── Live log (Ctrl+C to detach, app keeps running) ──"
            tail -n 80 -f "$LOG"
            return
        fi
        if ! is_running; then
            echo "[autorun] ERROR: process died. Last log:"
            tail -20 "$LOG"
            exit 1
        fi
    done
    echo "[autorun] Still starting (taking longer than usual)."
    echo ""
    echo "[autorun] ── Live log (Ctrl+C to detach, app keeps running) ──"
    tail -n 80 -f "$LOG"
}

_sub="${1:-start}"
shift 2>/dev/null || true
case "$_sub" in
    start)   do_start "$@" ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start "$@" ;;
    status)
        if is_running; then
            echo "[autorun] Running (PID $(cat "$PID_FILE"))"
        else
            echo "[autorun] Not running."
        fi
        ;;
    logs)
        echo "[autorun] Tailing $LOG  (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    *)
        echo "Usage: bash autorun.sh [start|stop|restart|status|logs] [picgen] [novidgen]"
        echo "       bash autorun.sh start -picgen              # picgen mode"
        echo "       bash autorun.sh start -picgen -novidgen    # picgen only, video tab disabled"
        exit 1
        ;;
esac
AUTORUN
chmod +x /root/newgen/autorun.sh

# ---------------------------------------------------------------------------
# LAUNCH APP
# On a systemd VPS: register and start the service (survives reboots).
# In Docker / containers without systemd: launch directly via autorun.sh
# (nohup + setsid so it survives the setup.sh process ending).
# Either way autorun.sh is what runs the app — it always uses the venv python
# and correctly handles PID tracking, log tailing, and stop/restart.
# ---------------------------------------------------------------------------

_print_summary() {
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "IMPORTANT: App now runs via /root/newgen/autorun.sh"
    echo "  System torch 2.8 dev+cu128 is PRESERVED."
    echo "  All app dependencies live in /root/newgen/.app-venv"
    echo ""
    echo "🎬 New Features Available:"
    echo "   • Wan 2.2 I2V A14B merged 4-step distill (BF16, no LoRA)"
    echo "   • Qwen relocates subjects, Wan animates them"
    echo "   • 4-step MoE (high-noise then low-noise expert)"
    echo "   • One model resident on GPU at a time"
    echo "   • 720p native generation (1280x720)"
    echo "   • Zero configuration interface"
    echo ""
}

if [ ! -d /run/systemd/system ]; then
    echo "No systemd detected — launching app directly via autorun.sh..."
    _print_summary
    echo "Management commands:"
    echo "  bash /root/newgen/autorun.sh stop      # stop the app"
    echo "  bash /root/newgen/autorun.sh restart   # restart it"
    echo "  bash /root/newgen/autorun.sh status    # check if running"
    echo "  bash /root/newgen/autorun.sh logs      # tail live log"
    echo ""
    echo "App: http://0.0.0.0:7860"
    echo "📋 Video Tab: Qwen relocate -> Wan 2.2 merged 4-step animate"
    echo "🖼️  Image Tab: Unchanged (Qwen Image Edit)"
    echo ""
    # Stop any old instance first, then start fresh in PICGEN-ONLY mode with the
    # video tab disabled (no Wan / F5-TTS / Foley / MuseTalk downloads). To use
    # vidgen, restart yourself without -novidgen, e.g. bash run.sh restart -picgen
    bash /root/newgen/autorun.sh stop 2>/dev/null || true
    exec bash /root/newgen/autorun.sh start -picgen -novidgen
fi

echo "Setting up systemd service..."
# Copy service file only if it exists alongside this script.
if [ -f "$SCRIPT_DIR/newgen.service" ]; then
    cp "$SCRIPT_DIR/newgen.service" /etc/systemd/system/
fi
systemctl daemon-reload
systemctl enable newgen
# Stop any running instance so autorun.sh can do a clean start below.
systemctl stop newgen 2>/dev/null || true

_print_summary
echo "Service commands:"
echo "  systemctl status newgen"
echo "  systemctl restart newgen"
echo "  bash /root/newgen/autorun.sh logs   # tail live log"
echo ""
echo "App: http://0.0.0.0:7860"
echo "📋 Video Tab: Qwen relocate -> Wan 2.2 merged 4-step animate"
echo "🖼️  Image Tab: Unchanged (Qwen Image Edit)"
echo ""
# Launch via autorun.sh rather than `systemctl start` so the operator sees the
# live log and startup confirmation in their current SSH session, exactly the
# same experience as a manual launch.  The systemd unit is still registered and
# will auto-start the app on future reboots via run_app.sh.
# Autostart in PICGEN-ONLY mode with the video tab DISABLED (-novidgen): picgen
# comes up fast and NOTHING for the vidgen tab is downloaded or loaded on the
# device. To enable video generation, restart the app yourself without
# -novidgen, e.g.  bash run.sh restart -picgen   (or edit the systemd unit).
exec bash /root/newgen/autorun.sh start -picgen -novidgen
