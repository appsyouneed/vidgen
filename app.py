# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys
import random
import math
import tempfile
import warnings
import logging
import time
import gc
import uuid
import threading
import json
import base64
import hashlib
import contextlib
import queue as _queue
import os
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchao").setLevel(logging.ERROR)
# The local qwenimage pipeline logs via diffusers' logger under its own module
# name, so silence it too (it emits the noisy "classifier-free guidance is not
# enabled since true_cfg_scale <= 1" / "negative_prompt is passed but ..."
# warnings on every generation).
logging.getLogger("qwenimage").setLevel(logging.ERROR)
logging.getLogger("qwenimage.pipeline_qwenimage_edit_plus").setLevel(logging.ERROR)

class _TorchaoFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage().lower()
        if "torchao" in msg:
            return False
        # Drop the classifier-free guidance / negative_prompt noise from the
        # Qwen pipeline regardless of which logger emits it.
        if "classifier-free guidance is not enabled" in msg:
            return False
        if "negative_prompt is passed but classifier-free guidance" in msg:
            return False
        return True

logging.getLogger().addFilter(_TorchaoFilter())


# ---------------------------------------------------------------------------
# SELF-HEAL: MuseTalk weights integrity check
#
# A snapshot_download that gets killed mid-transfer (OOM-killed process,
# SSH drop, disk-full) can leave a truncated file on disk that still passes
# .exists() checks forever — causing face-parsing / VAE / UNet to load
# garbage weights and produce corrupt or blank mouth regions instead of
# real lip motion. We check size at startup and delete+re-download anything
# suspiciously small. Runs once in a daemon thread so it never blocks startup.
# ---------------------------------------------------------------------------
def _selfheal_musetalk_weights():
    """
    Runs at startup in a daemon thread. Checks every MuseTalk weight file that
    can be silently truncated by an interrupted download and deletes any that
    are too small so _ensure_musetalk() re-downloads them cleanly.

    Symptoms of bad weights:
      - musetalk/pytorch_model.bin truncated -> UNet load error or garbage mouth region
      - sd-vae-ft-mse/diffusion_pytorch_model.bin truncated -> corrupt/blank frames
      - whisper/tiny.pt truncated -> RuntimeError on every job
    """
    _BASE = Path(__file__).parent / "MuseTalk" / "models"
    checks = [
        ("musetalk/pytorch_model.bin",                    5 * 1024 ** 2,   "MuseTalk UNet"),
        ("musetalkV15/unet.pth",                           5 * 1024 ** 2,   "MuseTalk v1.5 UNet"),
        ("sd-vae-ft-mse/diffusion_pytorch_model.bin",     100 * 1024 ** 2,  "SD VAE (~335 MB)"),
        ("whisper/tiny.pt",                                50 * 1024 ** 2,  "Whisper tiny (~74 MB)"),
        ("dwpose/dw-ll_ucoco_384.pth",                      5 * 1024 ** 2,  "DWPose"),
        ("face-parse-bisent/79999_iter.pth",                5 * 1024 ** 2,  "Face-parse BiSeNet"),
        ("face-parse-bisent/resnet18-5c106cde.pth",         5 * 1024 ** 2,  "Face-parse ResNet18"),
    ]
    for rel, min_bytes, label in checks:
        p = _BASE / rel
        try:
            if not p.exists():
                continue
            sz = p.stat().st_size
            if sz >= min_bytes:
                print(f"[SelfHeal] {rel} OK ({sz / (1024**2):.1f} MB)")
            else:
                print(f"[SelfHeal] {rel} TRUNCATED ({sz / (1024**2):.1f} MB, need >{min_bytes // (1024**2)} MB) — deleting for clean re-download.")
                p.unlink()
                print(f"[SelfHeal] {rel} removed — will re-download on next lip-sync job.")
        except Exception as _e:
            print(f"[SelfHeal] {rel} check failed (non-fatal): {_e}")

threading.Thread(target=_selfheal_musetalk_weights, daemon=True).start()
# ---------------------------------------------------------------------------


STARTUP_MODE = "vidgen"
for _arg in sys.argv[1:]:
    _flag = _arg.lstrip("-").lower()
    if _flag == "vidgen":
        STARTUP_MODE = "vidgen"
    elif _flag == "picgen":
        STARTUP_MODE = "picgen"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs("/dev/shm/newgen", exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "tmp", "gradio"), exist_ok=True)

# ---------------------------------------------------------------------------
# EARLY FILE PATCHES — runs before any gradio import.
# sysconfig.get_path("purelib") always returns the site-packages of the
# running interpreter, so this works correctly inside a venv.
# Both patches are idempotent — safe to run on every startup.
# ---------------------------------------------------------------------------
def _patch_gradio_oauth_early():
    import sysconfig, pathlib
    f = pathlib.Path(sysconfig.get_path("purelib")) / "gradio" / "oauth.py"
    if not f.exists():
        return
    t = f.read_text()
    A = "from fastapi.responses import RedirectResponse\n"
    B = "from .utils import get_space"
    if A not in t or B not in t:
        return
    s, e = t.index(A) + len(A), t.index(B)
    good = (
        "try:\n"
        "    from huggingface_hub import HfFolder, whoami\n"
        "except ImportError:\n"
        "    from huggingface_hub import whoami\n"
        "    try:\n"
        "        from huggingface_hub import get_token as _get_token\n"
        "    except ImportError:\n"
        "        _get_token = lambda: None  # noqa: E731\n"
        "\n"
        "    class HfFolder:  # noqa: N801\n"
        "        @staticmethod\n"
        "        def get_token():\n"
        "            return _get_token()\n"
        "\n"
    )
    patched = t[:s] + good + t[e:]
    if patched != t:
        f.write_text(patched)
        print("[EarlyPatch] gradio/oauth.py fixed")
    else:
        print("[EarlyPatch] gradio/oauth.py already clean")

def _patch_gradio_client_early():
    import sysconfig, pathlib
    f = pathlib.Path(sysconfig.get_path("purelib")) / "gradio_client" / "utils.py"
    if not f.exists():
        return
    t = f.read_text()
    if "if not isinstance(schema, dict):" in t:
        print("[EarlyPatch] gradio_client/utils.py already clean")
        return
    for old, new in [
        (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    \"\"\"Convert the json schema into a python type hint\"\"\"\n    if schema == {}:',
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    \"\"\"Convert the json schema into a python type hint\"\"\"\n    if not isinstance(schema, dict):\n        return \"Any\"\n    if schema == {}:',
        ),
        (
            'def get_type(schema: dict):\n    if \"const\" in schema:',
            'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return \"unknown\"\n    if \"const\" in schema:',
        ),
    ]:
        if old in t:
            f.write_text(t.replace(old, new, 1))
            print("[EarlyPatch] gradio_client/utils.py fixed")
            return

_patch_gradio_oauth_early()
_patch_gradio_client_early()
# ---------------------------------------------------------------------------


try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check
    del _aesgcm_check
except ImportError:
    print("[SelfHeal] cryptography missing — installing into current venv...")
    _r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "cryptography"],
        capture_output=True, text=True,
    )
    if _r.returncode != 0:
        raise RuntimeError(
            f"cryptography install failed:\n{_r.stderr.strip()}\n"
            "Run setup.sh to rebuild the app venv."
        )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check
    del _aesgcm_check
    print("[SelfHeal] cryptography installed OK")

# ---------------------------------------------------------------------------
# Version sanity checks — READ ONLY, never pip-install into the running env.
#
# All runtime dependencies (gradio, diffusers, transformers, numpy, etc.) are
# installed into the isolated app venv by setup.sh.  The self-heal functions
# that used to pip-install packages at runtime have been removed because they
# modified whichever Python environment launched the app — on a system where
# torch 2.8 dev+cu128 is already installed, that caused irreversible version
# downgrades.
#
# If a version check fails below the app exits immediately with a clear
# message so the operator can fix the venv, rather than silently degrading.
# ---------------------------------------------------------------------------

def _self_heal_patch_gradio_oauth():
    """Reapply the HfFolder shim into gradio/oauth.py if needed.
    Patches the file in-place (no pip, no env modification).
    Safe to call multiple times — is a no-op if already patched.

    Uses gradio.__file__ to locate the actual loaded gradio package so this
    works correctly inside a venv (site.getsitepackages() returns system paths
    when called from venv python, causing patches to land in the wrong place).
    """
    try:
        import gradio as _gradio
        candidate = Path(_gradio.__file__).parent / "oauth.py"
        if not candidate.exists():
            print(f"[Patch] gradio/oauth.py not found at {candidate}")
            return
        text = candidate.read_text()

        # Strip any previously botched double-try patch by replacing everything
        # between the RedirectResponse import and 'from .utils import get_space'
        # so repeated runs always produce a clean result.
        ANCHOR_START = "from fastapi.responses import RedirectResponse"
        ANCHOR_END = "from .utils import get_space"
        if ANCHOR_START not in text or ANCHOR_END not in text:
            print(f"[Patch] gradio/oauth.py: anchor lines not found, skipping")
            return

        # Check if already cleanly patched (has try/except but no double-try)
        between_start = text.index(ANCHOR_START) + len(ANCHOR_START)
        between_end = text.index(ANCHOR_END)
        between = text[between_start:between_end]
        if "except ImportError" in between and "    try:\n    from" not in between and "try:\n    try:" not in between:
            print(f"[Patch] gradio/oauth.py already correctly patched")
            return

        GOOD = (
            "\ntry:\n"
            "    from huggingface_hub import HfFolder, whoami\n"
            "except ImportError:\n"
            "    from huggingface_hub import whoami\n"
            "    try:\n"
            "        from huggingface_hub import get_token as _get_token\n"
            "    except ImportError:\n"
            "        _get_token = lambda: None  # noqa: E731\n\n"
            "    class HfFolder:  # noqa: N801\n"
            "        @staticmethod\n"
            "        def get_token():\n"
            "            return _get_token()\n\n"
        )
        text = text[:between_start] + GOOD + text[between_end:]
        candidate.write_text(text)
        print(f"[Patch] gradio/oauth.py patched at {candidate}")
    except Exception as _e:
        print(f"[Patch] gradio oauth patch failed (non-fatal): {_e}")


def _self_heal_patch_gradio_client_utils():
    """Reapply the pydantic v2 bool-schema guard into gradio_client/utils.py.
    Patches the file in-place (no pip, no env modification).
    Safe to call multiple times — is a no-op if already patched.

    Uses gradio_client.__file__ to locate the actual loaded package so this
    works correctly inside a venv.
    """
    try:
        import gradio_client as _gradio_client
        candidate = Path(_gradio_client.__file__).parent / "utils.py"
        if not candidate.exists():
            print(f"[Patch] gradio_client/utils.py not found at {candidate}")
            return
        text = candidate.read_text()
        if "if not isinstance(schema, dict):" in text:
            print(f"[Patch] gradio_client/utils.py already correctly patched")
            return
        OLD = (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n'
            '    """Convert the json schema into a python type hint"""\n'
            '    if schema == {}:'
        )
        NEW = (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n'
            '    """Convert the json schema into a python type hint"""\n'
            '    if not isinstance(schema, dict):\n'
            '        return "Any"\n'
            '    if schema == {}:'
        )
        if OLD in text:
            candidate.write_text(text.replace(OLD, NEW, 1))
            print(f"[Patch] gradio_client/utils.py patched at {candidate}")
            return
        OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
        NEW2 = (
            'def get_type(schema: dict):\n'
            '    if not isinstance(schema, dict):\n'
            '        return "unknown"\n'
            '    if "const" in schema:'
        )
        if OLD2 in text:
            candidate.write_text(text.replace(OLD2, NEW2, 1))
            print(f"[Patch] gradio_client/utils.py (get_type fallback) patched at {candidate}")
            return
        print(f"[Patch] gradio_client/utils.py: no matching pattern found, skipping")
    except Exception as _e:
        print(f"[Patch] gradio_client utils patch failed (non-fatal): {_e}")


def _self_heal_gradio_version():
    """Check gradio version and apply file patches. NEVER pip-installs.
    If the wrong version is found, prints a clear error — the operator must
    rebuild the app venv via setup.sh rather than having the app silently
    modify the Python environment at runtime."""
    _pinned_gradio        = "4.43.0"
    _pinned_gradio_client = "1.3.0"
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur_gradio = _pkg_version("gradio")
        except PackageNotFoundError:
            _cur_gradio = None
        if _cur_gradio != _pinned_gradio:
            print(
                f"[VersionCheck] WARNING: gradio {_cur_gradio} found, expected {_pinned_gradio}.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions.\n"
                f"  The app will attempt to continue but may crash due to API mismatches."
            )
        else:
            print(f"[VersionCheck] gradio {_cur_gradio} ✓")
        # Always reapply file patches — they are idempotent and touch only
        # the gradio/gradio_client source files inside the venv, not packages.
        _self_heal_patch_gradio_oauth()
        _self_heal_patch_gradio_client_utils()
    except Exception as _e:
        print(f"[VersionCheck] gradio version check failed (non-fatal): {_e}")


def _self_heal_transformers_version():
    """Check transformers version. NEVER pip-installs.
    Qwen-Image-Edit-2511 requires transformers>=4.52.0 for the nested
    text_config composite format. If the wrong version is found, prints
    a clear error pointing to setup.sh."""
    _min_transformers = (4, 52, 0)

    def _parse(v):
        parts = []
        for p in (v or "0").split(".")[:3]:
            num = ""
            for ch in p:
                if ch.isdigit(): num += ch
                else: break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur = _pkg_version("transformers")
        except PackageNotFoundError:
            _cur = None

        if _cur is None or _parse(_cur) < _min_transformers:
            _want = ".".join(str(x) for x in _min_transformers)
            print(
                f"[VersionCheck] WARNING: transformers {_cur} found, need >={_want}.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions."
            )
        else:
            print(f"[VersionCheck] transformers {_cur} ✓")
    except Exception as _e:
        print(f"[VersionCheck] transformers version check failed (non-fatal): {_e}")


def _self_heal_torch():
    """Verify torch is installed and has a valid __version__ string.
    NEVER reinstalls torch — the system torch 2.8 dev+cu128 must be preserved.
    Raises RuntimeError if torch is absent or broken so the user gets a clear
    message rather than a confusing downstream crash."""
    try:
        import subprocess as _sp
        _check = _sp.run(
            [sys.executable, "-c",
             "import torch; v=torch.__version__; "
             "assert v and v != 'None' and '.' in str(v), f'bad version: {v!r}'"],
            capture_output=True, text=True,
        )
        if _check.returncode == 0:
            print(f"[VersionCheck] torch ✓")
            return
        raise RuntimeError(
            f"torch sanity check failed: {_check.stderr.strip()[:300]}\n"
            "The system torch install may be broken. Do NOT reinstall torch via "
            "pip — that would overwrite the dev build. Check your CUDA/Python env."
        )
    except RuntimeError:
        raise
    except Exception as _e:
        print(f"[VersionCheck] torch check failed (non-fatal): {_e}")


def _self_heal_diffusers_version():
    """Check diffusers version. NEVER pip-installs.
    WanImageToVideoPipeline requires diffusers>=0.34.0 (added in that release).
    If the wrong version is found, prints a clear error pointing to setup.sh."""

    def _parse(v):
        parts = []
        for p in (v or "0").split(".")[:3]:
            num = ""
            for ch in p:
                if ch.isdigit(): num += ch
                else: break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur = _pkg_version("diffusers")
        except PackageNotFoundError:
            _cur = None

        if _cur is None or _parse(_cur) < (0, 34, 0):
            print(
                f"[VersionCheck] WARNING: diffusers {_cur} found, need >=0.34.0.\n"
                f"  WanImageToVideoPipeline will fail to import.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions."
            )
        else:
            print(f"[VersionCheck] diffusers {_cur} ✓")
    except Exception as _e:
        print(f"[VersionCheck] diffusers version check failed (non-fatal): {_e}")


_self_heal_torch()              # first: transformers crashes if torch.__version__ is None
_self_heal_diffusers_version()
_self_heal_transformers_version()
# Gradio file patches run here — BEFORE gradio is imported — by importing
# gradio temporarily just for __file__ resolution, patching, then letting the
# real import below pick up the corrected files.
_self_heal_gradio_version()     # patches oauth.py and gradio_client/utils.py in-place
_self_heal_diffusers_version()  # re-check after transformers may have rolled deps

# ---------------------------------------------------------------------------
# PRE-IMPORT DIFFUSERS BACKEND CACHE RESET
# diffusers evaluates is_torch_available() at module import time and caches
# the result.  If something imported diffusers before torch was on sys.path
# (e.g. a transitive import during the gradio patch functions above), the
# cache records "torch not available" and every subsequent pipeline class
# resolves to a dummy stub — causing the misleading "PyTorch library not
# found" error even though torch is perfectly installed.
#
# We reset the cache here, AFTER confirming torch is importable (_self_heal_torch
# would have raised if it weren't), and BEFORE the real 'import diffusers'
# below, so diffusers evaluates is_torch_available() fresh.
# ---------------------------------------------------------------------------
try:
    if "diffusers" in sys.modules or "diffusers.utils.import_utils" in sys.modules:
        _diu = sys.modules.get("diffusers.utils.import_utils")
        if _diu is not None:
            # Force the module-level bool back to True.
            if hasattr(_diu, "_torch_available"):
                _diu._torch_available = True
            # Clear any lru_cache wrapping is_torch_available.
            _fn = getattr(_diu, "is_torch_available", None)
            if _fn is not None and hasattr(_fn, "cache_clear"):
                _fn.cache_clear()
            # Also reset the transformers-availability flag since transformers
            # has the same caching pattern and is checked alongside torch.
            if hasattr(_diu, "_transformers_available"):
                _diu._transformers_available = True
            _fn2 = getattr(_diu, "is_transformers_available", None)
            if _fn2 is not None and hasattr(_fn2, "cache_clear"):
                _fn2.cache_clear()
            print("[StartupFix] diffusers backend availability cache reset.")
except Exception as _cache_reset_e:
    print(f"[StartupFix] diffusers cache reset (non-fatal): {_cache_reset_e}")

os.environ.update({
    "TMPDIR": "/dev/shm/newgen",
    "TEMP": "/dev/shm/newgen",
    "TMP": "/dev/shm/newgen",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "ABSL_MIN_LOG_LEVEL": "3",
    "GRPC_VERBOSITY": "ERROR",
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,backend:cudaMallocAsync",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    "HF_HOME": "/root/.cache/huggingface",
    "CUDA_LAUNCH_BLOCKING": "0",
    "OMP_NUM_THREADS": "8",
    # Suppress pip noise globally for all subprocess pip calls:
    # root-user warning, version-check nag, deprecation notices, conflict chatter.
    "PIP_ROOT_USER_ACTION": "ignore",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_WARN_CONFLICTS": "1",
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning",
    # Force onnxruntime to use CPU if CUDA libs missing (for rembg background removal)
    "ORT_FORCE_CPU": "1",
})

import cv2
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings as _warnings
# torch/transformers/diffusers install their own warnings filters during
# import, which can override the blanket warnings.filterwarnings("ignore")
# set at the top of this file. This specific message ("Unable to import
# `torchao` Tensor objects...") is emitted via warnings.warn (not logging),
# so the _TorchaoFilter logging.Filter above never catches it. Re-assert
# suppression here, after the heavy imports, targeting it by message text.
_warnings.filterwarnings("ignore", message=r".*torchao.*Tensor objects.*")

from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file

def _patch_wan_lora_loader_mixin():
    """
    diffusers 0.37.1 ships WanImageToVideoPipeline but its pipeline_wan_i2v
    module tries to import WanLoraLoaderMixin from diffusers.loaders, which
    does not exist in 0.37.1.  That import failure propagates through ALL three
    import paths tried below (including the top-level 'diffusers' path, because
    diffusers/__init__.py re-imports from the same broken submodule), leaving
    WanImageToVideoPipeline unresolvable and causing the misleading
    "PyTorch library not found" error from the dummy stub at runtime.

    Fix: inject an empty WanLoraLoaderMixin shim into diffusers.loaders
    BEFORE any wan submodule is touched.  This is safe because diffusers 0.37.1
    never actually calls any method on WanLoraLoaderMixin — the class is only
    listed as a base class in the pipeline definition.  When diffusers is
    upgraded to a version that ships the real class, the shim is never reached
    (the real class will already be present in diffusers.loaders).
    """
    try:
        import diffusers.loaders as _dl
        if hasattr(_dl, "WanLoraLoaderMixin"):
            return  # real class already present — nothing to do
        # Inject a no-op shim so the pipeline_wan_i2v 'from diffusers.loaders
        # import WanLoraLoaderMixin' succeeds.
        class WanLoraLoaderMixin:  # noqa: N801
            """Compatibility shim for diffusers < 0.38."""
        _dl.WanLoraLoaderMixin = WanLoraLoaderMixin
        # Also register on the already-loaded sys.modules entry so that
        # 'from diffusers.loaders import WanLoraLoaderMixin' in the wan
        # pipeline submodule resolves to the shim at import time.
        import sys as _sys
        _loaders_mod = _sys.modules.get("diffusers.loaders")
        if _loaders_mod is not None:
            setattr(_loaders_mod, "WanLoraLoaderMixin", WanLoraLoaderMixin)
        print("[WanImport] Injected WanLoraLoaderMixin shim into diffusers.loaders "
              "(diffusers 0.37.1 compat)")
    except Exception as _e:
        print(f"[WanImport] WanLoraLoaderMixin shim injection failed (non-fatal): {_e}")


# Apply WanLoraLoaderMixin shim NOW — before gradio is imported.
# gradio's transitive imports can touch diffusers' lazy loader which evaluates
# the wan pipeline submodule imports.  If WanLoraLoaderMixin is missing at
# that point, diffusers silently falls back to a dummy stub even though torch
# is correctly installed.  Patching diffusers.loaders here, before any gradio
# import, ensures the shim is in place for every subsequent import attempt.
_patch_wan_lora_loader_mixin()

import gradio as gr


def _import_wan_pipeline():
    """Try every known import path across diffusers versions.
    Catches Exception broadly (not just ImportError) because a broken torch
    causes transformers to raise packaging.version.InvalidVersion (ValueError)
    at import time, which would otherwise silently swallow the real cause.

    IMPORTANT: always try the top-level 'diffusers' path FIRST.
    diffusers/__init__.py imports WanImageToVideoPipeline via a lazy loader
    that avoids executing the full pipeline_wan_i2v module at import time,
    so it succeeds even if pipeline_wan_i2v has a missing-symbol error.
    The submodule paths are tried as fallbacks only.
    """
    # ---------------------------------------------------------------------------
    # PRE-FLIGHT: verify torch is actually importable before attempting any
    # diffusers import.  If torch is missing, diffusers marks ALL backends as
    # unavailable and returns dummy stubs for every pipeline class — the stubs'
    # from_pretrained() then raises the misleading "PyTorch library not found"
    # error even when the real problem is just a broken/absent torch install.
    # Catching this here gives a clear, actionable error message.
    # ---------------------------------------------------------------------------
    try:
        import torch as _torch_preflight
        _tv = _torch_preflight.__version__
        if not _tv or _tv == "None":
            raise ImportError(f"torch.__version__ is invalid: {_tv!r}")
        # Also verify CUDA is available (required for Wan pipeline)
        if not _torch_preflight.cuda.is_available():
            print("[WanImport] WARNING: torch.cuda.is_available() == False — "
                  "pipeline will be loaded on CPU and moved to GPU later.")
    except ImportError as _te:
        raise ImportError(
            f"torch is not importable in this Python environment ({sys.executable}).\n"
            f"Error: {_te}\n"
            "The venv relies on system torch symlinks placed by setup.sh.\n"
            "Fix: re-run setup.sh to rebuild the venv and re-create the symlinks."
        ) from _te

    # Force diffusers to re-evaluate its backend availability now that we have
    # confirmed torch is present.  diffusers caches is_torch_available() at
    # first import; if diffusers was imported before torch was on sys.path
    # (e.g. via gradio's transitive imports), the cache is stale.
    try:
        import diffusers.utils.import_utils as _diu
        # Reset the cached result so the next call re-runs find_spec("torch").
        if hasattr(_diu, "_torch_available"):
            # diffusers < 0.28 stored it as a module-level bool
            _diu._torch_available = True
        # diffusers >= 0.28 uses a cached_property / lru_cache; clear it.
        for _attr in ("is_torch_available",):
            _fn = getattr(_diu, _attr, None)
            if _fn is not None and hasattr(_fn, "cache_clear"):
                _fn.cache_clear()
        # Re-import torch inside diffusers' namespace so its availability flag
        # is definitely True for the wan pipeline import that follows.
        _diu.is_torch_available()
    except Exception as _diu_e:
        print(f"[WanImport] diffusers import_utils reset failed (non-fatal): {_diu_e}")

    # Inject the WanLoraLoaderMixin shim before any wan submodule is touched.
    _patch_wan_lora_loader_mixin()

    attempts = [
        # Top-level first — uses diffusers' lazy __init__ loader, avoids the
        # WanLoraLoaderMixin import in pipeline_wan_i2v entirely.
        ("diffusers",                                 "WanImageToVideoPipeline"),
        # Submodule paths as fallback (work once the shim is in place).
        ("diffusers.pipelines.wan",                   "WanImageToVideoPipeline"),
        ("diffusers.pipelines.wan.pipeline_wan_i2v",  "WanImageToVideoPipeline"),
    ]
    last_exc = None
    for module_path, class_name in attempts:
        try:
            import importlib as _il
            mod = _il.import_module(module_path)
            cls = getattr(mod, class_name, None)
            # Reject dummy stubs — they live in diffusers/utils/dummy_*.py and
            # their from_pretrained() raises ImportError("requires PyTorch")
            # regardless of the actual environment.
            # BUG-FIX: the original check was:
            #   "dummy" not in getattr(cls.__module__ or "", "", "")
            # getattr(str, "", "") calls getattr on the string object with
            # attr="" which always returns "" regardless of the string value,
            # so the check always passed and dummy stubs were silently accepted.
            if cls is not None:
                cls_module = getattr(cls, "__module__", "") or ""
                if "dummy" in cls_module:
                    print(f"[WanImport] {module_path!r} returned dummy stub "
                          f"(module={cls_module!r}), trying next path")
                else:
                    # Also verify from_pretrained won't immediately raise —
                    # real pipeline classes inherit from DiffusionPipeline
                    # which has from_pretrained defined in diffusers.pipelines.pipeline_utils,
                    # not in a dummy_ module.
                    fp_module = getattr(
                        getattr(cls, "from_pretrained", None), "__module__", ""
                    ) or ""
                    if "dummy" in fp_module:
                        print(f"[WanImport] {module_path!r} from_pretrained is dummy "
                              f"(module={fp_module!r}), trying next path")
                    else:
                        return cls
        except Exception as _e:
            last_exc = _e
            print(f"[WanImport] {module_path!r} failed: {type(_e).__name__}: {_e}")
    try:
        import importlib.metadata as _im
        dv = _im.version("diffusers")
    except Exception:
        dv = "unknown"
    raise ImportError(
        f"Cannot import WanImageToVideoPipeline from diffusers {dv}. "
        "Requires diffusers>=0.34.0 and a working torch install. "
        f"Last error: {last_exc}"
    ) from last_exc

WanImageToVideoPipeline = _import_wan_pipeline()

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

sys.path.insert(0, SCRIPT_DIR)


_media_store: dict[str, tuple[bytes, str]] = {}   # key -> (data, filename)
_media_store_lock = threading.Lock()



import secrets as _secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
from cryptography.hazmat.primitives import hashes as _hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF as _HKDF

_log_queue: "_queue.Queue[str]" = _queue.Queue(maxsize=2000)

_builtin_print = print

def print(*args, **kwargs):  # noqa: A001
    """Drop-in print replacement: writes to stdout AND pushes to _log_queue."""
    sep = kwargs.get("sep", " ")
    line = sep.join(str(a) for a in args)
    _builtin_print(*args, **kwargs)
    try:
        _log_queue.put_nowait(line)
    except Exception:
        pass


def _derive_media_key(browser_secret_hex: str) -> bytes:
    """Derive a 32-byte AES-256-GCM key from the browser's localStorage secret.

    Uses HKDF-SHA256 with a fixed salt and info so the same secret always
    produces the same key — but the key is never transmitted, only the secret.
    """
    try:
        secret_bytes = bytes.fromhex(browser_secret_hex)
    except ValueError:
        return _secrets.token_bytes(32)
    hkdf = _HKDF(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=b"newgen-media-v1",
        info=b"aes-256-gcm-media",
    )
    return hkdf.derive(secret_bytes)


def _encrypt_bytes(data: bytes, key_bytes: bytes) -> bytes:
    """Encrypt data with AES-256-GCM. Returns 12-byte nonce + ciphertext."""
    nonce = _secrets.token_bytes(12)
    aesgcm = _AESGCM(key_bytes)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext


def _derive_log_key(browser_secret_hex: str) -> bytes:
    """Derive a 32-byte AES-256-GCM key for log line encryption."""
    try:
        secret_bytes = bytes.fromhex(browser_secret_hex)
    except ValueError:
        return _secrets.token_bytes(32)
    hkdf = _HKDF(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=b"newgen-logs-v1",
        info=b"aes-256-gcm-logs",
    )
    return hkdf.derive(secret_bytes)


def _encrypt_log_line(line: str, key_bytes: bytes) -> str:
    """Encrypt a log line and return base64(nonce+ciphertext)."""
    nonce = _secrets.token_bytes(12)
    aesgcm = _AESGCM(key_bytes)
    ct = aesgcm.encrypt(nonce, line.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode()


def _media_store_put(data: bytes, filename: str) -> str:
    """Store bytes in RAM, return the /media/ URL to serve them."""
    key = uuid.uuid4().hex
    with _media_store_lock:
        _media_store[key] = (data, filename)
    return f"/media/{key}/{filename}"


def _media_store_get(key: str):
    """Return (bytes, filename) or None."""
    with _media_store_lock:
        return _media_store.get(key)


def _media_store_release(url_or_key: str):
    """Remove one entry by /media/<key>/... URL or bare key."""
    if not url_or_key:
        return
    key = url_or_key.split("/")[2] if url_or_key.startswith("/media/") else url_or_key
    with _media_store_lock:
        _media_store.pop(key, None)


def _media_store_release_prefix(prefix: str):
    """Remove all entries whose filename starts with prefix."""
    with _media_store_lock:
        remove = [k for k, (_, fn) in _media_store.items() if fn.startswith(prefix)]
        for k in remove:
            del _media_store[k]


VIDGEN_TMP_DIR = Path(os.path.join(SCRIPT_DIR, "tmp", "gradio"))


def _write_video_tmp(data: bytes, filename: str) -> str:
    """Write video bytes to the RAM filesystem and return the absolute path.
    
    Uses /root/newgen/tmp/gradio/ which is in allowed_paths so Gradio can serve it via /file=.
    """
    VIDGEN_TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDGEN_TMP_DIR / filename
    path.write_bytes(data)
    return str(path)


def _delete_video_tmp(path: str):
    """Delete a video tmp file. Called after the browser has downloaded it.
    
    Safe to call even if the file no longer exists (e.g. already cleaned up
    by _do_clear_storage). Silent no-op in that case.
    """
    try:
        if path and os.path.exists(path):
            os.unlink(path)
            print(f"[tmp] deleted {os.path.basename(path)}")
        # else: file already gone, silently skip
    except Exception as e:
        print(f"[tmp] delete failed (non-fatal): {e}")


def _media_name(kind: str, extension: str, index: int = None) -> str:
    """Build a collision-free download filename (no path, no folder)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    suffix = f"_{index:02d}" if index is not None else ""
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{kind}_{stamp}_{token}{suffix}{ext}"

from prompts import (
    solo_prompts_dict, couple_man_unseen_prompts_dict, couple_man_seen_prompts_dict, 
    multiple_women_prompts_dict, multiple_man_unseen_prompts_dict, multiple_man_seen_prompts_dict, multistep_prompts_dict,
    vid_solo_prompts_dict, vid_couple_prompts_dict, vid_multiple_prompts_dict, vid_multistep_prompts_dict,
    vid_environment_prompts_dict, vid_custom_prompts_dict, vid_multiple_man_unseen_prompts_dict, vid_multiple_man_seen_prompts_dict,
    update_solo_prompt, update_couple_man_unseen_prompt, update_couple_man_seen_prompt, 
    update_multiple_women_prompt, update_multiple_man_unseen_prompt, update_multiple_man_seen_prompt, update_multistep_prompt,
    update_vid_prompt1, update_vid_prompt2, update_vid_prompt3, update_vid_prompt4,
    update_vid_prompt5, update_vid_prompt6, update_vid_prompt7, update_vid_prompt8,
)



def extract_frame(video_url_or_buf, timestamp) -> str | None:
    """Extract a frame at `timestamp` seconds from a video.

    `video_url_or_buf` may be:
      - a /media/<key>/<name> URL  (generated video in _media_store)
      - raw bytes                   (direct buffer)
      - a file path string          (uploaded video on disk, legacy path)

    Returns a /media/ URL for the extracted JPEG, or None on failure.
    Nothing is written to disk.
    """
    import av as _av

    print(f"  [extract_frame] ts={timestamp}")

    if isinstance(video_url_or_buf, bytes):
        video_bytes = video_url_or_buf
    elif isinstance(video_url_or_buf, str) and video_url_or_buf.startswith("/media/"):
        key = video_url_or_buf.split("/")[2]
        entry = _media_store_get(key)
        if entry is None:
            print(" [extract_frame] stream key not found in _media_store")
            return None
        video_bytes, _ = entry
    else:
        path = str(video_url_or_buf)
        if not os.path.exists(path):
            print(f" [extract_frame] file not found: {path}")
            return None
        with open(path, "rb") as f:
            video_bytes = f.read()

    try:
        container = _av.open(BytesIO(video_bytes), mode="r")
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            container.close()
            print(" [extract_frame] no video stream found")
            return None

        fps = float(video_stream.average_rate or 16)
        seek_pts = max(0, int((timestamp - 1.0) * 1_000_000))
        try:
            container.seek(seek_pts, any_frame=True)
        except Exception:
            try:
                container.seek(0)
            except Exception:
                pass

        target_frame = None
        for frame in container.decode(video=0):
            frame_ts = float(frame.pts * video_stream.time_base) if frame.pts is not None else 0.0
            target_frame = frame
            if frame_ts >= timestamp:
                break
        if target_frame is None:
            try:
                container.seek(0)
                for frame in container.decode(video=0):
                    target_frame = frame
                    break
            except Exception:
                pass
        container.close()

        if target_frame is None:
            print(" [extract_frame] no frame decoded")
            return None

        pil_img = target_frame.to_image()
        filename = _media_name("extracted_frame", ".jpg")
        fpath = os.path.join(SCRIPT_DIR, "tmp", "gradio", filename)
        pil_img.save(fpath, format="JPEG", quality=95)
        print(f" [extract_frame] saved as {filename}")
        return fpath

    except Exception as e:
        print(f" [extract_frame] failed: {e}")
        return None



if not os.path.exists(os.path.join(SCRIPT_DIR, "train_log", "RIFE_HDv3.py")):
    print("Downloading RIFE Model...")
    if not os.path.exists(os.path.join(SCRIPT_DIR, "RIFEv4.26_0921.zip")):
        subprocess.run([
            "wget", "-q",
            "https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip",
            "-O", os.path.join(SCRIPT_DIR, "RIFEv4.26_0921.zip")
        ], check=True)
    subprocess.run(["unzip", "-n", os.path.join(SCRIPT_DIR, "RIFEv4.26_0921.zip")], check=True)

try:
    import torchvision
except ModuleNotFoundError:
    print("[SelfHeal] torchvision missing — installing...")
    _r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130"],
        capture_output=True, text=True,
    )
    if _r.returncode != 0:
        _r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "torchvision"],
            capture_output=True, text=True,
        )
    if _r.returncode != 0:
        raise RuntimeError(f"torchvision install failed:\n{_r.stderr.strip()}")
    print("[SelfHeal] torchvision installed OK")

sys.path.append(os.path.join(SCRIPT_DIR, "train_log"))
from train_log.RIFE_HDv3 import Model


FORCE_DUAL_RESIDENT = False
AGGRESSIVE_OPTIMIZATION = True

_gpu_count = torch.cuda.device_count()
if _gpu_count < 1:
    raise RuntimeError("No CUDA device visible  this app requires a GPU.")

DUAL_GPU = _gpu_count >= 2 and os.environ.get("NEWGEN_FORCE_SINGLE_GPU") != "1"
PIC_DEVICE = "cuda:0"
WAN_DEVICE = "cuda:1" if DUAL_GPU else "cuda:0"

if DUAL_GPU:
    print(f"Dual GPU: Qwen -> {PIC_DEVICE}, Wan -> {WAN_DEVICE}. Both load at startup, no swapping.")

PIC_QUEUE_ID = "pic-gpu" if DUAL_GPU else "gpu"
WAN_QUEUE_ID = "wan-gpu" if DUAL_GPU else "gpu"

device = torch.device(PIC_DEVICE)

rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()
rife_model.device()

rife_model.flownet = rife_model.flownet.float()


@torch.no_grad()
def interpolate_bits(frames_np, multiplier=2, scale=1.0):
    if isinstance(frames_np, list):
        T = len(frames_np)
        H, W, C = frames_np[0].shape
    else:
        T, H, W, C = frames_np.shape

    if multiplier < 2:
        if isinstance(frames_np, np.ndarray):
            return list(frames_np)
        return frames_np

    n_interp = multiplier - 1
    tmp = max(128, int(128 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    def to_tensor(frame_np):
        t = torch.from_numpy(frame_np).to(device)
        if t.dtype != torch.float32:
            t = t.float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        return F.pad(t, padding)

    def from_tensor(tensor):
        t = tensor[0, :, :H, :W]
        t = t.permute(1, 2, 0)
        return t.float().cpu().numpy()

    def make_inference(I0, I1, n):
        if rife_model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(rife_model.inference(I0, I1, (i + 1) * 1. / (n + 1), scale))
            return res
        else:
            middle = rife_model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n // 2)
            second_half = make_inference(middle, I1, n=n // 2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    output_frames = []
    I1 = to_tensor(frames_np[0])
    total_steps = T - 1

    with tqdm(total=total_steps, desc="Interpolating", unit="frame") as pbar:
        for i in range(total_steps):
            I0 = I1
            output_frames.append(from_tensor(I0))
            I1 = to_tensor(frames_np[i + 1])
            mid_tensors = make_inference(I0, I1, n_interp)
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))
            if (i + 1) % 50 == 0:
                pbar.update(50)
        pbar.update(total_steps % 50)
        output_frames.append(from_tensor(I1))

    del I0, I1, mid_tensors
    return output_frames



def encode_frames_to_bytes(frames: list, fps: int, quality: int = 8) -> bytes:
    """Encode a list of frames (PIL Images or numpy HxWx3 float/uint8) to MP4 in memory.

    Returns raw MP4 bytes — nothing is written to disk.
    `quality` maps to libx264's CRF (1=best, 51=worst; default 8 ≈ high quality).
    animate_frame() returns numpy float32 arrays in [0,1]; this function handles both.
    """
    import av as _av
    import numpy as _np

    if not frames:
        raise ValueError("encode_frames_to_bytes: empty frame list")

    pil_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            pil_frames.append(f)
        else:
            arr = _np.asarray(f)
            if arr.dtype != _np.uint8:
                arr = (arr * 255).clip(0, 255).astype(_np.uint8)
            pil_frames.append(Image.fromarray(arr))

    buf = BytesIO()
    w, h = pil_frames[0].size
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    container = _av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    crf = max(18, min(35, 36 - int(quality * 1.8)))
    stream.options = {"crf": str(crf), "preset": "fast"}

    for pil_frame in pil_frames:
        if pil_frame.size != (w, h):
            pil_frame = pil_frame.resize((w, h))
        arr = _np.array(pil_frame.convert("RGB"))
        av_frame = _av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():  # flush
        container.mux(pkt)
    container.close()
    return buf.getvalue()



# ---------------------------------------------------------------------------
# Dual Audio Engine: F5-TTS (voice cloning) + HunyuanVideo-Foley (SFX/Foley)
# Replaces MMAudio completely.
#
# F5-TTS    — installed into its own venv at /root/newgen/.f5tts-venv and
#              invoked via subprocess (isolated on purpose, see below).
# HunyuanVideo-Foley — cloned repo at /root/newgen/HunyuanVideo-Foley,
#              called via subprocess (it has no installable Python package).
# ---------------------------------------------------------------------------

FOLEY_REPO_DIR  = Path(SCRIPT_DIR) / "HunyuanVideo-Foley"
FOLEY_MODEL_DIR = Path(SCRIPT_DIR) / "HunyuanVideo-Foley-weights"

# ---------------------------------------------------------------------------
# Isolated venvs for audio/lipsync engines.
#
# ALL three engines (F5-TTS, HunyuanVideo-Foley, MuseTalk) live in their
# own venvs that are completely separate from both the system Python packages
# and the app venv (/root/newgen/.app-venv).  This means:
#   • protobuf conflicts between F5-TTS (>=6.33) and Foley (<5.0) are gone
#   • MuseTalk's mmcv/mmdet/mmpose (OpenMMLab) stack cannot downgrade the
#     app's numpy/torch, and its own torch copy never touches the system one
#   • the system torch 2.8 dev+cu128 is NEVER touched by any of these installs
#
# Each engine is invoked exclusively via subprocess using its own venv python.
# sys.executable is NEVER passed to pip for any of these engines.
# ---------------------------------------------------------------------------

# F5-TTS venv — isolated: protobuf>=6.33 conflict with Foley's <5.0
F5_VENV_DIR = Path(SCRIPT_DIR) / ".f5tts-venv"
F5_VENV_PY  = F5_VENV_DIR / "bin" / "python"

# HunyuanVideo-Foley venv — isolated: protobuf<5.0, own torch copy
FOLEY_VENV_DIR = Path(SCRIPT_DIR) / ".foley-venv"
FOLEY_VENV_PY  = FOLEY_VENV_DIR / "bin" / "python"

# MuseTalk venv — isolated: mmcv/mmdet/mmpose (OpenMMLab) + own torch copy,
# never touches system numpy/torch.
MUSETALK_VENV_DIR = Path(SCRIPT_DIR) / ".musetalk-venv"
MUSETALK_VENV_PY  = MUSETALK_VENV_DIR / "bin" / "python"

_AUDIO_ENGINE_AVAILABLE = False   # set True once both engines verified usable


def _ensure_audio_engines():
    """
    One-time setup: clone HunyuanVideo-Foley repo if missing, download model
    weights from HuggingFace, and pip-install f5-tts if not present.
    Called at startup so the first generation isn't delayed.
    """
    global _AUDIO_ENGINE_AVAILABLE

    # --- Foley venv bootstrap -------------------------------------------------
    # HunyuanVideo-Foley runs in its own isolated venv (.foley-venv) so its
    # protobuf<5.0 requirement from descript-audiotools never touches the system
    # python or the app venv. torch/torchaudio/torchcodec are installed inside
    # this venv — the system torch is untouched.
    _FOLEY_PIP_QUIET = [
        "--quiet", "--disable-pip-version-check", "--root-user-action=ignore",
        "--no-warn-conflicts", "--no-cache-dir",
    ]

    if not FOLEY_VENV_PY.exists():
        print("[AudioEngine] Creating isolated venv for HunyuanVideo-Foley ...")
        subprocess.run([sys.executable, "-m", "venv", str(FOLEY_VENV_DIR)], check=True)
        subprocess.run(
            [str(FOLEY_VENV_PY), "-m", "pip", "install"] + _FOLEY_PIP_QUIET +
            ["--upgrade", "pip"],
            check=True, capture_output=True,
        )
        # Install torch + torchaudio + torchcodec inside the Foley venv.
        # torchaudio 2.6+ routes torchaudio.save() through TorchCodec; without
        # torchcodec, infer.py crashes at torchaudio.save() with
        # "ModuleNotFoundError: No module named 'torchcodec'" and videos are silent.
        print("[AudioEngine] Installing torch/torchaudio/torchcodec into Foley venv ...")
        subprocess.run(
            [str(FOLEY_VENV_PY), "-m", "pip", "install"] + _FOLEY_PIP_QUIET +
            ["torch", "torchaudio"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [str(FOLEY_VENV_PY), "-m", "pip", "install"] + _FOLEY_PIP_QUIET +
            ["torchcodec"],
            check=False, capture_output=True,  # non-fatal: falls back to sox
        )
        print("[AudioEngine] Foley venv base ready.")

    # --- Ensure `transformers` is present in the Foley venv -----------------
    # HunyuanVideo-Foley's requirements.txt pins a git+ dev branch of
    # transformers. The repo-requirements install below deliberately strips
    # any line starting with "transformers" (and any git+transformers line)
    # out of that file — see _CONFLICTING_PREFIXES further down — because
    # blindly installing a random dev branch is risky. But since that git+
    # line was the ONLY place `transformers` would have been installed, the
    # net effect was that the Foley venv never got `transformers` at all,
    # so infer.py's `from transformers import ...` crashed with
    # "ModuleNotFoundError: No module named 'transformers'". The Foley venv
    # is fully isolated from the rest of the app, so installing a normal,
    # stable PyPI `transformers` release here is safe — it can't conflict
    # with anything outside `.foley-venv`. This check runs on every startup
    # (cheap when already satisfied) so it also repairs venvs created before
    # this fix existed, without needing to delete/recreate them.
    _foley_transformers_check = subprocess.run(
        [str(FOLEY_VENV_PY), "-c", "import transformers"],
        capture_output=True,
    )
    if _foley_transformers_check.returncode != 0:
        print("[AudioEngine] Installing transformers into Foley venv ...")
        _foley_tf_result = subprocess.run(
            [str(FOLEY_VENV_PY), "-m", "pip", "install"] + _FOLEY_PIP_QUIET +
            ["transformers==4.46.3"],
            capture_output=True,
        )
        if _foley_tf_result.returncode == 0:
            print("[AudioEngine] transformers installed into Foley venv.")
        else:
            print("[AudioEngine] Failed to install transformers into Foley venv: "
                  f"{_foley_tf_result.stderr[-500:]}")
    else:
        # Even if transformers is importable, an unpinned/too-new install
        # (e.g. from a prior run of this code before the version was pinned)
        # may be missing APIs HunyuanVideo-Foley's synchformer/ast_model.py
        # needs — e.g. transformers.pytorch_utils.find_pruneable_heads_and_indices
        # was removed in newer transformers releases, causing
        # "ImportError: cannot import name 'find_pruneable_heads_and_indices'
        # from 'transformers.pytorch_utils'". Detect that specific breakage
        # and pin down to a known-good version if so.
        _foley_ast_check = subprocess.run(
            [str(FOLEY_VENV_PY), "-c",
             "from transformers.pytorch_utils import find_pruneable_heads_and_indices"],
            capture_output=True,
        )
        if _foley_ast_check.returncode != 0:
            print("[AudioEngine] Foley venv's transformers is missing APIs "
                  "HunyuanVideo-Foley needs — pinning to a compatible version ...")
            _foley_tf_result = subprocess.run(
                [str(FOLEY_VENV_PY), "-m", "pip", "install"] + _FOLEY_PIP_QUIET +
                ["transformers==4.46.3"],
                capture_output=True,
            )
            if _foley_tf_result.returncode == 0:
                print("[AudioEngine] transformers pinned to 4.46.3 in Foley venv.")
            else:
                print("[AudioEngine] Failed to pin transformers in Foley venv: "
                      f"{_foley_tf_result.stderr[-500:]}")

    # --- F5-TTS (isolated venv) -----------------------------------------
    # F5-TTS's dependency chain (via cached_path -> google-cloud-storage ->
    # google-api-core) needs protobuf>=6.33.5, while HunyuanVideo-Foley's
    # descript-audiotools dependency pins protobuf<5.0.0. Both requirements
    # are real for the code paths that use them, so they cannot share one
    # site-packages: installing whichever runs second just uninstalls the
    # other's protobuf on disk. Worse, even if that install "worked", the
    # main app process would still have the OLD protobuf module cached in
    # sys.modules from whatever imported it first at startup � a pip install
    # after import never takes effect in the same process. So F5-TTS gets
    # its own venv (like HunyuanVideo-Foley gets its own subprocess/cwd) and
    # is invoked as a subprocess via _run_f5tts(), never imported in-process.
    # If the venv exists but has a broken f5-tts install (e.g. 1.1.4 with
    # missing configs/), reinstall the correct version automatically.
    if F5_VENV_PY.exists():
        _f5_check = subprocess.run(
            [str(F5_VENV_PY), "-c",
             "from f5_tts.api import F5TTS; import pathlib, f5_tts; "
             "p=pathlib.Path(f5_tts.__file__).parent/'configs'/'F5-TTS.yaml'; "
             "exit(0 if p.exists() else 1)"],
            capture_output=True,
        )
        if _f5_check.returncode != 0:
            print("[AudioEngine] f5-tts configs missing in existing venv — reinstalling correct version...")
            result = subprocess.run(
                [str(F5_VENV_PY), "-m", "pip", "install", "--quiet",
                 "--disable-pip-version-check", "--root-user-action=ignore",
                 "f5-tts==1.1.22", "openai-whisper"],
                capture_output=True,
            )
            if result.returncode == 0:
                print("[AudioEngine] f5-tts reinstalled.")
            else:
                print("[AudioEngine] f5-tts reinstall failed — worker will write missing yaml at runtime.")

    if not F5_VENV_PY.exists():
        # Deliberately NOT --system-site-packages: if the venv could see the
        # main env's protobuf, pip would treat it as "already satisfied" and
        # skip installing the venv's own copy, defeating the isolation this
        # venv exists for. Torch/torchaudio get reinstalled here too (an
        # extra multi-GB download, one-time) so the venv's site-packages are
        # fully self-contained and immune to whatever the main env does later.
        print("[AudioEngine] Creating isolated venv for F5-TTS (this downloads "
              "torch/torchaudio again, ~one-time cost, for full isolation)...")
        subprocess.run([sys.executable, "-m", "venv", str(F5_VENV_DIR)], check=True)
        subprocess.run(
            [str(F5_VENV_PY), "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--root-user-action=ignore",
             "--upgrade", "pip"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [str(F5_VENV_PY), "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--root-user-action=ignore",
             "torch", "torchaudio"],
            check=True, capture_output=True,
        )
        # 1.1.22: latest stable. configs/ may not be bundled by pip, but the
        # worker script writes the missing yaml itself at runtime if needed.
        subprocess.run(
            [str(F5_VENV_PY), "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--root-user-action=ignore",
             "f5-tts==1.1.22"],
            capture_output=True,
        )
        # openai-whisper: auto-transcribes the reference clip so F5-TTS knows
        # the language and content of the voice sample it is cloning from.
        subprocess.run(
            [str(F5_VENV_PY), "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--root-user-action=ignore",
             "openai-whisper"],
            check=True, capture_output=True,
        )
        print("[AudioEngine] F5-TTS venv ready.")

    # --- HunyuanVideo-Foley repo --------------------------------------
    if not FOLEY_REPO_DIR.exists():
        print("[AudioEngine] Cloning HunyuanVideo-Foley repo...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley",
             str(FOLEY_REPO_DIR)],
            check=True,
        )
        # Install repo dependencies into current env, but strip out lines that
        # would clobber our pinned stack. HunyuanVideo-Foley's own
        # requirements.txt pins gradio==3.50.2 and a git transformers branch,
        # and repeats torch/torchvision/torchaudio/numpy. Installing those
        # verbatim downgrades our pinned gradio==4.43.0 mid-run, which is what
        # caused "ImportError: cannot import name 'http_server' from 'gradio'"
        # at demo.launch() — gradio's package files ended up a mix of two
        # incompatible versions. Everything else in that file (av, einops,
        # omegaconf, pyyaml, scipy, timm, sentencepiece, accelerate, pandas,
        # pyarrow, loguru, easydict, descript-audiotools, etc.) is safe to
        # install as-is here -- audiotools' protobuf<5.0.0 pin is fine in
        # this env now that F5-TTS (the only thing that needed protobuf>=6.33)
        # runs in its own isolated venv instead of this shared site-packages.
        _CONFLICTING_PREFIXES = (
            "torch", "gradio", "transformers", "numpy", "urllib3",
        )
        req_file = FOLEY_REPO_DIR / "requirements.txt"
        if req_file.exists():
            filtered_lines = []
            for line in req_file.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                pkg_spec = stripped.lower()
                if pkg_spec.startswith("git+"):
                    # e.g. git+https://github.com/huggingface/transformers@...
                    if "transformers" in pkg_spec:
                        continue
                elif pkg_spec.startswith(_CONFLICTING_PREFIXES):
                    continue
                filtered_lines.append(line)

            filtered_req = FOLEY_REPO_DIR / "requirements.filtered.txt"
            filtered_req.write_text("\n".join(filtered_lines) + "\n")
            # Install into the FOLEY venv — NEVER into sys.executable / main env
            subprocess.run(
                [str(FOLEY_VENV_PY), "-m", "pip", "install", "--quiet",
                 "--disable-pip-version-check", "--root-user-action=ignore",
                 "--no-warn-conflicts", "--no-cache-dir",
                 "-r", str(filtered_req)],
                check=True, capture_output=True,
            )
        print("[AudioEngine] HunyuanVideo-Foley repo ready.")

    # --- HunyuanVideo-Foley model weights -----------------------------
    foley_ckpt = FOLEY_MODEL_DIR / "hunyuanvideo_foley_xl.pth"
    if not foley_ckpt.exists():
        print("[AudioEngine] Downloading HunyuanVideo-Foley XL weights...")
        FOLEY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="tencent/HunyuanVideo-Foley",
            filename="hunyuanvideo_foley_xl.pth",
            local_dir=str(FOLEY_MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("[AudioEngine] Foley main checkpoint downloaded.")

    # Always ensure auxiliary weights exist (they may be missing even if ckpt is present)
    for aux in ["synchformer_state_dict.pth", "vae_128d_48k.pth"]:
        aux_path = FOLEY_MODEL_DIR / aux
        if not aux_path.exists():
            try:
                print(f"[AudioEngine] Downloading auxiliary weight: {aux}")
                hf_hub_download(
                    repo_id="tencent/HunyuanVideo-Foley",
                    filename=aux,
                    local_dir=str(FOLEY_MODEL_DIR),
                    local_dir_use_symlinks=False,
                )
            except Exception as _e:
                print(f"[AudioEngine] Warning: could not download {aux}: {_e}")
    print("[AudioEngine] Foley weights ready.")

    _AUDIO_ENGINE_AVAILABLE = True
    print("[AudioEngine] Dual audio engine ready (F5-TTS + HunyuanVideo-Foley).")


F5_INFER_SCRIPT = Path(SCRIPT_DIR) / ".f5tts_infer_worker.py"

_F5_INFER_WORKER_SOURCE = '''\
"""Standalone F5-TTS inference worker. Run inside F5_VENV_PY only.
Args: <json_payload_file>
JSON keys: ref_file, gen_text, out_wav, ref_text (optional), speed (optional, default 1.0)
"""
import sys, json, os, pathlib, importlib.util, importlib.machinery, types as _types

# ---------------------------------------------------------------------------
# Stub wandb before any f5_tts import.
# f5_tts.model.trainer does `import wandb` at module level (training dep,
# not needed for inference). That triggers:
#   wandb -> pydantic_core -> typing_extensions.Sentinel (missing on old system)
# We replace wandb in sys.modules with a no-op stub BEFORE f5_tts is loaded.
#
# Critical: __spec__ must be a real ModuleSpec, NOT None.
# accelerate/tracking.py calls importlib.util.find_spec("wandb") which does:
#   if wandb.__spec__ is None: raise ValueError
# A ModuleSpec("wandb", None) satisfies find_spec without triggering a real import.
# ---------------------------------------------------------------------------
_wandb_stub = _types.ModuleType("wandb")
_wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
_wandb_stub.__version__ = "0.0.0"
_wandb_stub.init   = lambda *a, **kw: None
_wandb_stub.log    = lambda *a, **kw: None
_wandb_stub.finish = lambda *a, **kw: None
_wandb_stub.run    = None
for _mod in list(sys.modules.keys()):
    if _mod == "wandb" or _mod.startswith("wandb."):
        sys.modules.pop(_mod, None)
sys.modules["wandb"] = _wandb_stub
# ---------------------------------------------------------------------------

def find_f5_pkg_dir():
    spec = importlib.util.find_spec("f5_tts")
    if spec and spec.submodule_search_locations:
        return pathlib.Path(list(spec.submodule_search_locations)[0])
    import f5_tts
    if getattr(f5_tts, "__file__", None):
        return pathlib.Path(f5_tts.__file__).parent
    return pathlib.Path(list(f5_tts.__path__)[0])

def ensure_config():
    pkg_dir  = find_f5_pkg_dir()
    cfg_dir  = pkg_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "F5TTS_v1_Base.yaml"
    if not cfg_file.exists():
        # Exact structure required by api.py:
        #   model_cfg.model.backbone  -> class name under f5_tts.model.*
        #   model_cfg.model.arch      -> kwargs passed to model constructor
        #   model_cfg.model.mel_spec  -> mel spectrogram settings
        cfg_lines = [
            "model:",
            "  backbone: DiT",
            "  arch:",
            "    dim: 1024",
            "    depth: 22",
            "    heads: 16",
            "    ff_mult: 2",
            "    text_dim: 512",
            "    conv_layers: 4",
            "  mel_spec:",
            "    target_sample_rate: 24000",
            "    n_mel_channels: 100",
            "    hop_length: 256",
            "    win_length: 1024",
            "    n_fft: 1024",
            "    mel_spec_type: vocos",
        ]
        cfg_file.write_text(chr(10).join(cfg_lines) + chr(10))
        print(f"[F5Worker] Wrote {cfg_file}")
    return cfg_file

def main():
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        p = json.load(fh)

    ref_file = p["ref_file"]
    gen_text = p["gen_text"]
    out_wav  = p["out_wav"]
    ref_text = p.get("ref_text", "").strip()
    speed    = float(p.get("speed", 1.0))

    if not ref_text:
        try:
            import whisper
            result = whisper.load_model("base").transcribe(ref_file, language="en")
            ref_text = result.get("text", "").strip()
            print(f"[F5Worker] ref_text: {ref_text!r}")
        except Exception as e:
            print(f"[F5Worker] Whisper failed ({e}), using empty ref_text.")
            ref_text = ""

    ensure_config()

    from f5_tts.api import F5TTS
    model = F5TTS(model="F5TTS_v1_Base")
    model.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=out_wav,
        remove_silence=True,
        speed=speed,
    )

if __name__ == "__main__":
    main()
'''


def _run_f5tts(ref_file: str, gen_text: str, out_wav: str, speed: float = 1.0) -> bool:
    """Run F5-TTS voice cloning in its isolated venv via subprocess.

    Passes all arguments via a JSON temp file so spaces / special characters
    in gen_text are never mangled by shell argument splitting.

    speed < 1.0 slows the voice down; speed > 1.0 speeds it up (default 1.0).

    Returns True if out_wav was produced with nonzero size.
    """
    print(f"[LipSync→F5-TTS] Starting voice generation...")
    print(f"[LipSync→F5-TTS]   ref_file: {ref_file}")
    print(f"[LipSync→F5-TTS]   gen_text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
    print(f"[LipSync→F5-TTS]   out_wav: {out_wav}")
    print(f"[LipSync→F5-TTS]   speed: {speed}")
    
    if not F5_VENV_PY.exists():
        print(f"[LipSync→F5-TTS] ERROR: F5-TTS venv missing at {F5_VENV_PY}")
        return False
    
    if not os.path.exists(ref_file):
        print(f"[LipSync→F5-TTS] ERROR: Reference audio file not found: {ref_file}")
        return False
    
    ref_size = os.path.getsize(ref_file)
    print(f"[LipSync→F5-TTS] Reference audio size: {ref_size / 1024:.1f} KB")
    
    # Always rewrite the worker script so fixes take effect without a server restart
    F5_INFER_SCRIPT.write_text(_F5_INFER_WORKER_SOURCE)

    # Write payload to a temp JSON file — avoids ALL shell-splitting issues
    import tempfile
    payload = {"ref_file": ref_file, "gen_text": gen_text, "out_wav": out_wav, "speed": float(speed)}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(payload, tf)
        payload_path = tf.name
    
    print(f"[LipSync→F5-TTS] Payload written to: {payload_path}")

    try:
        cmd = [str(F5_VENV_PY), str(F5_INFER_SCRIPT), payload_path]
        print(f"[LipSync→F5-TTS] Running command: {' '.join(cmd)}")
        print(f"[LipSync→F5-TTS] Starting inference (timeout: 300s)...")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        print(f"[LipSync→F5-TTS] Process completed in {elapsed:.1f}s with return code {result.returncode}")
        
        if result.stdout.strip():
            print(f"[LipSync→F5-TTS] STDOUT: {result.stdout[-1000:]}")
        
        if result.returncode != 0:
            print(f"[LipSync→F5-TTS] ERROR: Process failed with return code {result.returncode}")
            print(f"[LipSync→F5-TTS] STDERR: {result.stderr[-2000:]}")
            return False
        
        if not os.path.exists(out_wav):
            print(f"[LipSync→F5-TTS] ERROR: Output WAV file was not created: {out_wav}")
            return False
        
        out_size = os.path.getsize(out_wav)
        print(f"[LipSync→F5-TTS] Output WAV size: {out_size / 1024:.1f} KB")
        
        if out_size == 0:
            print(f"[LipSync→F5-TTS] ERROR: Output WAV file is empty")
            return False
        
        print(f"[LipSync→F5-TTS] ✓ SUCCESS - Voice generation complete")
        return True
        
    except subprocess.TimeoutExpired:
        print("[LipSync→F5-TTS] ERROR: Process timed out after 300s")
        return False
    except Exception as e:
        print(f"[LipSync→F5-TTS] ERROR: Exception occurred: {e}")
        import traceback
        print(f"[LipSync→F5-TTS] Traceback:\n{traceback.format_exc()}")
        return False
    finally:
        try:
            os.unlink(payload_path)
            print(f"[LipSync→F5-TTS] Cleaned up payload file")
        except Exception as e:
            print(f"[LipSync→F5-TTS] Failed to cleanup payload (non-fatal): {e}")


def _run_foley(video_path: str, sfx_prompt: str, output_wav: str) -> bool:
    """
    Call HunyuanVideo-Foley's infer.py via subprocess, chaining for long videos.
    
    Foley can only generate ~15s at a time. For videos longer than 15s, split
    into chunks, generate audio per chunk, then concatenate.
    
    Returns True on success, False on failure.
    """
    foley_script = FOLEY_REPO_DIR / "infer.py"
    if not foley_script.exists():
        print(f"[AudioEngine] Foley infer.py not found at {foley_script}")
        return False

    # Get video duration
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        dur_str = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        video_duration = float(dur_str)
    except Exception:
        video_duration = 15.0

    out_dir = Path(output_wav).parent
    import torch as _torch
    gpu_id = 1 if _torch.cuda.device_count() >= 2 else 0
    max_chunk = 15.0

    # Single run if <= 15s
    if video_duration <= max_chunk:
        stem = Path(video_path).stem
        # Use FOLEY_VENV_PY — never sys.executable — so Foley deps stay isolated
        cmd = [
            str(FOLEY_VENV_PY), str(foley_script),
            "--model_path", str(FOLEY_MODEL_DIR),
            "--model_size", "xl",
            "--enable_offload",
            "--device", "cuda",
            "--gpu_id", str(gpu_id),
            "--single_video", video_path,
            "--single_prompt", sfx_prompt if sfx_prompt and sfx_prompt.strip() else "natural ambient sounds",
            "--output_dir", str(out_dir),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(FOLEY_REPO_DIR))
            if result.returncode != 0:
                print(f"[AudioEngine] Foley stderr: {result.stderr[-2000:]}")
                return False
            expected = out_dir / f"{stem}_generated.wav"
            if expected.exists():
                expected.rename(output_wav)
                return True
            for f in sorted(out_dir.glob("*.wav")):
                if f.name != Path(output_wav).name:
                    f.rename(output_wav)
                    return True
            print("[AudioEngine] Foley ran but output .wav not found.")
            return False
        except subprocess.TimeoutExpired:
            print("[AudioEngine] Foley timed out.")
            return False
        except Exception as e:
            print(f"[AudioEngine] Foley subprocess error: {e}")
            return False

    # Chain for long videos
    print(f"[AudioEngine] Foley chaining {video_duration:.1f}s video into {max_chunk}s chunks...")
    import tempfile
    chunk_wavs = []
    chunk_dir = tempfile.mkdtemp(prefix="foley_chain_")
    try:
        n_chunks = int(np.ceil(video_duration / max_chunk))
        for i in range(n_chunks):
            start_t = i * max_chunk
            chunk_vid = os.path.join(chunk_dir, f"chunk_{i}.mp4")
            chunk_wav_expected = os.path.join(chunk_dir, f"chunk_{i}_generated.wav")
            
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-ss", str(start_t),
                "-t", str(max_chunk), "-c", "copy", chunk_vid
            ], capture_output=True, timeout=60, check=True)
            
            # Use FOLEY_VENV_PY — never sys.executable — so Foley deps stay isolated
            cmd = [
                str(FOLEY_VENV_PY), str(foley_script),
                "--model_path", str(FOLEY_MODEL_DIR),
                "--model_size", "xl",
                "--enable_offload",
                "--device", "cuda",
                "--gpu_id", str(gpu_id),
                "--single_video", chunk_vid,
                "--single_prompt", sfx_prompt if sfx_prompt and sfx_prompt.strip() else "natural ambient sounds",
                "--output_dir", chunk_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(FOLEY_REPO_DIR))
            if result.returncode == 0 and os.path.exists(chunk_wav_expected):
                chunk_wavs.append(chunk_wav_expected)
            else:
                print(f"[AudioEngine] Foley chunk {i} failed, skipping.")
        
        if not chunk_wavs:
            return False
        
        if len(chunk_wavs) == 1:
            shutil.copy(chunk_wavs[0], output_wav)
        else:
            concat_list = os.path.join(chunk_dir, "concat.txt")
            with open(concat_list, "w") as f:
                for w in chunk_wavs:
                    f.write(f"file '{w}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list, "-c", "copy", output_wav
            ], capture_output=True, timeout=60, check=True)
        
        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except Exception as e:
        print(f"[AudioEngine] Foley chaining error: {e}")
        return False
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)



def add_audio_to_video(
    video_buf: bytes,
    sfx_prompt: str,
    duration_sec: float,
    audio_negative_prompt: str = "",
    ref_audio_path: str | None = None,
    dialogue_text: str = "",
    voice_speed: float = 1.0,
) -> bytes:
    """Dual-engine audio synthesis for a generated video.

    Branch A (Foley)  — HunyuanVideo-Foley reads the video frames and
                        generates synchronized physical sound effects / ambience.
    Branch B (Voice)  — F5-TTS clones the supplied reference voice and speaks
                        the dialogue_text. Skipped when either field is empty.

    Both branches produce a .wav file; FFmpeg mixes them and muxes into the
    original video container. Falls back to returning the original bytes on any error.

    Accepts and returns raw MP4 bytes. Temp files are created in /dev/shm/newgen
    (RAM disk) and deleted before this function returns.
    """
    if not _AUDIO_ENGINE_AVAILABLE:
        return video_buf

    # Release Wan VRAM before loading Foley/F5-TTS models.
    # After video generation the Wan transformer + VAE are still resident in
    # VRAM. On a single GPU that leaves almost no room for Foley's SigLIP /
    # HunyuanVideo-Foley weights, causing an OOM inside the subprocess.
    # PyTorch's allocator only returns pages to CUDA on an explicit cache
    # flush, so we do that here before spawning either audio engine.
    # gc.collect() first ensures any Python-side tensor references are dropped
    # so empty_cache() can actually release the backing memory.
    try:
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    tmp_dir = Path("/dev/shm/newgen")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]

    vid_tmp      = str(tmp_dir / f"ae_in_{token}.mp4")
    foley_wav    = str(tmp_dir / f"ae_foley_{token}.wav")
    voice_wav    = str(tmp_dir / f"ae_voice_{token}.wav")
    out_vid_tmp  = str(tmp_dir / f"ae_out_{token}.mp4")
    _tmps = [vid_tmp, foley_wav, voice_wav, out_vid_tmp]

    def _cleanup():
        for p in _tmps:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

    try:
        # Write input video to RAM disk so both engines can read it
        with open(vid_tmp, "wb") as fh:
            fh.write(video_buf)

        has_foley = False
        has_voice = False

        # ---- Branch A: Foley / SFX -----------------------------------
        print("[AudioEngine] Running HunyuanVideo-Foley...")
        has_foley = _run_foley(vid_tmp, sfx_prompt, foley_wav)
        if has_foley:
            print("[AudioEngine] Foley track generated.")
        else:
            print("[AudioEngine] Foley failed — continuing without SFX track.")

        # ---- Branch B: Voice cloning ---------------------------------
        # gr.File (Gradio 3.x) returns an object with .name; gr.Audio returns a path string.
        # Normalise to a plain string path.
        _ref_path = None
        if ref_audio_path is not None:
            if hasattr(ref_audio_path, "name"):
                _ref_path = ref_audio_path.name
            else:
                _ref_path = str(ref_audio_path) if ref_audio_path else None

        voice_active = (
            _ref_path
            and os.path.exists(_ref_path)
            and dialogue_text
            and dialogue_text.strip()
        )
        if voice_active:
            try:
                print(f"[AudioEngine] Running F5-TTS voice clone... (speed={voice_speed:.2f})")
                has_voice = _run_f5tts(_ref_path, dialogue_text.strip(), voice_wav, speed=voice_speed)
                if has_voice:
                    print("[AudioEngine] Voice track generated.")
                else:
                    print("[AudioEngine] F5-TTS produced no output — continuing without voice track.")
            except Exception as ve:
                print(f"[AudioEngine] F5-TTS failed: {ve}")
                has_voice = False

        # ---- Detect whether the input video already carries audio --------
        # (e.g. dialogue MuseTalk just muxed in during lip-sync). If so,
        # it must be preserved and mixed with any new tracks below — not
        # silently dropped, which is what the old foley/voice-only ffmpeg
        # commands did (they only mapped the NEW track's audio, discarding
        # stream 0's existing audio entirely).
        _probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", vid_tmp],
            capture_output=True, text=True,
        )
        has_input_audio = bool(_probe.stdout.strip())

        # ---- Neither branch succeeded --------------------------------
        if not has_foley and not has_voice:
            if has_input_audio:
                # Nothing new to add, but the video already has audio (e.g.
                # dialogue baked in by MuseTalk) — return it as-is rather
                # than reporting failure and returning the original bytes
                # unchanged (which would still be fine here, but the log
                # message below would be misleading).
                print("[AudioEngine] No new audio to add — video already has an audio track.")
                _cleanup()
                with open(vid_tmp, "rb") as fh:
                    return fh.read()
            print("[AudioEngine] Both audio branches failed — returning silent video.")
            _cleanup()
            return video_buf

        # ---- FFmpeg mix & mux ----------------------------------------
        # Build the list of audio sources to mix: the video's own existing
        # audio (if any), the new voice track (if any), the new foley track
        # (if any). Always mix rather than blindly overwrite.
        extra_inputs = []      # extra -i args, in order, after vid_tmp
        filter_parts = []      # per-source volume/label filters
        mix_labels = []        # labels to feed into amix
        next_in = 1

        if has_input_audio:
            filter_parts.append("[0:a]volume=1.0[existing]")
            mix_labels.append("[existing]")
        if has_voice:
            extra_inputs.append(voice_wav)
            filter_parts.append(f"[{next_in}:a]volume=1.0[voice]")
            mix_labels.append("[voice]")
            next_in += 1
        if has_foley:
            extra_inputs.append(foley_wav)
            filter_parts.append(f"[{next_in}:a]volume=0.65[foley]")
            mix_labels.append("[foley]")
            next_in += 1

        ffmpeg_cmd = ["ffmpeg", "-y", "-i", vid_tmp]
        for _extra in extra_inputs:
            ffmpeg_cmd += ["-i", _extra]

        if len(mix_labels) == 1:
            # Only one audio source in play — no amix needed, just relabel it.
            filter_complex = ";".join(filter_parts) + f";{mix_labels[0]}anull[aout]"
        else:
            filter_complex = (
                ";".join(filter_parts)
                + f";{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest[aout]"
            )

        ffmpeg_cmd += [
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            out_vid_tmp,
        ]

        mix_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
        if mix_result.returncode != 0:
            print(f"[AudioEngine] FFmpeg mix failed: {mix_result.stderr[-1000:]}")
            _cleanup()
            return video_buf

        with open(out_vid_tmp, "rb") as fh:
            result_bytes = fh.read()

        _cleanup()
        return result_bytes

    except Exception as e:
        import traceback as _tb
        print(f"[AudioEngine] add_audio_to_video failed: {e}\n{_tb.format_exc()}")
        _cleanup()
        return video_buf


_ensure_audio_engines()


# ── MuseTalk (lip-sync post-processor) ───────────────────────────────────────
# Replaces the old LatentSync integration. Same task (video + audio -> lip
# synced video), same isolated-venv/subprocess architecture, but MuseTalk's
# real-time latent-inpainting UNet produces sharper mouth/teeth detail than
# LatentSync's diffusion UNet and runs several times faster per clip.
MUSETALK_DIR          = Path(SCRIPT_DIR) / "MuseTalk"
MUSETALK_MODELS_DIR   = MUSETALK_DIR / "models"
MUSETALK_UNET_CKPT    = MUSETALK_MODELS_DIR / "musetalk" / "pytorch_model.bin"
MUSETALK_VAE_CKPT     = MUSETALK_MODELS_DIR / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin"
MUSETALK_WHISPER_CKPT = MUSETALK_MODELS_DIR / "whisper" / "tiny.pt"

_musetalk_ready = False   # set True once install confirmed
_musetalk_lock  = threading.Lock()


def _ensure_musetalk():
    """Clone MuseTalk, create its isolated venv, install the OpenMMLab stack,
    and download all model weights.

    MuseTalk's deps (mmcv, mmdet, mmpose, its own torch copy, etc.) are
    installed ONLY into the isolated .musetalk-venv — they NEVER touch the
    system Python, the system torch, or the app venv. All inference is done
    by spawning MUSETALK_VENV_PY as a subprocess (see _run_musetalk).

    Key design decisions:
      • No --system-site-packages: the venv is fully self-contained.
      • torch/torchvision/torchaudio are installed inside the venv first, so
        `mim install mmcv` auto-selects a prebuilt wheel matching THIS venv's
        torch/CUDA build rather than the system torch 2.8 dev build.
      • All model weights (UNet, VAE, whisper, dwpose, face-parse) are pulled
        in one shot via huggingface_hub.snapshot_download of TMElyralab/MuseTalk
        into models/, matching the repo's documented layout — this avoids
        depending on the exact contents of the repo's own download_weights.sh,
        which has changed across revisions.
    """
    global _musetalk_ready
    with _musetalk_lock:
        if _musetalk_ready:
            return True
        try:
            # --- Clone repo -----------------------------------------------
            if not MUSETALK_DIR.exists():
                print("[LipSync] Cloning MuseTalk...")
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/TMElyralab/MuseTalk.git",
                     str(MUSETALK_DIR)],
                    check=True, capture_output=True,
                )
                print("[LipSync] MuseTalk cloned.")

            # --- Create isolated venv ------------------------------------
            _MT_PIP_QUIET = [
                "--quiet", "--disable-pip-version-check",
                "--root-user-action=ignore", "--no-warn-conflicts", "--no-cache-dir",
            ]
            # pip uses TMPDIR for build isolation; /dev/shm is often noexec.
            # CRITICAL: strip PYTHONPATH. run_app.sh exports PYTHONPATH
            # pointing at the SYSTEM site-packages so the app venv can see
            # system torch. That same env var leaks into every subprocess
            # spawned here (os.environ is inherited), which means any pip
            # build step in THIS venv silently falls back to importing the
            # system torch instead of the venv's own isolated copy the
            # instant the venv's own torch is missing or not yet installed.
            # That's what caused the "Unknown CUDA arch (9.0,12.0)" mmcv
            # build failure: it compiled against the system's older torch
            # dev build, which doesn't recognize Blackwell (compute 12.0).
            _mt_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            _mt_env["TMPDIR"] = "/tmp"

            _fresh_venv = not MUSETALK_VENV_PY.exists()

            if _fresh_venv:
                print("[LipSync] Creating isolated venv for MuseTalk ...")
                subprocess.run(
                    [sys.executable, "-m", "venv", str(MUSETALK_VENV_DIR)],
                    check=True,
                )

            # BUG 1 FIX: Always upgrade pip/setuptools/wheel as the VERY FIRST
            # step — before torch and everything else — both on freshly created
            # venvs and on every subsequent startup to self-heal existing venvs.
            # Python 3.12's `venv` stopped auto-installing setuptools/wheel, so
            # ANY source build (mmcv, xtcocotools, etc.) fails immediately with:
            #   BackendUnavailable: Cannot import 'setuptools.build_meta'
            # Running this unconditionally is cheap (pip detects "already up to
            # date" in <1s) and repairs venvs created before this fix existed
            # without needing a manual `rm -rf .musetalk-venv`.
            # NOTE: setuptools<81 is intentional — setuptools 81 removed
            # pkg_resources, which the OpenMMLab stack still depends on.
            print("[LipSync] Upgrading pip/setuptools/wheel in MuseTalk venv ...")
            subprocess.run(
                [str(MUSETALK_VENV_PY), "-m", "pip", "install"] +
                _MT_PIP_QUIET + ["--upgrade", "pip", "setuptools<81", "wheel"],
                check=False, capture_output=True, env=_mt_env,
            )

            # Always verify torch actually imports inside THIS venv — don't
            # gate the install on "venv is newly created". A venv that
            # already exists from a prior partial/failed run (as here) will
            # otherwise never get torch installed, ever.
            _needs_torch = True
            if not _fresh_venv:
                _torch_check = subprocess.run(
                    [str(MUSETALK_VENV_PY), "-c", "import torch"],
                    capture_output=True, env=_mt_env,
                )
                _needs_torch = _torch_check.returncode != 0

            if _needs_torch:
                # Install torch inside the MuseTalk venv (separate copy,
                # does NOT affect the system torch 2.8 dev build). This must
                # happen BEFORE mim install mmcv below, so mim can detect
                # this venv's torch/CUDA version and fetch a matching wheel.
                print("[LipSync] Installing torch into MuseTalk venv "
                      "(cu128 wheel index, needed for Blackwell/sm_120 support) ...")
                subprocess.run(
                    [str(MUSETALK_VENV_PY), "-m", "pip", "install"] +
                    _MT_PIP_QUIET +
                    ["--index-url", "https://download.pytorch.org/whl/cu128",
                     "torch", "torchvision", "torchaudio"],
                    check=True, capture_output=True, env=_mt_env,
                )

            # --- Install MuseTalk's own requirements.txt -------------------
            # MuseTalk's upstream requirements.txt pins tensorflow==2.12.0
            # and numpy==1.23.5. Neither has a Python 3.12 wheel (TensorFlow
            # added 3.12 support in 2.16+, NumPy in 1.26+), so pip falls back
            # to building both from source. That source build imports
            # setuptools.build_meta -> pkg_resources, which calls
            # `register_finder(pkgutil.ImpImporter, ...)` — an attribute
            # pkgutil dropped in Python 3.12 — crashing with
            # "module 'pkgutil' has no attribute 'ImpImporter'". Confirmed
            # from the actual traceback, not a guess.
            #
            # Fix: relax ONLY those two exact pins to unconstrained versions
            # so pip resolves prebuilt cp312 wheels instead of source-
            # building 2023-era releases. Every other pin is left untouched.
            req_file = MUSETALK_DIR / "requirements.txt"
            if req_file.exists():
                _req_text = req_file.read_text()
                _req_lines = []
                for _line in _req_text.splitlines():
                    _stripped = _line.strip()
                    if _stripped.startswith("tensorflow==") or _stripped.startswith("tensorboard=="):
                        # Relax the pin silently — no cp312 wheel for the exact version
                        _req_lines.append(_stripped.split("==")[0])
                    elif _stripped.startswith("numpy=="):
                        # Replace with a version that pip can actually resolve on py3.12
                        # (numpy==1.19.5 source build fails on py3.12 — pip will pick
                        # the best available prebuilt wheel with no pin)
                        _req_lines.append("numpy")
                        # Skip numpy==1.19.5 silently; force it via --force-reinstall later
                    else:
                        _req_lines.append(_line)
                _patched_req_file = MUSETALK_VENV_DIR / "requirements.cp312.txt"
                _patched_req_file.write_text("\n".join(_req_lines) + "\n")
                # --no-build-isolation: reuse the venv's own already-upgraded
                # setuptools/wheel for any source builds, instead of spawning
                # an ephemeral isolated build environment that doesn't have
                # setuptools at all (causing BackendUnavailable on py3.12).
                # capture_output=True silences all pip noise (numpy build
                # warnings, metadata-generation-failed errors, etc.) that
                # would otherwise spam the terminal on every startup.
                subprocess.run(
                    [str(MUSETALK_VENV_PY), "-m", "pip", "install",
                     "--no-build-isolation",
                     "-r", str(_patched_req_file)] + _MT_PIP_QUIET,
                    check=False, capture_output=True, env=_mt_env,
                )

            # --- MuseTalk's bundled whisper (encoder-only) ------------------
            # Only pip-install it if it's actually a real installable package
            # (setup.py/pyproject.toml present) — on current repo revisions
            # this directory is plain source with no build metadata, so
            # `pip install --editable` fails with "does not appear to be a
            # Python project". In that case it needs no install at all: we
            # always invoke MuseTalk via `-m scripts.inference` with cwd set
            # to the repo root, which puts musetalk/ on sys.path as a normal
            # subpackage already.
            whisper_pkg_dir = MUSETALK_DIR / "musetalk" / "whisper"
            _whisper_installable = (
                (whisper_pkg_dir / "setup.py").exists()
                or (whisper_pkg_dir / "pyproject.toml").exists()
            )
            if whisper_pkg_dir.exists() and _whisper_installable:
                subprocess.run(
                    [str(MUSETALK_VENV_PY), "-m", "pip", "install",
                     "--editable", str(whisper_pkg_dir)] + _MT_PIP_QUIET,
                    check=False, env=_mt_env,
                )
            elif whisper_pkg_dir.exists():
                print("[LipSync] musetalk/whisper has no setup.py/pyproject.toml — "
                      "skipping pip install, it's used via sys.path instead.")

            # --- OpenMMLab stack via plain pip (NOT openmim) -----------------
            # `mim` was dropped entirely. Both failure modes we hit
            # ("No module named 'pkg_resources'" and "No module named
            # 'pip'") come from mim's own internal subprocess probing
            # script crashing — not from mmcv/mmpose themselves. mim spawns
            # a throwaway `python -c "..."` snippet to detect the venv's
            # torch/CUDA build, and that snippet assumes pkg_resources /
            # pip are importable in ways that don't hold up on newer
            # setuptools or in this venv's isolated build environment.
            #
            # On top of that, this venv's torch is a normal PyPI torch
            # (installed a few lines up), but if it's ever bumped to a
            # bleeding-edge dev build there is no prebuilt mmcv wheel for it
            # anyway — mim would fall back to a from-source build regardless.
            # So we just build from source ourselves with
            # `pip --no-build-isolation`, which reuses this venv's own
            # already-installed pip/setuptools/torch for the build instead
            # of mim's flaky ephemeral probe.
            #
            # mmcv needs its CUDA ops compiled against the venv's torch;
            # MMCV_WITH_OPS=1 enables that (mmcv builds CPU-only otherwise).
            #
            # TORCH_CUDA_ARCH_LIST fix: torch cu128 wheels include arch 12.0
            # (Blackwell/sm_120) in their arch list, but mmcv's cpp_extension.py
            # raises "Unknown CUDA arch (9.0,12.0)" because mmcv's source
            # predates Blackwell support. Capping at "8.0;8.6;8.9;9.0" covers
            # all real GPUs this stack runs on and avoids the crash.
            _mmcv_env = {
                **_mt_env,
                "MMCV_WITH_OPS": "1",
                # Restrict CUDA arches to those mmcv's cpp_extension.py knows.
                # arch 12.0 (Blackwell/sm_120) was added to PyTorch cu128 but
                # is not yet in mmcv's arch table — including it causes a hard
                # ValueError crash during wheel build.
                "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
            }

            # ------------------------------------------------------------------
            # xtcocotools — the unfixable dependency.
            #
            # xtcocotools is a C extension that wraps COCO's _mask.c.  Its
            # source build ALWAYS fails inside the MuseTalk venv because:
            #   • pip's build isolation injects a binary-only numpy (no headers)
            #     into its ephemeral build env, so gcc cannot find
            #     numpy/arrayobject.h, which _mask.c #includes.
            #   • --no-build-isolation doesn't help because mmpose lists
            #     xtcocotools in its own install_requires, so pip re-resolves
            #     and tries to build it again as a *dependency*, applying full
            #     isolation regardless of the flag we pass to the mmpose install.
            #
            # The definitive fix is to make pip believe xtcocotools is already
            # installed so it never attempts the build at all — not for the
            # direct install, and not when mmpose pulls it as a dependency.
            # We do this by:
            #   1. Installing pycocotools (binary wheel, cp312, x86-64) which
            #      provides the identical C extension under a different name.
            #   2. Planting a minimal PEP 566 dist-info directory for
            #      xtcocotools in the venv's site-packages.  pip reads these
            #      .dist-info directories to determine what is installed.
            #      Once the record exists, pip's resolver sees xtcocotools as
            #      "already satisfied" and never queues it for a build —
            #      regardless of whether it appears as a direct install target
            #      or as a transitive dependency of mmpose/mmdet.
            #   3. Pointing the fake record's top_level.txt at pycocotools so
            #      any `import xtcocotools` that mmpose might attempt is caught
            #      by the sys.modules shim we inject below.
            #   4. Adding a sys.modules alias xtcocotools -> pycocotools inside
            #      the running Python process (belt-and-suspenders: this only
            #      matters if someone imports xtcocotools at runtime, not at
            #      install time, but it costs nothing).
            # ------------------------------------------------------------------

            # Step 1 — install pycocotools (binary wheel, no compile needed).
            _r_pycoco = subprocess.run(
                [str(MUSETALK_VENV_PY), "-m", "pip", "install"] +
                _MT_PIP_QUIET + ["pycocotools"],
                capture_output=True, text=True, env=_mt_env,
            )
            if _r_pycoco.returncode == 0:
                print("[LipSync] pycocotools installed OK.")
            else:
                print(f"[LipSync] pycocotools install warning: {_r_pycoco.stderr[-200:]}")

            # Step 2 — discover the venv's site-packages path.
            _sp_result = subprocess.run(
                [str(MUSETALK_VENV_PY), "-c",
                 "import sysconfig; print(sysconfig.get_path('purelib'))"],
                capture_output=True, text=True, env=_mt_env,
            )
            _mt_site = _sp_result.stdout.strip()

            # Step 3 — plant a fake xtcocotools dist-info AND a minimal stub
            # wheel so pip's resolver sees xtcocotools as already satisfied,
            # even when it appears as a transitive dependency of mmpose/mmdet
            # (where --no-build-isolation does NOT prevent pip from triggering
            # an isolated xtcocotools source build).
            #
            # BUG 2 FIX: The original stub only wrote dist-info files directly
            # into site-packages.  That satisfies pip's "is it installed?"
            # check for the direct-install path, but when mmpose/mmdet list
            # xtcocotools in their install_requires, pip re-resolves ALL
            # dependencies from scratch using its internal dependency resolver
            # (PEP 517 backend), which reads Direct-URL and dist-info RECORD
            # more strictly — and the empty RECORD caused it to treat the stub
            # as "corrupt" and queue a fresh source build anyway.
            #
            # The reliable fix is two-pronged:
            #   A. A proper .dist-info with a non-empty RECORD so pip's resolver
            #      marks it as a valid installation (not corrupt).
            #   B. A constraints file passed to every mmpose/mmdet pip install
            #      via -c, hard-pinning xtcocotools==1.3.  pip honours -c
            #      constraints even for transitive deps resolved during
            #      dependency-chain traversal, which prevents a fresh build
            #      even when mmpose declares xtcocotools in install_requires.
            _xt_constraints_file = MUSETALK_VENV_DIR / "xtcocotools-constraints.txt"
            _xt_constraints_file.write_text("xtcocotools==1.3\n", encoding="utf-8")

            if _mt_site:
                _xt_dist = Path(_mt_site) / "xtcocotools-1.3.dist-info"
                _xt_dist.mkdir(exist_ok=True)
                (_xt_dist / "METADATA").write_text(
                    "Metadata-Version: 2.1\n"
                    "Name: xtcocotools\n"
                    "Version: 1.3\n"
                    "Summary: Fake stub — satisfied by pycocotools (same C extension)\n"
                    "Requires-Dist: pycocotools\n",
                    encoding="utf-8",
                )
                (_xt_dist / "INSTALLER").write_text("pip\n", encoding="utf-8")
                # Non-empty RECORD: pip's dependency resolver rejects an empty
                # RECORD as "corrupt install" and re-queues a fresh build.
                # Listing the RECORD file itself (standard practice for
                # dist-info-only packages) marks the installation as valid.
                _record_entry = (
                    f"xtcocotools-1.3.dist-info/RECORD,,\n"
                    f"xtcocotools-1.3.dist-info/METADATA,,\n"
                    f"xtcocotools-1.3.dist-info/INSTALLER,,\n"
                    f"xtcocotools-1.3.dist-info/WHEEL,,\n"
                    f"xtcocotools-1.3.dist-info/top_level.txt,,\n"
                )
                (_xt_dist / "RECORD").write_text(_record_entry, encoding="utf-8")
                # top_level tells importlib where to find the package.
                (_xt_dist / "top_level.txt").write_text("pycocotools\n", encoding="utf-8")
                # WHEEL file satisfies pip's wheel-format check.
                (_xt_dist / "WHEEL").write_text(
                    "Wheel-Version: 1.0\n"
                    "Generator: newgen-compat-shim\n"
                    "Root-Is-Purelib: true\n"
                    "Tag: py3-none-any\n",
                    encoding="utf-8",
                )
                # direct_url.json marks this as a direct installation so pip
                # does not attempt to re-resolve it from PyPI.
                import json as _json
                (_xt_dist / "direct_url.json").write_text(
                    _json.dumps({"url": "https://files.pythonhosted.org/packages/xtcocotools-stub",
                                 "dir_info": {"editable": False}}),
                    encoding="utf-8",
                )
                print("[LipSync] xtcocotools dist-info stub written (non-empty RECORD) — "
                      "pip resolver will treat it as already installed.")
            else:
                print("[LipSync] Warning: could not determine venv site-packages path; "
                      "xtcocotools stub not written (mmpose install may still fail).")

            # Step 4 — sys.modules alias in THIS process so any runtime
            # `import xtcocotools` by mmpose resolves to pycocotools.
            # (Belt-and-suspenders; the dist-info is what prevents the build.)
            _alias_script = (
                "import sys\n"
                "try:\n"
                "    import pycocotools as _pc\n"
                "    sys.modules.setdefault('xtcocotools', _pc)\n"
                "    # Also alias common sub-modules mmpose may import directly\n"
                "    for _sub in ['mask', 'coco', 'cocoeval']:\n"
                "        _full = f'pycocotools.{_sub}'\n"
                "        _xt   = f'xtcocotools.{_sub}'\n"
                "        try:\n"
                "            import importlib; _m = importlib.import_module(_full)\n"
                "            sys.modules.setdefault(_xt, _m)\n"
                "        except ImportError:\n"
                "            pass\n"
                "except ImportError:\n"
                "    pass\n"
            )
            subprocess.run(
                [str(MUSETALK_VENV_PY), "-c", _alias_script],
                capture_output=True, env=_mt_env,
            )

            # Check whether mmpose is already importable — if all four
            # OpenMMLab packages import cleanly, skip the install loop
            # entirely.  This avoids pip re-resolving (and re-attempting to
            # build) xtcocotools on an existing venv where everything already
            # works, even if the dist-info stub wasn't present on that run.
            _mmpose_check = subprocess.run(
                [str(MUSETALK_VENV_PY), "-c",
                 "import mmengine, mmcv, mmdet, mmpose; "
                 "print('mmpose', mmpose.__version__)"],
                capture_output=True, text=True, env=_mt_env,
            )
            if _mmpose_check.returncode == 0:
                print(f"[LipSync] OpenMMLab stack already installed "
                      f"({_mmpose_check.stdout.strip()}) — skipping install.")
            else:
                # Install mmengine, mmcv, mmdet normally (no xtcocotools dependency).
                # mmpose is installed with --no-deps to completely bypass its
                # xtcocotools install_requires — that's the only dep that can't
                # be resolved (version conflict) or built (C extension / no headers).
                # pycocotools (already installed above) provides the identical API.
                for _pkg in ["mmengine", "mmcv>=2.0.1", "mmdet>=3.1.0"]:
                    _r = subprocess.run(
                        [str(MUSETALK_VENV_PY), "-m", "pip", "install",
                         "--no-build-isolation"] + _MT_PIP_QUIET + [_pkg],
                        capture_output=True, text=True, env=_mmcv_env,
                    )
                    if _r.returncode != 0:
                        print(f"[LipSync] pip install {_pkg} failed (non-fatal): "
                              f"{_r.stderr[-500:]}")
                    else:
                        print(f"[LipSync] {_pkg} installed OK.")

                # mmpose: install without deps so pip never tries to resolve or
                # build xtcocotools (which has conflicting version pins across
                # mmpose releases AND requires numpy headers to compile from source).
                # All mmpose runtime deps (mmcv, mmdet, mmengine, pycocotools,
                # numpy, opencv-python, scipy, etc.) are already installed above.
                _r_mmpose = subprocess.run(
                    [str(MUSETALK_VENV_PY), "-m", "pip", "install",
                     "--no-build-isolation", "--no-deps"] + _MT_PIP_QUIET + ["mmpose>=1.1.0"],
                    capture_output=True, text=True, env=_mmcv_env,
                )
                if _r_mmpose.returncode != 0:
                    print(f"[LipSync] pip install mmpose>=1.1.0 failed (non-fatal): "
                          f"{_r_mmpose.stderr[-500:]}")
                else:
                    print("[LipSync] mmpose>=1.1.0 installed OK.")
            print("[LipSync] MuseTalk venv deps installed.")

            # --- Force numpy 1.19.5 for compatibility ------------------------
            # mmcv/mmdet install numpy 2.x as dependency. Force numpy 1.19.5
            # (last version with numpy.long) to ensure compatibility with old
            # OpenCV binaries, old scipy, and old diffusers in MuseTalk venv.
            subprocess.run(
                [str(MUSETALK_VENV_PY), "-m", "pip", "install", 
                 "--force-reinstall", "--no-deps", "numpy==1.19.5"],
                capture_output=True, text=True, timeout=60,
            )

            # --- huggingface_hub (with CLI extra) for weight download ------
            subprocess.run(
                [str(MUSETALK_VENV_PY), "-m", "pip", "install"] +
                _MT_PIP_QUIET + ["huggingface_hub[cli]<1.0"],
                check=False, env=_mt_env,
            )

            # --- Download all weights in one shot -------------------------
            # Covers musetalk/, musetalkV15/, dwpose/, face-parse-bisent/,
            # sd-vae-ft-mse/, whisper/ in a single snapshot_download, matching
            # the layout MuseTalk's own scripts expect under models/.
            MUSETALK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _weights_missing = (
                not MUSETALK_UNET_CKPT.exists() or MUSETALK_UNET_CKPT.stat().st_size < 5 * 1024 ** 2
                or not MUSETALK_VAE_CKPT.exists() or MUSETALK_VAE_CKPT.stat().st_size < 100 * 1024 ** 2
                or not MUSETALK_WHISPER_CKPT.exists() or MUSETALK_WHISPER_CKPT.stat().st_size < 50 * 1024 ** 2
            )
            if _weights_missing:
                print("[LipSync] Downloading MuseTalk model weights (several GB, one-time)...")
                _dl_script = (
                    "from huggingface_hub import snapshot_download\n"
                    "snapshot_download(repo_id='TMElyralab/MuseTalk', "
                    f"local_dir={str(MUSETALK_MODELS_DIR)!r})\n"
                )
                _r = subprocess.run(
                    [str(MUSETALK_VENV_PY), "-c", _dl_script],
                    capture_output=True, text=True, env=_mt_env, timeout=3600,
                )
                if _r.stdout.strip():
                    print(f"[LipSync] weights download stdout: {_r.stdout[-1500:]}")
                if _r.returncode != 0:
                    raise RuntimeError(f"weights download failed: {_r.stderr[-1500:]}")
                if not MUSETALK_UNET_CKPT.exists():
                    raise RuntimeError(
                        f"snapshot_download completed but {MUSETALK_UNET_CKPT} is still missing — "
                        "the upstream repo layout may have changed."
                    )
                print("[LipSync] MuseTalk weights ready.")

            _musetalk_ready = True
            return True
        except Exception as _e:
            print(f"[LipSync] _ensure_musetalk failed: {_e}")
            return False


def _run_musetalk(video_path: str, audio_path: str, output_path: str,
                   bbox_shift: int = 0, batch_size: int = 4) -> bool:
    """
    Run MuseTalk on (video_path + audio_path) → output_path.
    Returns True on success.

    bbox_shift: controls mouth openness (positive = more open). Same
                parameter MuseTalk's own docs recommend tuning per-subject.
    batch_size: frames processed per UNet batch. Higher = faster but more
                VRAM. 4 is a safe default; lower it if you see CUDA OOM.

    Blocks until MuseTalk is ready (downloading weights if needed). This
    ensures Lip-Synced Speaking mode always gets real MuseTalk output even
    when the first call happens before the background download thread
    finishes — previously the race caused silent fallback to no-lip-sync.
    """
    print(f"[LipSync→MuseTalk] Called with:")
    print(f"[LipSync→MuseTalk]   video_path: {video_path}")
    print(f"[LipSync→MuseTalk]   audio_path: {audio_path}")
    print(f"[LipSync→MuseTalk]   output_path: {output_path}")
    print(f"[LipSync→MuseTalk]   bbox_shift: {bbox_shift} (mouth openness)")
    print(f"[LipSync→MuseTalk]   batch_size: {batch_size}")
    
    # Verify input files exist and are not empty
    if not os.path.exists(video_path):
        print(f"[LipSync→MuseTalk] ERROR: Video file does not exist: {video_path}")
        return False
    if not os.path.exists(audio_path):
        print(f"[LipSync→MuseTalk] ERROR: Audio file does not exist: {audio_path}")
        return False
    
    video_size = os.path.getsize(video_path)
    audio_size = os.path.getsize(audio_path)
    print(f"[LipSync→MuseTalk] Video size: {video_size / 1024:.1f} KB")
    print(f"[LipSync→MuseTalk] Audio size: {audio_size / 1024:.1f} KB")
    
    if video_size == 0:
        print(f"[LipSync→MuseTalk] ERROR: Video file is empty")
        return False
    if audio_size == 0:
        print(f"[LipSync→MuseTalk] ERROR: Audio file is empty")
        return False
    
    # If MuseTalk isn't ready yet, wait for it (it may still be downloading
    # weights in the background thread kicked off at startup). Give it up to
    # 20 minutes — the initial snapshot_download is several GB.
    if not _musetalk_ready:
        print("[LipSync] MuseTalk not ready yet — waiting for background setup to complete (up to 20 min)...")
        _wait_start = time.time()
        _last_print = 0.0
        while not _musetalk_ready:
            _elapsed = time.time() - _wait_start
            if _elapsed > 1200:  # 20 minutes max
                print("[LipSync] MuseTalk setup timed out after 20 minutes — skipping lip sync.")
                return False
            if _elapsed - _last_print >= 30:
                print(f"[LipSync] Still waiting for MuseTalk setup... ({_elapsed:.0f}s elapsed)")
                _last_print = _elapsed
            time.sleep(2)
        print(f"[LipSync] MuseTalk is now ready — proceeding with lip sync.")

    if not _ensure_musetalk():
        print("[LipSync→MuseTalk] MuseTalk not available — skipping lip sync.")
        return False

    # --- Vendor correct numpy/diffusers for MuseTalk runtime ---------------
    # Even though we force-installed numpy 1.19.5 during venv setup, mmcv or
    # other deps might have reinstalled numpy 2.x. Create a vendor directory
    # with the EXACT versions we need and prepend to sys.path at runtime.
    VENDOR_DIR = Path(SCRIPT_DIR) / ".musetalk_vendor"
    VENDOR_DIR.mkdir(exist_ok=True)
    NUMPY_VENDOR = VENDOR_DIR / "numpy_1195"
    DIFFUSERS_VENDOR = VENDOR_DIR / "diffusers_0102"
    CV2_VENDOR = VENDOR_DIR / "cv2_compat"
    
    # Install numpy 1.19.5 into vendor dir if not present
    if not NUMPY_VENDOR.exists():
        subprocess.run(
            [str(MUSETALK_VENV_PY), "-m", "pip", "install", "--target", str(NUMPY_VENDOR), 
             "--no-deps", "--no-cache-dir", "numpy==1.19.5"],
            capture_output=True, timeout=120,
        )
    
    # Install diffusers 0.10.2 (last version with numpy.long support) into vendor dir
    if not DIFFUSERS_VENDOR.exists():
        subprocess.run(
            [str(MUSETALK_VENV_PY), "-m", "pip", "install", "--target", str(DIFFUSERS_VENDOR), 
             "--no-deps", "--no-cache-dir", "diffusers==0.10.2"],
            capture_output=True, timeout=120,
        )
    
    # Install opencv-python compiled for numpy 1.19 into vendor dir
    if not CV2_VENDOR.exists():
        subprocess.run(
            [str(MUSETALK_VENV_PY), "-m", "pip", "install", "--target", str(CV2_VENDOR), 
             "--no-deps", "--no-cache-dir", "opencv-python==4.5.5.64"],
            capture_output=True, timeout=120,
        )

    print(f"[LipSync→MuseTalk] Setting up job directory...")
    job_dir = Path(SCRIPT_DIR) / "tmp" / f"musetalk_job_{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "task.yaml"
    result_dir = job_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Hand-write the YAML — it's a trivially simple two-key mapping, so we
    # avoid adding a PyYAML dependency to the parent app venv just for this.
    yaml_content = (
        "task_0:\n"
        f"  video_path: {os.path.abspath(video_path)}\n"
        f"  audio_path: {os.path.abspath(audio_path)}\n"
    )
    cfg_path.write_text(yaml_content, encoding="utf-8")
    print(f"[LipSync→MuseTalk] Config written to {cfg_path}:")
    print(f"[LipSync→MuseTalk] {yaml_content}")

    _start_time = time.time()

    # Create wrapper script that forces vendored numpy/cv2/diffusers before MuseTalk runs
    wrapper_script = job_dir / "run_with_vendored_deps.py"
    wrapper_script.write_text(f"""#!/usr/bin/env python3
import sys
import os

# CRITICAL: Remove any pre-loaded numpy/cv2 modules that might have wrong versions
for mod in list(sys.modules.keys()):
    if mod.startswith('numpy') or mod.startswith('cv2'):
        del sys.modules[mod]

# CRITICAL: Prepend vendor dirs FIRST before ANY imports
sys.path.insert(0, r"{NUMPY_VENDOR}")
sys.path.insert(0, r"{CV2_VENDOR}")
sys.path.insert(0, r"{DIFFUSERS_VENDOR}")

# Change to MuseTalk directory so 'scripts' module is importable
os.chdir(r"{MUSETALK_DIR}")
# Add MuseTalk dir to path so scripts module is found
sys.path.insert(0, r"{MUSETALK_DIR}")

# Verify correct numpy is loaded
import numpy as np
if not hasattr(np, 'long'):
    print(f"ERROR: numpy {{np.__version__}} does not have 'long' attribute", file=sys.stderr)
    sys.exit(1)

# Now run MuseTalk with correct dependencies
import runpy
runpy.run_module("scripts.inference", run_name="__main__")
""")

    # Run via wrapper script instead of direct -m invocation
    _base_cmd = [
        str(MUSETALK_VENV_PY), str(wrapper_script),
        "--inference_config", str(cfg_path),
        "--bbox_shift", str(bbox_shift),
        "--batch_size", str(batch_size),
    ]
    
    print(f"[LipSync→MuseTalk] Running MuseTalk command:")
    print(f"[LipSync→MuseTalk] {' '.join(_base_cmd)}")
    print(f"[LipSync→MuseTalk] Working directory: {MUSETALK_DIR}")

    def _invoke(cmd):
        print(f"[LipSync→MuseTalk] Executing: {' '.join(cmd)}")
        return subprocess.run(
            cmd, cwd=str(MUSETALK_DIR), capture_output=True, text=True, timeout=900,
        )

    try:
        print(f"[LipSync→MuseTalk] Starting MuseTalk inference (timeout: 900s)...")
        result = _invoke(_base_cmd + ["--result_dir", str(result_dir)])
        
        elapsed = time.time() - _start_time
        print(f"[LipSync→MuseTalk] Process completed in {elapsed:.1f}s with return code {result.returncode}")
        
        if result.returncode != 0 and "unrecognized arguments" in (result.stderr or "").lower() \
                and "result_dir" in (result.stderr or "").lower():
            print("[LipSync→MuseTalk] This MuseTalk revision doesn't accept --result_dir — retrying without it.")
            result = _invoke(_base_cmd)
            elapsed = time.time() - _start_time
            print(f"[LipSync→MuseTalk] Retry completed in {elapsed:.1f}s with return code {result.returncode}")

        if result.stdout.strip():
            print(f"[LipSync→MuseTalk] ===== STDOUT START =====")
            print(f"{result.stdout[-3000:]}")
            print(f"[LipSync→MuseTalk] ===== STDOUT END =====")
        if result.returncode != 0:
            print(f"[LipSync→MuseTalk] ===== STDERR START (return code {result.returncode}) =====")
            print(f"{result.stderr[-2000:]}")
            print(f"[LipSync→MuseTalk] ===== STDERR END =====")
            return False

        # Find the newest .mp4 produced by this run, searching our dedicated
        # result_dir first, then falling back to the repo's default
        # results/ tree in case this revision ignores --result_dir entirely.
        print(f"[LipSync→MuseTalk] Searching for output video...")
        print(f"[LipSync→MuseTalk] Checking result_dir: {result_dir}")
        candidates = list(result_dir.rglob("*.mp4"))
        print(f"[LipSync→MuseTalk] Found {len(candidates)} .mp4 files in result_dir")
        
        if not candidates:
            default_results = MUSETALK_DIR / "results"
            print(f"[LipSync→MuseTalk] Checking default results: {default_results}")
            if default_results.exists():
                all_mp4s = list(default_results.rglob("*.mp4"))
                candidates = [
                    p for p in all_mp4s
                    if p.stat().st_mtime >= _start_time - 1
                ]
                print(f"[LipSync→MuseTalk] Found {len(candidates)} recent .mp4 files (out of {len(all_mp4s)} total)")
        
        if not candidates:
            print("[LipSync→MuseTalk] ERROR: MuseTalk ran but no output .mp4 was found.")
            return False
            
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[LipSync→MuseTalk] Selected output: {newest}")
        print(f"[LipSync→MuseTalk] Output size: {newest.stat().st_size / 1024:.1f} KB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(newest), output_path)
        print(f"[LipSync→MuseTalk] ✓ Copied to final output: {output_path}")
        print(f"[LipSync→MuseTalk] ✓ SUCCESS - Lip-sync processing complete")
        return True
    except subprocess.TimeoutExpired:
        print("[LipSync→MuseTalk] ERROR: Process timed out after 900s")
        return False
    except Exception as _e:
        print(f"[LipSync→MuseTalk] ERROR: Subprocess exception: {_e}")
        import traceback
        print(f"[LipSync→MuseTalk] Traceback:\n{traceback.format_exc()}")
        return False
    finally:
        print(f"[LipSync→MuseTalk] Cleaning up job directory: {job_dir}")
        shutil.rmtree(str(job_dir), ignore_errors=True)


def generate_lip_sync_video(
    ref_image, prompt, scene_mode_val,
    end_img, dur, res, fmul, qual, sd, rsd,
    audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
    neg_pt, esteps, eguid, fsa, fs,
    lipsync_bbox_shift, lipsync_batch_size,
    *lora_args,
):
    """
    Lip-Synced Speaking mode:
      1. Generate video normally (using dialogue_text in the motion prompt).
      2. Generate F5-TTS voice WAV from the dialogue script.
      3. Run MuseTalk to replace the mouth region with audio-driven synthesis.
      4. Mix Foley SFX on top via the normal add_audio_to_video path.
    """
    print(f"[LipSync] Starting Lip-Synced Speaking mode...")
    print(f"[LipSync] Parameters: bbox_shift={lipsync_bbox_shift}, batch_size={lipsync_batch_size}")
    
    # Step 1: generate base video (with audio=False so we control audio ourselves)
    print(f"[LipSync] Step 1/4: Generating base video...")
    base_result = _generate_video_blocking(
        ref_image,
        # Append speaking cue to prompt so Wan poses the mouth open
        (prompt or "") + ", subject speaking, mouth moving, talking, lips moving",
        MODE_KEEP,      # always keep scene; user can replace BG in a prior step
        end_img, dur, res, fmul, qual, sd, rsd,
        False,          # no audio yet — we handle it below
        audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
        neg_pt, esteps, eguid, fsa, fs,
        *lora_args,
    )
    if base_result is None or base_result[0] is None:
        print("[LipSync] ERROR: Base video generation returned None")
        return None, None, "Lip-sync: base video generation failed."

    base_video_path = base_result[1]   # raw file path (video_file component)
    if not base_video_path or not Path(base_video_path).exists():
        print(f"[LipSync] ERROR: Base video path missing or doesn't exist: {base_video_path}")
        return base_result[0], base_result[1], "Lip-sync: base video path missing."

    print(f"[LipSync] Base video generated successfully: {base_video_path}")
    
    # Ensure tmp directory exists for all intermediate files
    _lipsync_tmp = Path(SCRIPT_DIR) / "tmp"
    _lipsync_tmp.mkdir(parents=True, exist_ok=True)

    # Step 2: generate voice WAV (we need the raw WAV, not the mixed video)
    print(f"[LipSync] Step 2/4: Generating voice WAV...")
    print(f"[LipSync] Dialogue text provided: {repr(dlg_txt)}")
    print(f"[LipSync] Dialogue text stripped: {repr(dlg_txt.strip() if dlg_txt else '')}")
    print(f"[LipSync] Reference audio provided: {ref_aud is not None}")
    
    voice_wav = None
    _voice_wav_path = None
    if ref_aud and dlg_txt and dlg_txt.strip():
        _voice_wav_path = str(_lipsync_tmp / f"lipsync_voice_{uuid.uuid4().hex}.wav")
        _ref_path = ref_aud.name if hasattr(ref_aud, "name") else (str(ref_aud) if ref_aud else None)
        if not _ref_path and isinstance(ref_aud, dict):
            _ref_path = ref_aud.get("path") or ref_aud.get("name") or ref_aud.get("url")
        
        print(f"[LipSync] Reference audio path resolved to: {_ref_path}")
        
        if _ref_path and os.path.exists(str(_ref_path)):
            print(f"[LipSync] ✓ Reference audio file exists")
            print(f"[LipSync] Generating F5-TTS voice WAV for dialogue:")
            print(f"[LipSync]   Full text: {dlg_txt.strip()}")
            print(f"[LipSync]   Text length: {len(dlg_txt.strip())} characters")
            print(f"[LipSync]   Voice speed: {v_speed}")
            
            ok = _run_f5tts(str(_ref_path), dlg_txt.strip(), _voice_wav_path, speed=float(v_speed))
            
            if ok and Path(_voice_wav_path).exists() and Path(_voice_wav_path).stat().st_size > 0:
                voice_wav = _voice_wav_path
                wav_size_kb = Path(voice_wav).stat().st_size // 1024
                print(f"[LipSync] ✓ F5-TTS voice WAV ready: {wav_size_kb} KB")
                print(f"[LipSync] ✓ Voice WAV will be used for lip-sync: {voice_wav}")
            else:
                print("[LipSync] ✗ F5-TTS produced no output — will do lip sync without voice WAV.")
        else:
            print(f"[LipSync] ✗ Voice reference path not found or doesn't exist: {_ref_path!r}")
            print(f"[LipSync]   Skipping F5-TTS voice generation")
    else:
        print("[LipSync] ✗ Missing requirements for voice generation:")
        if not ref_aud:
            print("[LipSync]   - No voice reference audio provided")
        if not dlg_txt or not dlg_txt.strip():
            print(f"[LipSync]   - No dialogue text provided (got: {repr(dlg_txt)})")

    if voice_wav is None or not Path(voice_wav).exists():
        # Can't lip-sync without audio — fall back to normal mix
        print("[LipSync] No voice WAV available — falling back to normal audio mix.")
        print("[LipSync] Step 3/4: Skipped (no audio for lip-sync)")
        print("[LipSync] Step 4/4: Mixing normal audio...")
        with open(base_video_path, "rb") as _f:
            _video_bytes = _f.read()
        final = add_audio_to_video(
            _video_bytes, audio_pt, float(dur), audio_neg_pt, ref_aud, dlg_txt, float(v_speed),
        )
        out_path = str(_lipsync_tmp / f"lipsync_out_{uuid.uuid4().hex}.mp4")
        if final:
            with open(out_path, "wb") as _f:
                _f.write(final)
        else:
            shutil.copy(base_video_path, out_path)
        print(f"[LipSync] Completed with fallback audio. Output: {out_path}")
        return out_path, out_path, "Lip-sync skipped (no voice WAV) — normal audio mixed."

    # Step 3: MuseTalk — replace mouth region with audio-driven lip animation.
    # _run_musetalk will wait for MuseTalk weights to finish downloading if needed.
    print(f"[LipSync] Step 3/4: Running MuseTalk for lip animation...")
    ls_out = str(_lipsync_tmp / f"lipsync_ls_{uuid.uuid4().hex}.mp4")
    print(f"[LipSync] Running MuseTalk: video={base_video_path}, audio={voice_wav}")
    ls_ok = _run_musetalk(
        base_video_path, voice_wav, ls_out,
        bbox_shift=int(lipsync_bbox_shift),
        batch_size=int(lipsync_batch_size),
    )
    synced_video = ls_out if (ls_ok and Path(ls_out).exists()) else base_video_path

    if ls_ok:
        print(f"[LipSync] MuseTalk succeeded — lip-synced video at {ls_out}")
    else:
        print("[LipSync] MuseTalk failed or returned no output — using base video with F5-TTS audio.")

    # Step 4: mix audio on top of the (attempted) lip-synced video.
    print(f"[LipSync] Step 4/4: Mixing final audio tracks...")
    #   If MuseTalk succeeded, the voice is already embedded in the video
    #   frames/audio it produced — only add Foley SFX on top.
    #   If MuseTalk failed, synced_video is just the silent base video, so
    #   the F5-TTS voice_wav generated in Step 2 was NEVER embedded anywhere.
    #   Re-mix it here (along with Foley) instead of discarding it, or the
    #   dialogue never makes it into the output at all.
    with open(synced_video, "rb") as _f:
        _synced_bytes = _f.read()
    print(f"[LipSync] Read synced video: {len(_synced_bytes)} bytes")
    
    if ls_ok:
        # MuseTalk already baked the voice into the lip-synced video.
        # Only add Foley SFX on top — pass None/empty to skip F5-TTS re-run.
        print("[LipSync] Adding Foley SFX to lip-synced video...")
        final = add_audio_to_video(
            _synced_bytes, audio_pt, float(dur), audio_neg_pt,
            None, "",   # skip F5-TTS — voice already embedded by MuseTalk
            float(v_speed),
        )
    else:
        # MuseTalk never ran — the base video is silent. Mix the F5-TTS voice
        # (generated above) plus Foley SFX into the base video.
        print("[LipSync] Re-mixing F5-TTS voice + Foley into base video...")
        final = add_audio_to_video(
            _synced_bytes, audio_pt, float(dur), audio_neg_pt,
            ref_aud, dlg_txt,   # re-run F5-TTS inside add_audio_to_video
            float(v_speed),
        )
    out_path = str(_lipsync_tmp / f"lipsync_final_{uuid.uuid4().hex}.mp4")
    if final:
        with open(out_path, "wb") as _f:
            _f.write(final)
        print(f"[LipSync] Final output written: {Path(out_path).stat().st_size // 1024} KB")
    else:
        print("[LipSync] Audio mix returned empty — using synced video as-is.")
        shutil.copy(synced_video, out_path)

    # Clean up intermediate WAV (no longer needed)
    if _voice_wav_path and os.path.exists(_voice_wav_path):
        try:
            os.unlink(_voice_wav_path)
            print(f"[LipSync] Cleaned up intermediate WAV: {_voice_wav_path}")
        except Exception as e:
            print(f"[LipSync] Failed to clean up WAV (non-fatal): {e}")

    status = "✅ Lip-sync complete." if ls_ok else "⚠️ MuseTalk failed — dialogue voice + Foley mixed without lip sync."
    print(f"[LipSync] COMPLETE: {status}")
    print(f"[LipSync] Output file: {out_path}")
    return out_path, out_path, status


# kick off MuseTalk pre-download in the background so it's ready when needed
threading.Thread(target=_ensure_musetalk, daemon=True).start()


MAX_SEED = np.iinfo(np.int32).max


WAN_MODEL_REPO = "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING"
print(f"Video model: WAMU v2  Wan 2.2 I2V Lightning merge (NSFW-capable)")

LORA_DIR = Path(SCRIPT_DIR) / "loras"
LORA_DIR.mkdir(exist_ok=True)
LORA_CONFIG_FILE = LORA_DIR / "loras.json"

def load_lora_config():
    """Load LoRA configuration from JSON file."""
    if not LORA_CONFIG_FILE.exists():
        return {}
    try:
        with open(LORA_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def save_lora_config(config):
    """Save LoRA configuration to JSON file."""
    try:
        with open(LORA_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False

def download_lora_file(url, filename, progress_callback=None):
    """Download a LoRA file from URL with progress display."""
    import requests
    
    output_path = LORA_DIR / filename
    if output_path.exists():
        print(f" {filename} already exists")
        return True, "File already exists"
    
    try:
        print(f"\n Downloading {filename}")
        print(f"   Source: {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        chunk_size = len(chunk)
                        downloaded += chunk_size
                        pbar.update(chunk_size)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded / total_size)
        
        print(f" Downloaded: {filename}\n")
        return True, "Download complete"
    except Exception as e:
        print(f" Download failed: {e}\n")
        if output_path.exists():
            output_path.unlink()
        return False, str(e)

def check_lora_status(config):
    """Check which LoRAs are downloaded."""
    status = {}
    for lora_id, lora_info in config.items():
        high_exists = False
        low_exists = False
        
        if lora_info.get('high_filename'):
            high_path = LORA_DIR / lora_info['high_filename']
            high_exists = high_path.exists()
        
        if lora_info.get('low_filename'):
            low_path = LORA_DIR / lora_info['low_filename']
            low_exists = low_path.exists()
        
        status[lora_id] = {
            'high_exists': high_exists,
            'low_exists': low_exists,
            'high_downloadable': bool(lora_info.get('high_url')),
            'low_downloadable': bool(lora_info.get('low_url')),
        }
    
    return status

def discover_loras():
    """Auto-discover all .safetensors LoRA files and merge with config."""
    config = load_lora_config()
    loras = {}
    
    for lora_id, lora_info in config.items():
        loras[lora_id] = {
            'display_name': lora_info.get('display_name', lora_id),
            'description': lora_info.get('description', ''),
            'high': str(LORA_DIR / lora_info['high_filename']) if lora_info.get('high_filename') and (LORA_DIR / lora_info['high_filename']).exists() else None,
            'low': str(LORA_DIR / lora_info['low_filename']) if lora_info.get('low_filename') and (LORA_DIR / lora_info['low_filename']).exists() else None,
            'trigger_prompt': lora_info.get('trigger_prompt'),
            'trigger_aliases': lora_info.get('trigger_aliases', []),
            'prompt_mode': lora_info.get('prompt_mode', 'prepend'),
            'example_prompts': lora_info.get('example_prompts', []),
            'high_weight': lora_info.get('high_weight', 1.0),
            'low_weight': lora_info.get('low_weight', 1.0),
            'recommended_steps': lora_info.get('recommended_steps'),
            'recommended_flow_shift': lora_info.get('recommended_flow_shift'),
            'notes': lora_info.get('notes', ''),
            'config': lora_info,  # Keep full config for downloads
        }
    
    if LORA_DIR.exists():
        for lora_file in LORA_DIR.glob("*.safetensors"):
            filename = lora_file.name
            already_configured = False
            for lora_info in config.values():
                if filename in [lora_info.get('high_filename'), lora_info.get('low_filename')]:
                    already_configured = True
                    break
            
            if not already_configured:
                filename_lower = filename.lower()
                if "_high" in filename_lower or "_high_" in filename_lower:
                    noise_type = "high"
                elif "_low" in filename_lower or "_low_" in filename_lower:
                    noise_type = "low"
                else:
                    noise_type = "unknown"
                
                base_name = filename.replace('.safetensors', '')
                for suffix in ["_high_noise", "_low_noise", "_high", "_low"]:
                    if suffix in filename_lower:
                        idx = filename_lower.index(suffix)
                        base_name = filename[:idx]
                        break
                
                if base_name not in loras:
                    loras[base_name] = {
                        'display_name': base_name,
                        'description': 'Unconfigured LoRA',
                        'high': None,
                        'low': None,
                        'trigger_prompt': None,
                        'prompt_mode': 'append',
                        'example_prompts': [],
                        'high_weight': 1.0,
                        'low_weight': 1.0,
                        'notes': 'Add to loras.json for full configuration',
                        'config': {},
                    }
                
                loras[base_name][noise_type] = str(lora_file)
    
    return loras

_active_loras = {}  # {base_name: {"high": path, "low": path}}

def _wan_flat_key_to_module_path(flat_key: str) -> str | None:  # unused — kept for reference
    """
    Convert a Wan-native flat LoRA key to a dotted transformer parameter path.

    Wan LoRA files encode the full module path in a flat underscore-separated
    prefix.  For example:
        lora_unet_blocks_0_self_attn_q   →   blocks.0.self_attn.q
        lora_unet_blocks_27_ffn_0        →   blocks.27.ffn.0
        lora_unet_blocks_3_cross_attn_k  →   blocks.3.cross_attn.k

    The mapping rules:
      - Strip leading "lora_unet_" (or "lora_" alone if no "unet_").
      - Replace "_blocks_N_" with ".blocks.N." (N is an integer).
      - The remainder maps component names: self_attn, cross_attn, ffn stay as-is
        but use dots; the trailing single-letter (q/k/v/o) stays as-is.
      - ffn layers: ffn_0 → ffn.0, ffn_2 → ffn.2

    Returns a dotted path string, or None if the key cannot be decoded.
    """
    import re

    if flat_key.startswith("lora_unet_"):
        rest = flat_key[len("lora_unet_"):]
    elif flat_key.startswith("lora_"):
        rest = flat_key[len("lora_"):]
    else:
        return None

    m = re.match(r'^blocks_(\d+)_(.+)$', rest)
    if not m:
        return None

    block_idx = m.group(1)
    layer_part = m.group(2)  # e.g. "self_attn_q", "cross_attn_k", "ffn_0"

    attn_m = re.match(r'^(self_attn|cross_attn)_([qkvo])$', layer_part)
    if attn_m:
        attn_type = attn_m.group(1)   # self_attn or cross_attn
        proj = attn_m.group(2)        # q / k / v / o
        return f"blocks.{block_idx}.{attn_type}.{proj}"

    ffn_m = re.match(r'^ffn_(\d+)$', layer_part)
    if ffn_m:
        ffn_idx = ffn_m.group(1)
        return f"blocks.{block_idx}.ffn.{ffn_idx}"

    return None


def load_loras_to_pipeline(pipe, selected_loras):
    """
    Load selected LoRAs into the Wan 2.2 dual-transformer pipeline using PEFT native API.

    Wan 2.2 I2V has two transformer experts:
      pipe.transformer   — high-noise denoising expert (early steps)
      pipe.transformer_2 — low-noise denoising expert (late steps)

    Uses diffusers/PEFT's built-in load_lora_weights() which correctly handles
    all LoRA key formats automatically.
    """
    global _active_loras

    if pipe is None:
        return

    try:
        if hasattr(pipe, 'unfuse_lora'):
            pipe.unfuse_lora()
    except Exception:
        pass
    try:
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
    except Exception:
        pass

    _active_loras = {}

    if not selected_loras:
        print("[LoRA] No LoRAs selected — all unloaded.")
        return

    loaded_count = 0

    for base_name, lora_info in selected_loras.items():
        high_path   = lora_info.get("high")
        low_path    = lora_info.get("low")
        high_weight = float(lora_info.get("high_weight") or 1.0)
        low_weight  = float(lora_info.get("low_weight")  or 1.0)
        lora_ok     = False

        if high_path and os.path.exists(high_path):
            try:
                pipe.load_lora_weights(high_path, adapter_name=f"{base_name}_high")
                pipe.set_adapters([f"{base_name}_high"], adapter_weights=[high_weight])
                print(f"[LoRA] transformer: '{base_name}_high' loaded (weight={high_weight:.3f}) <- {os.path.basename(high_path)}")
                lora_ok = True
            except Exception as e:
                print(f"[LoRA] ERROR: failed to load high-noise LoRA for '{base_name}': {e}")

        has_t2 = hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None
        if low_path and os.path.exists(low_path):
            if has_t2:
                try:
                    pipe.load_lora_weights(low_path, adapter_name=f"{base_name}_low")
                    pipe.set_adapters([f"{base_name}_low"], adapter_weights=[low_weight])
                    print(f"[LoRA] transformer_2: '{base_name}_low' loaded (weight={low_weight:.3f}) <- {os.path.basename(low_path)}")
                    lora_ok = True
                except Exception as e:
                    print(f"[LoRA] ERROR: failed to load low-noise LoRA for '{base_name}': {e}")
            else:
                print(f"[LoRA] WARNING: '{base_name}' has low-noise file but pipeline has no transformer_2")

        if lora_ok:
            _active_loras[base_name] = lora_info
            loaded_count += 1
        else:
            print(f"[LoRA] WARNING: '{base_name}' — no files loaded (high={high_path}, low={low_path})")

    if loaded_count:
        print(f"[LoRA] {loaded_count} LoRA(s) loaded and activated successfully.")
    else:
        print("[LoRA] WARNING: no LoRAs were successfully loaded.")


def apply_lora_prompt_modifications(base_prompt, selected_loras_info):
    """
    Apply trigger prompts from active LoRAs to the user's base prompt.

    Alias matching: if any trigger_alias appears in the user's prompt (whole-word
    match), the trigger is considered already satisfied — the alias IS the trigger
    for prompt-routing purposes.  The trigger token itself is still prepended/appended
    so the LoRA keys fire, but we don't double-inject if the user already wrote it.

    Modes:
    - prepend: add trigger to the front (default for identity LoRAs — T5 gives
               front tokens more context weight, which is critical for face recall)
    - append:  add trigger to the end (for style/motion LoRAs where the trigger
               word is a technical suffix, not a subject description)
    - replace: the trigger IS the prompt — replaces the base prompt entirely.
               Used for LoRAs (like deepthroat) that require a very specific
               prompt structure.
    """
    modified_prompt = base_prompt.strip()

    for lora_info in selected_loras_info.values():
        trigger = (lora_info.get('trigger_prompt') or "").strip()
        if not trigger:
            continue

        mode = lora_info.get('prompt_mode', 'prepend')

        trigger_present = trigger.lower() in modified_prompt.lower()

        aliases = lora_info.get('trigger_aliases', [])
        alias_present = False
        if aliases and not trigger_present:
            prompt_lower = modified_prompt.lower()
            for alias in aliases:
                alias_lower = alias.strip().lower()
                if not alias_lower:
                    continue
                import re
                pattern = r'(?<![a-z0-9_])' + re.escape(alias_lower) + r'(?![a-z0-9_])'
                if re.search(pattern, prompt_lower):
                    alias_present = True
                    break

        already_anchored = trigger_present or alias_present

        if mode == 'replace':
            if not modified_prompt:
                modified_prompt = trigger
            elif not trigger_present:
                modified_prompt = f"{trigger}, {modified_prompt}"

        elif mode == 'prepend':
            if not trigger_present:
                modified_prompt = f"{trigger}, {modified_prompt}"

        else:  # append (default for style/motion LoRAs) — also handles 'natural'
            if not trigger_present:
                modified_prompt = f"{modified_prompt}, {trigger}"

    return modified_prompt.strip().strip(',').strip()

def check_lora_compatibility(selected_loras_info):
    """
    Check compatibility between selected LoRAs and return merged settings.
    Returns: (is_compatible, compatibility_message, merged_settings)
    """
    if not selected_loras_info:
        return True, "", {}
    
    lora_names = list(selected_loras_info.keys())
    
    if len(lora_names) == 1:
        lora_info = list(selected_loras_info.values())[0]
        return True, f" Using {lora_info['display_name']}", {
            'recommended_steps': lora_info.get('recommended_steps'),
            'recommended_flow_shift': lora_info.get('recommended_flow_shift'),
            'high_weight': lora_info.get('high_weight', 1.0),
            'low_weight': lora_info.get('low_weight', 1.0),
        }
    
    recommended_steps = []
    recommended_flow_shifts = []
    high_weights = []
    low_weights = []
    warnings = []
    
    for lora_id, lora_info in selected_loras_info.items():
        if lora_info.get('recommended_steps'):
            recommended_steps.append(lora_info['recommended_steps'])
        if lora_info.get('recommended_flow_shift'):
            recommended_flow_shifts.append(lora_info['recommended_flow_shift'])
        high_weights.append(lora_info.get('high_weight', 1.0))
        low_weights.append(lora_info.get('low_weight', 1.0))
    
    steps_compatible = len(set(recommended_steps)) <= 1 if recommended_steps else True
    flow_compatible = len(set(recommended_flow_shifts)) <= 1 if recommended_flow_shifts else True
    
    display_names = [info['display_name'] for info in selected_loras_info.values()]
    
    if not steps_compatible:
        warnings.append(f" Step conflict: {', '.join(map(str, set(recommended_steps)))} steps recommended")
    
    if not flow_compatible:
        warnings.append(f" Flow shift conflict: {', '.join(map(str, set(recommended_flow_shifts)))}")
    
    merged_steps = recommended_steps[0] if steps_compatible and recommended_steps else None
    merged_flow = recommended_flow_shifts[0] if flow_compatible and recommended_flow_shifts else None
    
    merged_high_weight = sum(high_weights) / len(high_weights)
    merged_low_weight = sum(low_weights) / len(low_weights)
    
    is_compatible = steps_compatible and flow_compatible
    
    if is_compatible:
        message = f" Compatible: {', '.join(display_names)}"
    else:
        message = f" Combining: {', '.join(display_names)}\n" + '\n'.join(warnings)
    
    merged_settings = {
        'recommended_steps': merged_steps,
        'recommended_flow_shift': merged_flow,
        'high_weight': merged_high_weight,
        'low_weight': merged_low_weight,
    }
    
    return is_compatible, message, merged_settings

def apply_lora_settings(selected_loras_info, user_steps, user_flow_shift, flow_shift_auto):
    """
    Apply LoRA-recommended settings when a LoRA is active.

    Rules:
    - recommended_steps: ALWAYS applied when a LoRA is active — overrides whatever
      the slider currently shows. The UI already sets the slider on checkbox change
      via update_lora_compatibility_and_steps, so this ensures the actual generation
      call uses the right value even if the slider drifted.
    - recommended_flow_shift: applied unless the user explicitly put the slider into
      manual mode (flow_shift_auto=False AND they moved it off the LoRA recommendation).
      When in auto mode the LoRA recommendation always wins.
    - high_weight / low_weight: used by load_loras_to_pipeline which reads them from
      the lora_info dict — no extra action needed here.
    """
    if not selected_loras_info:
        return user_steps, user_flow_shift, ""

    is_compatible, compat_msg, merged_settings = check_lora_compatibility(selected_loras_info)
    messages = [compat_msg] if compat_msg else []

    final_steps = user_steps
    recommended_steps = merged_settings.get('recommended_steps')
    if recommended_steps is not None and user_steps == 4:
        # Only apply the LoRA recommendation when the slider is still at the
        # default value of 4 (meaning the user hasn't deliberately changed it).
        # If the user moved the slider to any other value, respect that choice.
        final_steps = recommended_steps
        messages.append(f"Steps set to {recommended_steps} (LoRA recommendation)")

    final_flow_shift = user_flow_shift
    recommended_flow = merged_settings.get('recommended_flow_shift')
    if recommended_flow is not None:
        if flow_shift_auto:
            final_flow_shift = recommended_flow
            messages.append(f"Flow shift set to {recommended_flow} (LoRA recommendation, auto mode)")
        else:
            final_flow_shift = recommended_flow
            messages.append(f"Flow shift set to {recommended_flow} (LoRA recommendation)")

    if not is_compatible and len(selected_loras_info) > 1:
        messages.append("Note: multiple LoRAs with conflicting settings — using first recommendation.")

    return final_steps, final_flow_shift, "\n".join(messages)

LORA_CONFIG = load_lora_config()
AVAILABLE_LORAS = discover_loras()
LORA_STATUS = check_lora_status(LORA_CONFIG)

for lora_id, info in AVAILABLE_LORAS.items():
    status = LORA_STATUS.get(lora_id, {})
    high_status = "OK" if info['high'] else ("DL" if status.get('high_downloadable') else "X")
    low_status = "OK" if info['low'] else ("DL" if status.get('low_downloadable') else "X")


FIXED_FPS = 16

WAN_STEPS = 3  # Default fallback, actual steps come from UI slider
WAN_FLOW_SHIFT = 6.9
WAN_GUIDANCE = 1.0

MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 97       # 6s per segment (97 frames 16 fps = 6.06s)  keeps quality high
SEGMENT_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)   # ~6.1s per segment
MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = 600.0        # 10 minutes max via chaining

AREA_1080P = 1920 * 1080
AREA_720P  = 1280 * 720
AREA_600P  = 1024 * 576   # in-between: ~16:9 at ~600p
AREA_480P  = 832  * 480
AREA_240P  = 416  * 240   # half of 480p, ultra-fast/small
MULTIPLE_OF = 16

wan_pipe = None
_wan_loaded = False
_wan_scheduler_config = None

_protected_image_paths = set()
_protected_paths_lock = threading.Lock()

_generation_active_paths = set()
_generation_active_lock = threading.Lock()


def _generation_protect(path):
    """Pin a path as in-use by a running generation."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.add(p)
        _protect_path(p)          # also in the general set
    except Exception:
        pass


def _generation_release(path):
    """Release a path from the generation-active set (generation done/error)."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.discard(p)
    except Exception:
        pass


def _is_generation_active(path) -> bool:
    """True if path is currently held by a running generation."""
    if not path:
        return False
    try:
        p = str(path)
        with _generation_active_lock:
            return p in _generation_active_paths
    except Exception:
        return False


_protected_image_filenames = set()
_protected_filenames_lock = threading.Lock()


def _protect_filename(path):
    """Register the basename of path as protected from clear_storage()."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.add(name)
    except Exception:
        pass


def _unprotect_filename(path):
    """Remove the basename of path from the protected-filenames set."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.discard(name)
    except Exception:
        pass


def _is_filename_protected(item_path) -> bool:
    """True if the item's basename is in the protected-filenames set."""
    try:
        name = os.path.basename(str(item_path))
        with _protected_filenames_lock:
            return name in _protected_image_filenames
    except Exception:
        return False


def _protect_path(path):
    """Mark a filesystem path as protected from clear_storage()."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.add(p)


def _unprotect_path(path):
    """Remove a path from the protected set (safe no-op if absent)."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.discard(p)


def _is_protected(item_path) -> bool:
    """True if item_path is, or contains, a currently-protected input image.

    Two layers of protection:
    1. Full-path set: _protected_image_paths (and legacy _current_input_image_path).
    2. Filename set: _protected_image_filenames — any item whose *basename*
       matches a currently-live first-frame or last-frame filename is skipped,
       even if the full path lookup misses (e.g. Gradio moved/renamed the file).
    """
    if _is_filename_protected(item_path):
        return True

    item_path = Path(item_path)
    with _protected_paths_lock:
        protected_now = set(_protected_image_paths)
    if _current_input_image_path:
        protected_now.add(_current_input_image_path)
    with _merge_output_lock:
        _mop = _current_merge_output_path
    for p in protected_now:
        if not p:
            continue
        if p.startswith("/media/"):
            continue
        protected_path = Path(p)
        if item_path == protected_path:
            return True
        if item_path.is_dir():
            try:
                protected_path.relative_to(item_path)
                return True
            except ValueError:
                pass
    return False



_current_input_image_path = None

_current_merge_output_path = None
_merge_output_lock = threading.Lock()


def _set_flow_shift(pipe, flow_shift):
    """
    Rebuild the scheduler with a different flow shift.

    Flow shift controls how the noise schedule is warped, which materially
    affects motion amount on Wan. Always derived from the pristine config so
    repeated changes cannot compound. Safe to mutate because video jobs are
    serialised by the queue.
    """
    if flow_shift is None or _wan_scheduler_config is None:
        return

    cfg = dict(_wan_scheduler_config)
    key = "flow_shift" if "flow_shift" in cfg else "shift" if "shift" in cfg else None
    if key is None:
        return

    try:
        if abs(float(cfg.get(key) or 0.0) - float(flow_shift)) < 1e-6:
            current = getattr(pipe.scheduler.config, key, None)
            if current is not None and abs(float(current) - float(flow_shift)) < 1e-6:
                return
        cfg[key] = float(flow_shift)
        pipe.scheduler = type(pipe.scheduler).from_config(cfg)
    except Exception as e:
        print(f"Could not apply flow_shift={flow_shift} ({e})  keeping default.")


_wan_load_lock = threading.Lock()


def _load_wan(target_device="cpu"):
    """
    Build the Wan 2.2 I2V pipeline with both merged 4-step experts.

    Guarded by its own lock, separate from the VRAM swap lock, so the one-time
    ~57 GB download and model build cannot block the Photo Editor tab.
    """
    global wan_pipe, _wan_loaded
    if _wan_loaded and wan_pipe is not None:
        return wan_pipe

    with _wan_load_lock:
        if _wan_loaded and wan_pipe is not None:
            return wan_pipe
        return _build_wan_pipeline(target_device)


def _from_pretrained_cached(repo: str, **kwargs):
    """Try the local cache first, then fall back to downloading."""
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(
            repo, local_files_only=True, **kwargs
        )
        return pipe
    except Exception:
        return WanImageToVideoPipeline.from_pretrained(repo, **kwargs)


def _build_wan_pipeline(target_device="cpu"):
    global wan_pipe, _wan_loaded, _wan_scheduler_config

    if target_device == "cpu":
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        print(f" WAMU v2 loaded to CPU (ready for fast swapping)")
    else:
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        pipeline = pipeline.to(target_device)
        torch.cuda.synchronize(target_device)
        print(f" WAMU v2 loaded directly to {target_device} - Ready for video generation!")

    _wan_scheduler_config = dict(pipeline.scheduler.config)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    wan_pipe = pipeline
    _wan_loaded = True
    return wan_pipe




default_video_prompt = (
    "Cinematic, slow sweeping camera movement, natural body motion, "
    "detailed skin texture, dramatic ambient lighting"
)
default_negative_prompt = (
    "static, frozen, blurry, low quality, distorted, deformed, "
    "extra limbs, watermark, text, logo"
)


def model_title():
    gpu_note = (
        f"Dual GPU: Qwen on {PIC_DEVICE}, Wan on {WAN_DEVICE} (no swapping)."
        if DUAL_GPU else
        "Single GPU: models swap on demand."
    )
    return (
        "<div style='text-align:center'>"
        "<h2 style='margin-bottom:4px'>&#xFEFF; WAMU v2 &#x2014; Wan 2.2 I2V Lightning (NSFW)</h2>"
        f"<p style='margin:0'>4-step distilled merge. No LoRAs. {gpu_note}</p>"
        "</div>"
    )


def _ensure_pil(image):
    """
    Normalize a Gradio image value to a PIL Image (or None).

    Handles:
      - None / "" / falsy  → None
      - /media/<key>/...  → resolve from _media_store, open as PIL
      - file path string   → Image.open()
      - PIL Image          → returned as-is
    """
    if not image:
        return None
    if isinstance(image, str):
        if image.startswith("/media/"):
            key = image.split("/")[2]
            entry = _media_store_get(key)
            if entry is None:
                return None
            return Image.open(BytesIO(entry[0])).convert("RGB")
        return Image.open(image).convert("RGB")
    return image


def resize_image_for_wan(image: Image.Image, resolution: str = "720p") -> Image.Image:
    """
    Fit an image to a target pixel area while preserving aspect ratio.

    This mirrors Wan-AI's own sizing recipe: pick an area budget, derive width
    and height from the image's aspect ratio, then round both to a multiple of
    16. For I2V the area matters, not fixed dimensions.
    """
    image = _ensure_pil(image)

    cached = _get_cached_resized(image, resolution)
    if cached is not None:
        return cached
    
    if resolution == "240p":
        max_area = AREA_240P
    elif resolution == "480p":
        max_area = AREA_480P
    elif resolution == "600p":
        max_area = AREA_600P
    elif resolution == "1080p":
        max_area = AREA_1080P
    else:
        max_area = AREA_720P

    aspect = image.height / image.width
    height = int(round(np.sqrt(max_area * aspect))) // MULTIPLE_OF * MULTIPLE_OF
    width = int(round(np.sqrt(max_area / aspect))) // MULTIPLE_OF * MULTIPLE_OF

    height = max(MULTIPLE_OF * 8, height)  # Increased minimum for VAE compatibility
    width = max(MULTIPLE_OF * 8, width)   # Increased minimum for VAE compatibility
    
    if height % 32 != 0:
        height = (height // 32 + 1) * 32
    if width % 32 != 0:
        width = (width // 32 + 1) * 32
    
    resized = image.resize((width, height), Image.LANCZOS)
    
    _cache_resized(image, resolution, resized)
    return resized


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def _remove_background_rembg(pil_image: Image.Image) -> Image.Image:
    """
    Remove background from a PIL image using rembg with birefnet-general model.
    Returns RGBA image. Falls back to original image converted to RGBA if rembg unavailable.
    """
    try:
        # BUG 3 FIX: Force CPU-only execution by passing providers explicitly.
        # Setting ORT_FORCE_CPU=1 in os.environ is ignored by onnxruntime after
        # the first session is created — the CUDA provider is already registered
        # in the process and the env-var check only runs at import time.
        # Passing providers=['CPUExecutionProvider'] directly to new_session()
        # bypasses the CUDA provider entirely, regardless of process-level state.
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        from rembg import new_session, remove
        session = new_session("birefnet-general", providers=["CPUExecutionProvider"])
        rgba = remove(pil_image.convert("RGBA"), session=session)
        return rgba
    except ImportError:
        print("rembg not installed — background removal skipped. Install with: pip install rembg[gpu]")
        return pil_image.convert("RGBA")
    except Exception as e:
        print(f"Background removal failed: {e} — using original image")
        return pil_image.convert("RGBA")


MERGE_BG_LABELS = [
    "Person A",
    "Person B",
    "Hotel Room",
    "Hotel Shower",
    "Hotel On Bed",
    "Her Room On Bed",
    "Outside Wilderness",
    "Back Seat Of Vehicle",
    "Custom",
]

MERGE_BG_PROMPTS = {
    "Person A": None,
    "Person B": None,
    "Hotel Room": (
        "a luxurious fancy hotel room with elegant decor, large bed with white linens, "
        "warm soft lighting, polished wooden floors, city view window"
    ),
    "Hotel Shower": (
        "a sleek luxury hotel bathroom shower, glass-walled shower with marble tiles, "
        "rainfall showerhead, soft ambient lighting, high-end toiletries"
    ),
    "Hotel On Bed": (
        "on a large king-sized hotel bed with crisp white linens and fluffy pillows, "
        "elegant nightstands with lamps, warm hotel room lighting"
    ),
    "Her Room On Bed": (
        "on a cozy feminine bedroom bed with soft pink and white bedding, "
        "fairy lights, plush pillows, warm intimate lighting, personal decor around the room"
    ),
    "Outside Wilderness": (
        "outdoors in a lush natural wilderness setting, tall trees, golden hour sunlight "
        "filtering through leaves, green grass, natural earthy environment"
    ),
    "Back Seat Of Vehicle": (
        "sitting side by side together in the back seat of a large luxury SUV or "
        "limousine, leather interior, tinted windows with city lights outside, "
        "soft cabin lighting"
    ),
}

# Options that must stay as close to the original photos as possible: only the
# background/setting changes, and the people are preserved with maximum
# fidelity (faces fully identical and visible, all visible body parts exactly
# unchanged). These use MERGE_BG_INSTRUCTION_PRESERVE instead of the standard
# instruction, and skip the heavier "reposition into the scene" fit wording.
MERGE_BG_MINIMAL_CHANGE = {"Hotel On Bed", "Back Seat Of Vehicle"}

MERGE_BG_INSTRUCTION = (
    "Keep both people exactly as they are — identical faces, facial features, hair, "
    "body shape, proportions and skin tone. Do not alter their identities or bodies, "
    "and do not add, remove or duplicate any person. "
    "Replace the entire background and environment with: {prompt}. "
    "Relight both subjects naturally to match the new environment."
)

# Auto-appended to EVERY scene generation (presets and custom) so the caller
# never has to describe framing/fit/placement. It guarantees both full bodies
# are visible and correctly sized, and that they are posed naturally for the
# scene (e.g. actually sitting ON a bed rather than floating in front of it).
MERGE_FIT_INSTRUCTION = (
    " Position both people naturally within the scene, correctly sized relative "
    "to each other and to the environment, and interacting with it realistically "
    "(for example, actually sitting or lying on the bed/furniture if the scene "
    "has one, with proper contact and perspective). Show BOTH people's FULL "
    "bodies head-to-toe, entirely inside the frame, with nothing cut off at any "
    "edge. Keep both faces and bodies unchanged."
)

# Maximum-preservation instruction for options that should barely change the
# people (Hotel On Bed, Back Seat Of Vehicle). Only the background/setting is
# swapped; the people themselves must stay as close to their original photos as
# possible — faces fully identical and clearly visible, every visible body part
# exactly unchanged. Kept short and firm so the model does the minimum edit.
MERGE_BG_INSTRUCTION_PRESERVE = (
    "Change ONLY the background and setting to: {prompt}. "
    "Keep both people exactly as in the original photos — do not alter, restyle, "
    "or move them. Their FACES must remain fully identical and clearly visible, "
    "and every visible body part (skin, hair, hands, limbs, torso) must stay "
    "exactly the same, unchanged. Make the smallest possible change: only "
    "replace the surroundings and relight subtly to match. Do not add, remove or "
    "duplicate any person, and keep both people's full bodies visible in frame."
)


def _complete_body_safe(rgba: Image.Image) -> Image.Image:
    """
    Generatively fill ONLY the missing parts of a person whose body is clipped
    at the top (head) or bottom (feet) of their photo, WITHOUT altering a
    single original pixel.

    How the original is protected:
    - We pad transparent space only on the clipped side(s).
    - Qwen fills the whole padded canvas, but we then hard-paste the EXACT
      original RGBA back on top at full opacity, so every original pixel is
      byte-for-byte unchanged. Qwen output is used ONLY inside the padded
      (previously-empty) region.
    - The alpha for the newly generated region is derived by re-running rembg
      on the Qwen output and masking it to the padded region only; the
      original alpha is preserved verbatim everywhere it existed. This avoids
      the earlier bug where re-running rembg over the whole composite let the
      generated alpha bleed into and mangle the visible body.

    Returns an RGBA image (trimmed to its bbox). If nothing is clipped, the
    input is returned untouched (no diffusion call → fast).
    """
    if rgba is None:
        return rgba

    W, H = rgba.size
    alpha_arr = np.array(rgba.split()[3], dtype=np.uint8)
    if alpha_arr.size == 0:
        return rgba

    # A body is "clipped" on a side if opaque subject pixels touch that edge.
    EDGE = 6
    bottom_clipped = bool(np.any(alpha_arr[-EDGE:, :] > 128))
    top_clipped = bool(np.any(alpha_arr[:EDGE, :] > 128))

    if not bottom_clipped and not top_clipped:
        return rgba  # full body already visible — nothing to fill

    pad_bottom = int(H * 0.40) if bottom_clipped else 0
    pad_top = int(H * 0.40) if top_clipped else 0
    new_w = W
    new_h = H + pad_top + pad_bottom

    # Compose the subject onto a white canvas with transparent padding regions.
    padded_rgba = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 0))
    padded_rgba.paste(rgba, (0, pad_top), rgba)
    rgb_canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    rgb_canvas.paste(padded_rgba.convert("RGB"), mask=padded_rgba.split()[3])

    parts = []
    if bottom_clipped:
        parts.append("legs and feet at the bottom")
    if top_clipped:
        parts.append("head and hair at the top")
    what = " and ".join(parts)

    instruction = (
        f"The person's {what} is cropped off. Extend and complete ONLY the "
        "missing body parts into the empty area, matching their exact skin "
        "tone, body shape, and proportions. Do NOT change, redraw, or move any "
        "part of the body that is already visible — keep it pixel-for-pixel "
        "identical. Only paint into the empty space."
    )

    try:
        activate_pic()
        torch.cuda.set_device(PIC_DEVICE)
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = pic_pipe(
                    image=[rgb_canvas],
                    prompt=instruction,
                    negative_prompt="distorted, deformed, extra limbs, mutated, blurry",
                    num_inference_steps=4,
                    true_cfg_scale=1.0,
                    generator=torch.Generator(device=PIC_DEVICE).manual_seed(
                        random.randint(0, np.iinfo(np.int32).max)
                    ),
                )
        gen_rgb = result.images[0]
        if gen_rgb.size != (new_w, new_h):
            gen_rgb = gen_rgb.resize((new_w, new_h), Image.LANCZOS)

        # Alpha for the generated result (whole canvas), from rembg.
        # This must be computed here so gen_alpha is defined before use below.
        gen_alpha = np.array(
            _remove_background_rembg(gen_rgb).split()[3], dtype=np.uint8
        )

        # Build the output: start from the generated RGB, and give it an alpha
        # that is (a) the ORIGINAL alpha wherever the original subject existed,
        # and (b) the generated alpha ONLY in the padded regions.
        out_rgb = np.array(gen_rgb.convert("RGB"), dtype=np.uint8)
        out_alpha = np.zeros((new_h, new_w), dtype=np.uint8)

        # Restrict generated alpha to the padded rows only.
        if pad_top:
            out_alpha[:pad_top, :] = gen_alpha[:pad_top, :]
        if pad_bottom:
            out_alpha[new_h - pad_bottom:, :] = gen_alpha[new_h - pad_bottom:, :]

        # Original region alpha preserved verbatim.
        out_alpha[pad_top:pad_top + H, :] = alpha_arr

        out_arr = np.dstack([out_rgb, out_alpha]).astype(np.uint8)

        # Hard-paste the EXACT original RGB back over its region so not one
        # visible pixel is altered by the diffusion pass.
        orig_rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
        region = out_arr[pad_top:pad_top + H, :, :3]
        orig_mask = (alpha_arr > 0)
        region[orig_mask] = orig_rgb[orig_mask]
        out_arr[pad_top:pad_top + H, :, :3] = region

        result_rgba = Image.fromarray(out_arr, mode="RGBA")
        bbox = result_rgba.getbbox()
        if bbox:
            result_rgba = result_rgba.crop(bbox)
        return result_rgba

    except Exception as e:
        print(f"[merge] safe body completion failed ({e}); using original subject")
        return rgba


def _scale_subject_to_fit(rgba: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """
    Scale an RGBA subject image so it fits within (max_w, max_h) while
    preserving aspect ratio. Never upscales beyond natural size.
    """
    sw, sh = rgba.size
    scale = min(max_w / sw, max_h / sh, 1.0)
    if scale >= 0.999:
        return rgba
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b)).resize((new_w, new_h), Image.LANCZOS)
    alpha_s = a.resize((new_w, new_h), Image.LANCZOS)
    alpha_arr = np.where(np.array(alpha_s, dtype=np.uint8) > 128, 255, 0).astype(np.uint8)
    out = rgb.convert("RGBA")
    out.putalpha(Image.fromarray(alpha_arr, mode="L"))
    return out


def _extract_background_rgb(pil_img: Image.Image) -> Image.Image:
    """Return a 1280x720 cover-crop of the original image for use as background."""
    OUT_W, OUT_H = 1280, 720
    scale = max(OUT_W / pil_img.width, OUT_H / pil_img.height)
    new_w = int(pil_img.width * scale)
    new_h = int(pil_img.height * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - OUT_W) // 2
    top = (new_h - OUT_H) // 2
    return resized.crop((left, top, left + OUT_W, top + OUT_H)).convert("RGB")


def _apply_merge_background(
    merged_rgba: Image.Image,
    bg_choice: str | None,
    pil_a: Image.Image,
    pil_b: Image.Image,
    extra_prompt: str | None = None,
) -> Image.Image:
    """Composite the merged subjects onto the chosen background.

    `extra_prompt` is optional free text from the always-visible custom box;
    when present it is combined with the chosen option's scene prompt.
    """
    OUT_W, OUT_H = 1280, 720
    extra = (extra_prompt or "").strip()

    if not bg_choice:
        bg = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
        final = bg.copy()
        final.paste(merged_rgba.convert("RGB"), mask=merged_rgba.split()[3])
        return final

    # Note: "Person A" / "Person B" are handled entirely inside merge_photos_fn
    # (via _merge_into_person_photo) and never reach this function, so they are
    # not handled here.
    if True:
        prompt_text = MERGE_BG_PROMPTS.get(bg_choice)
        # Combine the option's scene prompt with any extra custom text.
        if prompt_text and extra:
            prompt_text = f"{prompt_text}. {extra}"
        elif extra and not prompt_text:
            prompt_text = extra
        if not prompt_text:
            bg = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
        else:
            try:
                # Minimal-change options (Hotel On Bed, Back Seat Of Vehicle)
                # use the maximum-preservation instruction so faces/bodies stay
                # as in the originals; others use the standard swap + fit wording.
                if bg_choice in MERGE_BG_MINIMAL_CHANGE:
                    instruction = MERGE_BG_INSTRUCTION_PRESERVE.format(prompt=prompt_text)
                else:
                    instruction = MERGE_BG_INSTRUCTION.format(prompt=prompt_text) + MERGE_FIT_INSTRUCTION
                white_canvas = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
                base = white_canvas.copy()
                _m = merged_rgba if merged_rgba.mode == "RGBA" else merged_rgba.convert("RGBA")
                base.paste(_m.convert("RGB"), mask=_m.split()[3])

                activate_pic()
                torch.cuda.set_device(PIC_DEVICE)
                with torch.cuda.device(PIC_DEVICE):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        result = pic_pipe(
                            image=[base],
                            prompt=instruction,
                            negative_prompt=" ",
                            num_inference_steps=4,
                            true_cfg_scale=1.0,
                            generator=torch.Generator(device=PIC_DEVICE).manual_seed(
                                random.randint(0, np.iinfo(np.int32).max)
                            ),
                        )
                edited = result.images[0]
                if edited.size != (OUT_W, OUT_H):
                    edited = edited.resize((OUT_W, OUT_H), Image.LANCZOS)
                return edited
            except Exception as e:
                print(f"[merge bg] Qwen edit failed ({e}), falling back to white background")
                bg = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))

    final = bg.copy()
    final.paste(merged_rgba.convert("RGB"), mask=merged_rgba.split()[3])
    return final


def _complete_body_on_photo(photo: Image.Image) -> Image.Image:
    """
    Generatively fill ONLY the missing body parts of the person in `photo`,
    directly on their own full photo (background kept), WITHOUT altering any
    existing pixel. Returns an RGB photo, possibly taller than the input if
    body parts were added at the top/bottom.

    Protection of the original: Qwen fills a padded canvas, but afterwards we
    hard-paste the EXACT original photo back over its region, so every original
    pixel (person AND their background) is byte-for-byte unchanged. Qwen output
    is used ONLY in the added padding rows.

    If the person is not clipped at top/bottom, the photo is returned untouched
    (no diffusion call → fast).
    """
    if photo is None:
        return photo
    rgb = photo.convert("RGB")

    # Detect whether the SUBJECT touches the top/bottom edge (i.e. is clipped).
    # Use rembg alpha so we test the person, not the background.
    try:
        subj_alpha = np.array(_remove_background_rembg(rgb).split()[3], dtype=np.uint8)
    except Exception:
        return rgb
    if subj_alpha.size == 0:
        return rgb

    W, H = rgb.size
    EDGE = 6
    bottom_clipped = bool(np.any(subj_alpha[-EDGE:, :] > 128))
    top_clipped = bool(np.any(subj_alpha[:EDGE, :] > 128))
    if not bottom_clipped and not top_clipped:
        return rgb  # full body already in frame — nothing to add

    pad_top = int(H * 0.40) if top_clipped else 0
    pad_bottom = int(H * 0.40) if bottom_clipped else 0
    new_w = W
    new_h = H + pad_top + pad_bottom

    # Pad with a neutral fill; place the original photo in the middle band.
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(rgb, (0, pad_top))

    parts = []
    if bottom_clipped:
        parts.append("legs and feet at the bottom")
    if top_clipped:
        parts.append("head and hair at the top")
    what = " and ".join(parts)
    instruction = (
        f"The person's {what} is cropped off. Extend and complete ONLY the "
        "missing body parts and the surrounding background into the empty area, "
        "matching their exact skin tone, body shape and the existing scene. Do "
        "NOT change, move or redraw anything already visible — keep it identical. "
        "Only paint into the empty space."
    )

    try:
        activate_pic()
        torch.cuda.set_device(PIC_DEVICE)
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = pic_pipe(
                    image=[canvas],
                    prompt=instruction,
                    negative_prompt="distorted, deformed, extra limbs, mutated, duplicate person",
                    num_inference_steps=4,
                    true_cfg_scale=1.0,
                    generator=torch.Generator(device=PIC_DEVICE).manual_seed(
                        random.randint(0, np.iinfo(np.int32).max)
                    ),
                )
        gen = result.images[0]
        if gen.size != (new_w, new_h):
            gen = gen.resize((new_w, new_h), Image.LANCZOS)
        # Hard-restore the exact original photo band; keep Qwen only in padding.
        out = gen.convert("RGB")
        out.paste(rgb, (0, pad_top))
        return out
    except Exception as e:
        print(f"[merge] on-photo body completion failed ({e}); using original photo")
        return rgb


def _merge_into_person_photo(
    base_photo: Image.Image,
    other_rgba: Image.Image,
    add_side: str,
    gender_base: str | None = None,
    gender_other: str | None = None,
) -> Image.Image:
    """
    Pixel-accurate Person A/B merge:

    1. Scale the other person's crop to match the base person's height (so they
       look the same size standing next to each other).
    2. Measure the scaled crop's pixel width and add a comfortable padding margin
       (20 % extra) — this is exactly how many pixels to add to the base photo.
    3. Scale the base photo to fit the output height, place it on the chosen side
       of an extended canvas whose extra strip is exactly (scaled-other-width +
       margin) pixels wide.
    4. Run ONE Qwen outpaint pass to fill the empty strip so the background
       continues seamlessly — Qwen only ever sees white in the empty strip, so
       it cannot bleed the other person's content in.
    5. Hard-paste the exact original base photo back over its region (the outpaint
       pass must NOT change anything already visible).
    6. Composite the other person's crop into the filled strip at the correct
       position and height — bottom-aligned to the base person's feet.

    add_side = "right" → base photo on the LEFT,  other person on the RIGHT
    add_side = "left"  → base photo on the RIGHT, other person on the LEFT
    """
    # ── Constants ──────────────────────────────────────────────────────────────
    OUT_H = 720          # fixed output height
    PADDING_FACTOR = 1.25  # 25 % extra space so the other person isn't crammed
    MARGIN_PX = 24       # minimum extra pixels on each side of the pasted person

    # ── Step 1: scale base photo to output height ──────────────────────────────
    base_rgb = base_photo.convert("RGB")
    base_scale = OUT_H / base_rgb.height
    base_w = max(1, int(base_rgb.width  * base_scale))
    base_h = OUT_H
    base_fit = base_rgb.resize((base_w, base_h), Image.LANCZOS)
    print(f"[merge] base photo scaled to {base_w}×{base_h}")

    # ── Step 2: body-complete + scale other person to match base height ────────
    # Target: the other person should appear roughly the same height as the base
    # person (who fills the full OUT_H).  We scale the other person so their
    # body height equals OUT_H * 0.92 (slightly shorter so the composition feels
    # natural — two people standing together are rarely exactly the same height
    # in frame).
    other_target_h = int(OUT_H * 0.92)

    # Scale other person proportionally to target height; never upscale beyond
    # their natural resolution.
    ow, oh = other_rgba.size
    o_scale = min(other_target_h / oh, 1.0)   # never upscale
    other_w = max(1, int(ow * o_scale))
    other_h = max(1, int(oh * o_scale))
    # Resize RGBA (split+merge to preserve alpha)
    _or, _og, _ob, _oa = other_rgba.split()
    other_rgb_r = Image.merge("RGB", (_or, _og, _ob)).resize((other_w, other_h), Image.LANCZOS)
    other_alpha_r = _oa.resize((other_w, other_h), Image.LANCZOS)
    other_alpha_arr = np.where(np.array(other_alpha_r, dtype=np.uint8) > 128, 255, 0).astype(np.uint8)
    other_scaled = other_rgb_r.convert("RGBA")
    other_scaled.putalpha(Image.fromarray(other_alpha_arr, "L"))
    print(f"[merge] other person scaled to {other_w}×{other_h} (target h={other_target_h})")

    # ── Step 3: compute strip width and total canvas width ────────────────────
    strip_w = max(MARGIN_PX * 2 + other_w, int(other_w * PADDING_FACTOR))
    OUT_W   = base_w + strip_w
    print(f"[merge] strip_w={strip_w}, total canvas={OUT_W}×{OUT_H}")

    # ── Step 4: build the canvas — base photo on the anchored side, white strip ─
    canvas = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
    if add_side == "right":
        base_x = 0
    else:
        base_x = strip_w
    canvas.paste(base_fit, (base_x, 0))

    # ── Step 5: gender preservation clause ────────────────────────────────────
    _gender_notes = []
    if gender_base and gender_base.lower() == "man":
        _gender_notes.append("the man's penis remains the same in every way, completely unchanged and still fully visible")
    if gender_other and gender_other.lower() == "man":
        _gender_notes.append("the man's penis remains the same in every way, completely unchanged and still fully visible")
    _gender_clause = (". " + "; ".join(_gender_notes)) if _gender_notes else ""

    # ── Step 6: Qwen outpaint — fill the white strip to match the background ──
    extend_dir = "to the right" if add_side == "right" else "to the left"
    outpaint_instruction = (
        f"Extend and continue this photo's background and environment {extend_dir} "
        "into the empty white area on that side, seamlessly matching the existing "
        "scene, lighting, colors, floor, walls and perspective. Keep everything "
        "already visible in the photo completely unchanged — do not alter the "
        f"person or anything else already visible. Only paint into the empty white area{_gender_clause}."
    )

    extended_bg = canvas.copy()
    try:
        activate_pic()
        torch.cuda.set_device(PIC_DEVICE)
        torch.cuda.synchronize(PIC_DEVICE)
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = pic_pipe(
                    image=[canvas],
                    prompt=outpaint_instruction,
                    negative_prompt="people, person, extra person, duplicate, seam, dividing line, border, frame, watermark",
                    num_inference_steps=4,
                    true_cfg_scale=1.0,
                    generator=torch.Generator(device=PIC_DEVICE).manual_seed(
                        random.randint(0, np.iinfo(np.int32).max)
                    ),
                )
        gen = result.images[0]
        if gen.size != (OUT_W, OUT_H):
            gen = gen.resize((OUT_W, OUT_H), Image.LANCZOS)
        extended_bg = gen
        print("[merge] outpaint succeeded")
    except Exception as e:
        print(f"[merge] outpaint failed ({e}); using plain canvas")
        extended_bg = canvas.copy()

    # ── Step 7: hard-restore the EXACT original base photo pixels ─────────────
    # The outpaint must not alter anything already in the photo.  Re-paste the
    # exact scaled base photo over its region so nothing changes there.
    final = extended_bg.copy()
    final.paste(base_fit, (base_x, 0))

    # ── Step 8: composite the other person into the strip ─────────────────────
    # Position: horizontally centred in the strip, vertically bottom-aligned
    # to the base person's feet (both people share the same floor line).
    base_baseline = OUT_H   # base person fills full height → feet at bottom

    if add_side == "right":
        strip_left = base_w
    else:
        strip_left = 0

    other_x = strip_left + max(0, (strip_w - other_w) // 2)
    other_x = max(0, min(other_x, OUT_W - other_w))
    other_y = max(0, base_baseline - other_h)

    final_rgba = final.convert("RGBA")
    final_rgba.paste(other_scaled, (other_x, other_y), other_scaled)
    result_rgb = final_rgba.convert("RGB")
    print(f"[merge] other person pasted at ({other_x}, {other_y}), final canvas {OUT_W}×{OUT_H}")
    return result_rgb


def merge_photos_fn(img_a, img_b, bg_choice: str | None = None,
                    extra_prompt: str | None = None,
                    gender_a: str | None = None,
                    gender_b: str | None = None,
                    add_side: str = "right") -> Image.Image | None:
    """
    Merge two photos side by side on a chosen background.

    Pipeline:
    1. Remove backgrounds with BiRefNet.
    2. Trim each subject to their alpha bounding box.
    3. Scale both subjects so they fit within their respective half-canvas slot
       at up to 95% canvas height, never upscaling.
    4. Bottom-align both subjects so feet sit at the same baseline.
    5. Composite onto the chosen background (white, Person A/B photo, or Qwen scene).

    Subjects are never regenerated — each person is composited using the exact
    pixels rembg extracts from their original photo, so their bodies are kept
    at 100% original quality with nothing altered or hallucinated.
    """
    if img_a is None or img_b is None:
        return None

    pil_a = _ensure_pil(img_a)
    pil_b = _ensure_pil(img_b)

    rgba_a = _remove_background_rembg(pil_a)
    rgba_b = _remove_background_rembg(pil_b)

    def trim_to_subject(rgba: Image.Image) -> Image.Image:
        bbox = rgba.getbbox()
        return rgba.crop(bbox) if bbox else rgba

    rgba_a = trim_to_subject(rgba_a)
    rgba_b = trim_to_subject(rgba_b)

    # Person A / Person B backgrounds follow this exact order:
    #   1. body-complete the CHOSEN person on their own photo (missing parts
    #      only, nothing else about them or their background altered),
    #   2. outpaint to extend their environment to one side,
    #   3. body-complete the OTHER person, crop them, and place them into the
    #      added space without altering them further.
    # The chosen person is never duplicated — see _merge_into_person_photo.
    # Only the cutout that is actually used gets the (Qwen) body-completion, so
    # no diffusion work is wasted.
    # Person A / Person B: the chosen person's photo is the base scene; the
    # other person is placed into edge-filled space beside them and everything
    # is blended in ONE Qwen pass (no separate normalize/outpaint passes — keeps
    # it fast). Body-completion only runs if a subject is actually clipped.
    if bg_choice == "Person A":
        print(f"[merge] Person A: complete A on-photo, place B to the {add_side}...")
        base_a = _complete_body_on_photo(pil_a)          # missing parts only (skipped if whole)
        rgba_b = _complete_body_safe(rgba_b)             # missing parts only (skipped if whole)
        return _merge_into_person_photo(base_a, rgba_b, add_side=add_side,
                                        gender_base=gender_a, gender_other=gender_b)
    if bg_choice == "Person B":
        print(f"[merge] Person B: complete B on-photo, place A to the {add_side}...")
        base_b = _complete_body_on_photo(pil_b)          # missing parts only (skipped if whole)
        rgba_a = _complete_body_safe(rgba_a)             # missing parts only (skipped if whole)
        return _merge_into_person_photo(base_b, rgba_a, add_side=add_side,
                                        gender_base=gender_b, gender_other=gender_a)

    # All other backgrounds (white / Qwen scene): body-complete both cutouts,
    # then lay them out side by side. No original pixels are altered.
    print("[merge] completing clipped bodies (missing parts only)...")
    rgba_a = _complete_body_safe(rgba_a)
    rgba_b = _complete_body_safe(rgba_b)

    OUT_W, OUT_H = 1280, 720
    SLOT_W = OUT_W // 2
    MAX_H = int(OUT_H * 0.95)

    # BOTH subjects are always scaled to fit fully inside their half-canvas
    # slot. This is what guarantees both full bodies are visible and neither
    # person is oversized or cropped by the canvas edges — regardless of which
    # background is chosen. (Previously the "Person A/B" options pasted that
    # person's full-resolution photo as the background, so they appeared huge
    # and cropped. Now they are a fitted cutout like everyone else, and their
    # photo is only used as a soft backdrop scene behind both people.)
    rgba_a = _scale_subject_to_fit(rgba_a, SLOT_W - 8, MAX_H)
    rgba_b = _scale_subject_to_fit(rgba_b, SLOT_W - 8, MAX_H)

    baseline = OUT_H - int(OUT_H * 0.02)

    y_a = baseline - rgba_a.height
    x_a = max(0, SLOT_W - rgba_a.width + 4)
    y_b = baseline - rgba_b.height
    x_b = SLOT_W - 4

    canvas = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    canvas.paste(rgba_a, (x_a, y_a), rgba_a)
    canvas.paste(rgba_b, (x_b, y_b), rgba_b)
    return _apply_merge_background(canvas, bg_choice, pil_a, pil_b, extra_prompt)


def get_num_frames(duration_seconds: float) -> int:
    """
    Frame count for a duration, snapped to the NEAREST 4n+1 layout Wan's VAE
    requires and capped at MAX_FRAMES_MODEL (97 frames = ~6s) to stay within
    the quality window before prompt degradation occurs.

    Snapping to the nearest 4n+1 (rather than always rounding down) keeps each
    segment's real length as close as possible to what was asked for, so the
    chained total doesn't drift short. e.g. 6.06s (97 frames) is reachable
    instead of collapsing to 93 frames / 5.81s.
    """
    raw = int(round(float(duration_seconds) * FIXED_FPS))
    raw = int(np.clip(raw, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
    # nearest n such that frames = 4n+1
    n = round((raw - 1) / 4)
    frames = int(n) * 4 + 1
    return max(9, min(MAX_FRAMES_MODEL, frames))



MODE_KEEP = "Keep original scene"
MODE_REPLACE = "Replace background / environment"
MODE_CUSTOM = "Custom edit instruction"
MODE_AUTORUN = "Autorun"
MODE_SEQUENCE = "Sequence"
MODE_CUSTOM_SEQ = "Custom edit sequence"
MODE_LIP_SYNC = "Lip-Synced Speaking"
SCENE_MODES = [MODE_KEEP, MODE_REPLACE, MODE_CUSTOM, MODE_AUTORUN, MODE_SEQUENCE, MODE_CUSTOM_SEQ, MODE_LIP_SYNC]
SEQUENCE_MAX_SLOTS = 10
CUSTOM_SEQ_MAX_SLOTS = 10

AUTORUN_DIR = Path(SCRIPT_DIR) / "autorun"
AUTORUN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PUSH_LOCAL_DOWNLOAD_DIR = r"D:\Apps\newgen\downloads"


def discover_autorun_images():
    """
    Return a sorted list of image paths from the autorun folder.
    Raises gr.Error if the folder is missing or contains no supported images.
    """
    if not AUTORUN_DIR.exists():
        raise gr.Error(
            f"Autorun folder not found: {AUTORUN_DIR}  "
            "Create an 'autorun' directory next to app.py and add images."
        )
    files = [
        f for f in AUTORUN_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUTORUN_EXTENSIONS
    ]
    if not files:
        raise gr.Error(
            f"No supported images found in {AUTORUN_DIR}  "
            "Add .jpg, .jpeg, .png, or .webp files and try again."
        )
    return sorted(files, key=lambda p: p.name.lower())

RELOCATE_INSTRUCTION = (
    "Keep the people exactly as they are  identical faces, facial features, "
    "hair, body shape, proportions and skin tone. Do not alter their identity "
    "or their bodies. Replace the entire background and environment with: "
    "{prompt}. Relight the subjects to match the new environment."
)

_vidgen_cache = {
    "resized_images": {},        # keyed by image hash + resolution
}
_vidgen_cache_lock = threading.Lock()
MAX_VIDGEN_CACHE = 10


def _hash_pil_image(img):
    """Fast hash of a PIL image."""
    if isinstance(img, str):
        hasher = hashlib.sha256()
        hasher.update(img.encode())
        try:
            hasher.update(str(os.path.getsize(img)).encode())
        except OSError:
            pass
        return hasher.hexdigest()
    hasher = hashlib.sha256()
    hasher.update(f"{img.size}".encode())
    img_array = np.array(img.resize((64, 64), Image.LANCZOS))
    hasher.update(img_array.tobytes())
    return hasher.hexdigest()


def _get_cached_resized(image, resolution):
    """Get cached resized image if available."""
    img_hash = _hash_pil_image(image)
    key = (img_hash, resolution)
    with _vidgen_cache_lock:
        return _vidgen_cache["resized_images"].get(key)


def _cache_resized(image, resolution, resized):
    """Cache resized image."""
    img_hash = _hash_pil_image(image)
    key = (img_hash, resolution)
    with _vidgen_cache_lock:
        cache = _vidgen_cache["resized_images"]
        if len(cache) >= MAX_VIDGEN_CACHE:
            cache.pop(next(iter(cache)))
        cache[key] = resized


def edit_reference_frame(
    image: Image.Image,
    mode: str,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
) -> Image.Image:
    """
    Optional stage 1  Qwen Image Edit prepares the starting frame.

    Returns the image untouched in keep-scene mode, so nothing is repainted and
    the original background and bodies survive exactly as shot.
    """
    if mode in (MODE_KEEP, MODE_AUTORUN, MODE_SEQUENCE, MODE_CUSTOM_SEQ):
        print("[1/2] Keep/Autorun/Sequence/CustomSeq mode  reference frame used as-is.")
        return image

    if mode == MODE_CUSTOM:
        instruction = f"Edit this image to match the following description, keeping the people's identities and features intact: {prompt}"
    else:
        instruction = RELOCATE_INSTRUCTION.format(prompt=prompt)

    print(f"[1/2] Qwen editing frame -> {instruction[:80]}...")
    
    swap_start = time.time()
    activate_pic()
    swap_time = time.time() - swap_start
    if swap_time > 3.0:
        print(f" Model swap completed in {swap_time:.1f}s")

    torch.cuda.set_device(PIC_DEVICE)
    with torch.cuda.device(PIC_DEVICE):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = pic_pipe(
                image=[image],
                prompt=instruction,
                negative_prompt=" ",
                num_inference_steps=int(steps),
                true_cfg_scale=float(guidance),
                generator=torch.Generator(device=PIC_DEVICE).manual_seed(seed),
            )
    edited = result.images[0]
    if edited.size != image.size:
        edited = edited.resize(image.size, Image.LANCZOS)
    return edited


def animate_frame(
    frame: Image.Image,
    last_frame,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    seed: int,
    flow_shift: float = None,
    wan_steps: int = None,
    lora_selections=None,
    selected_loras_info=None,
    progress=None,
):
    """Stage 2  animate a frame with WAMU v2 merged model."""
    activate_wan()

    selected = {}
    if lora_selections:
        selected = {k: v for k, v in AVAILABLE_LORAS.items() if lora_selections.get(k, False)}

    currently_active = set(_active_loras.keys())
    desired_active   = set(selected.keys())

    if currently_active != desired_active:
        load_loras_to_pipeline(wan_pipe, selected)

    if selected and selected_loras_info:
        original_prompt = prompt
        prompt = apply_lora_prompt_modifications(prompt, selected_loras_info)
        if prompt != original_prompt:
            print(f"[LoRA] Prompt modified: ...{prompt[-80:]}")

    steps = wan_steps if wan_steps is not None else WAN_STEPS

    with torch.cuda.device(WAN_DEVICE):
        _set_flow_shift(wan_pipe, flow_shift if flow_shift is not None else WAN_FLOW_SHIFT)

        print(f"[2/2] Wan animating {num_frames} frames at {frame.size}...")
        print(f"Seed: {seed} | Steps: {steps}")

        kwargs = dict(
            image=frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=frame.height,
            width=frame.width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=WAN_GUIDANCE,
            guidance_scale_2=WAN_GUIDANCE,
            generator=torch.Generator(device=WAN_DEVICE).manual_seed(seed),
            output_type="np",
        )

        # Drive the caller's step-progress callback (expects a 0..1 fraction of
        # THIS segment) from the diffusion pipeline's per-step callback, so the
        # output-box progress advances smoothly during denoising instead of
        # freezing at the segment's starting percentage.
        if callable(progress):
            _total_steps = max(1, int(steps))

            def _step_cb(pipe, step_index, timestep, cb_kwargs):
                try:
                    progress((step_index + 1) / _total_steps)
                except Exception:
                    pass
                return cb_kwargs

            kwargs["callback_on_step_end"] = _step_cb

        def _run(**kw):
            try:
                return wan_pipe(**kw).frames[0]
            except TypeError as e:
                # Older pipeline without callback_on_step_end support: retry
                # without the callback rather than failing.
                if "callback_on_step_end" in str(e) and "callback_on_step_end" in kw:
                    kw.pop("callback_on_step_end", None)
                    return wan_pipe(**kw).frames[0]
                raise

        if last_frame is None:
            return _run(**kwargs)

        try:
            return _run(last_image=last_frame, **kwargs)
        except TypeError as e:
            print(f"End frame not supported by this pipeline ({e}); ignoring it.")
            return _run(**kwargs)


def concatenate_videos(segment_bufs: list) -> bytes:
    """Join MP4 segment buffers using ffmpeg concat demuxer via RAM temp files.

    Writes each segment to /dev/shm/newgen/ temp files, runs ffmpeg -f concat
    -c copy to join them losslessly, reads the result back, then deletes all
    temp files. Falls back to the first segment on any error.
    """
    if len(segment_bufs) == 1:
        return segment_bufs[0]

    import subprocess as _sp

    tmp_dir = "/dev/shm/newgen"
    os.makedirs(tmp_dir, exist_ok=True)
    pid = os.getpid()
    seg_paths = []
    list_path = None
    out_path = None
    try:
        for i, seg_bytes in enumerate(segment_bufs):
            p = os.path.join(tmp_dir, "_cs_%d_%d.mp4" % (pid, i))
            with open(p, "wb") as fh:
                fh.write(seg_bytes)
            seg_paths.append(p)

        list_path = os.path.join(tmp_dir, "_cl_%d.txt" % pid)
        lines = ["file '" + p + "'" for p in seg_paths]
        with open(list_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        out_path = os.path.join(tmp_dir, "_co_%d.mp4" % pid)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", "-movflags", "+faststart", out_path,
        ]
        result = _sp.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-500:]
            print("[concat] ffmpeg error: " + err)
            return segment_bufs[0]
        with open(out_path, "rb") as fh:
            return fh.read()
    except Exception as e:
        print("[concat] failed: " + str(e))
        return segment_bufs[0]
    finally:
        cleanup = seg_paths[:]
        if list_path:
            cleanup.append(list_path)
        if out_path:
            cleanup.append(out_path)
        for p in cleanup:
            try:
                os.unlink(p)
            except Exception:
                pass


def _last_frame_of(video_buf: bytes):
    """Read the final frame of an in-memory MP4 as a PIL Image, for chaining.

    Accepts raw MP4 bytes (never a file path). Returns a PIL Image or None.
    PyAV writing to BytesIO may produce duration=0 — so we always decode
    all frames without seeking and return the last one.
    """
    import av as _av

    try:
        container = _av.open(BytesIO(video_buf), mode="r")
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            container.close()
            return None

        last_frame = None
        for frame in container.decode(video=0):
            last_frame = frame
        container.close()

        if last_frame is None:
            return None
        return last_frame.to_image()
    except Exception as e:
        print(f"_last_frame_of failed: {e}")
        return None


_last_generated_frame = None
_last_generated_frame_lock = threading.Lock()


def _extract_file_path(file_value):
    """Pull a real filesystem path out of a gr.File component value.

    gr.File values may arrive as a plain string path, an object with a
    .name attribute (older Gradio's tempfile wrapper), or a dict with
    'path'/'name'/'url' keys (newer Gradio FileData).
    """
    if not file_value:
        return None
    if isinstance(file_value, str):
        return file_value
    if hasattr(file_value, "name"):
        return file_value.name
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name") or file_value.get("url")
    return None


def _cache_last_frame_from_video(video_file):
    """Grab the last frame of the just-generated video into memory only.

    video_file is a gr.File value pointing at a real filepath on
    /root/newgen/tmp/gradio/. Reads the file, extracts the last frame,
    caches as PIL Image.
    """
    global _last_generated_frame
    path = _extract_file_path(video_file)
    if not path:
        return
    try:
        if not os.path.exists(path):
            return
        with open(path, "rb") as _vf:
            video_bytes = _vf.read()
        frame = _last_frame_of(video_bytes)
    except Exception as e:
        print(f"Could not cache last frame from video: {e}")
        frame = None
    if frame is not None:
        with _last_generated_frame_lock:
            _last_generated_frame = frame.copy()


def _use_last_generated_frame():
    """Hand the in-memory cached last frame to the Reference Photo widget.

    Reads only the in-memory cache populated by
    _cache_last_frame_from_video() -- no disk access happens here. If
    nothing has been generated yet this session (or the process was
    restarted since), the cache is empty and the current Reference Photo
    is left untouched.
    """
    with _last_generated_frame_lock:
        frame = _last_generated_frame
    if frame is None:
        gr.Warning("No generated video yet this session -- generate a video first.")
        return gr.update()
    return frame.copy()


def generate_with_preset(prompt_dict, choice, reference_image, scene_mode,
                        end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, audio_negative_prompt_tb,
                        ref_audio_path, dialogue_text, vid_negative_prompt,
                        edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    """Generate video using preset prompt without changing the prompt textbox."""
    if choice and choice in prompt_dict:
        preset_prompt = prompt_dict[choice]
        return generate_video(
            reference_image, preset_prompt, scene_mode,
            end_image, duration_seconds, resolution, frame_multiplier,
            export_quality, seed, randomize_seed, add_audio_cb,
            audio_prompt_tb, audio_negative_prompt_tb,
            ref_audio_path, dialogue_text, vid_negative_prompt,
            edit_steps, edit_guidance, flow_shift_auto, flow_shift
        )
    return None, None, gr.update()

def generate_with_solo(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                      export_quality, seed, randomize_seed, add_audio_cb,
                      audio_prompt_tb, audio_negative_prompt_tb,
                      ref_audio_path, dialogue_text, vid_negative_prompt,
                      edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_solo_prompts_dict, choice, reference_image, scene_mode,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_couple(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, audio_negative_prompt_tb,
                        ref_audio_path, dialogue_text, vid_negative_prompt,
                        edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_couple_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_multiple(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                          export_quality, seed, randomize_seed, add_audio_cb,
                          audio_prompt_tb, audio_negative_prompt_tb,
                          ref_audio_path, dialogue_text, vid_negative_prompt,
                          edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_multistep(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                           export_quality, seed, randomize_seed, add_audio_cb,
                           audio_prompt_tb, audio_negative_prompt_tb,
                           ref_audio_path, dialogue_text, vid_negative_prompt,
                           edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multistep_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_environment(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                             export_quality, seed, randomize_seed, add_audio_cb,
                             audio_prompt_tb, audio_negative_prompt_tb,
                             ref_audio_path, dialogue_text, vid_negative_prompt,
                             edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_environment_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_custom(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, audio_negative_prompt_tb,
                        ref_audio_path, dialogue_text, vid_negative_prompt,
                        edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_custom_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_multiple_unseen(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                                 export_quality, seed, randomize_seed, add_audio_cb,
                                 audio_prompt_tb, audio_negative_prompt_tb,
                                 ref_audio_path, dialogue_text, vid_negative_prompt,
                                 edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_unseen_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)

def generate_with_multiple_seen(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_seen_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, audio_negative_prompt_tb,
                               ref_audio_path, dialogue_text, vid_negative_prompt,
                               edit_steps, edit_guidance, flow_shift_auto, flow_shift)


def generate_video(
    reference_image,
    prompt,
    scene_mode,
    end_image=None,
    duration_seconds=6.0,
    resolution="480p",
    frame_multiplier=16,
    export_quality=10,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="quiet ambience, soft room tone",
    audio_negative_prompt_tb="music, noise, wind, crowd",
    ref_audio_path=None,
    dialogue_text="",
    voice_speed=0.8,
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,  # Variable number of LoRA checkbox states
    progress=gr.Progress(track_tqdm=True),
):
    """
    Photo(s) + prompt -> video.

    Stage 1 (optional) prepares the starting frame with Qwen Image Edit.
    Stage 2 animates it with WAMU v2 4-step Lightning (NSFW merge), chaining
    ~5.1s segments for anything longer than one native window.
    
    No interpolation by default (frame_multiplier=16 = native 16fps).
    """
    global _current_input_image_path
    
    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info
    
    if selected_loras_info:
        edit_steps, flow_shift, lora_settings_msg = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )
        if lora_settings_msg:
            pass
    
    if isinstance(reference_image, str) and not reference_image.startswith("/media/"):
        _current_input_image_path = reference_image
    elif hasattr(reference_image, 'filename'):
        _current_input_image_path = reference_image.filename
    else:
        _current_input_image_path = None
    _generation_protect(_current_input_image_path)

    _end_image_protect_path = None
    if isinstance(end_image, str):
        _end_image_protect_path = end_image
    elif hasattr(end_image, 'filename'):
        _end_image_protect_path = end_image.filename
    _generation_protect(_end_image_protect_path)

    try:
        reference_image = _ensure_pil(reference_image)
        end_image = _ensure_pil(end_image)
    except Exception as e:
        raise gr.Error(f"Could not read the input image(s): {e}")

    if reference_image is None:
        raise gr.Error("Please upload a reference photo.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the motion and scene.")
    
    if flow_shift_auto:
        if duration_seconds <= 6.0:
            adaptive_flow_shift = 6.9
        elif duration_seconds <= 10.0:
            adaptive_flow_shift = 5.5
        elif duration_seconds <= 20.0:
            adaptive_flow_shift = 4.5
        else:
            adaptive_flow_shift = 4.0
        
        print(f" Auto flow_shift: {adaptive_flow_shift:.1f} (duration: {duration_seconds}s)")
        flow_shift = adaptive_flow_shift
    else:
        if flow_shift is None:
            flow_shift = WAN_FLOW_SHIFT
        print(f" Manual flow_shift: {flow_shift} (user override)")

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    segment_bufs = []  # list[bytes] — in-memory MP4 segments

    try:
        if progress is not None:
            progress(0.02, desc="Preparing reference frame")
        sized = resize_image_for_wan(reference_image, resolution)


        if progress is not None:
            progress(0.10, desc="Preparing reference frame")
        start_frame = edit_reference_frame(
            sized, scene_mode, prompt,
            current_seed, edit_steps, edit_guidance,
        )

        processed_end = None
        if end_image is not None:
            processed_end = resize_and_crop_to_match(end_image, start_frame)

        remaining = float(duration_seconds)
        current_frame = start_frame
        seg_seed = current_seed
        seg_index = 0

        # Total number of segments this whole generation will produce, so the
        # progress bar reflects overall progress across ALL segments (and, for
        # a single segment, spans the whole 0.15->0.90 range as its steps run)
        # instead of freezing at the segment's start percentage.
        import math as _math
        total_segments = max(1, _math.ceil(float(duration_seconds) / SEGMENT_DURATION))
        # Progress band reserved for animation (after the 0.15 setup, before the
        # final 0.90->1.0 encode/audio).
        _P_START, _P_END = 0.15, 0.90

        while remaining > 0.01:
            seg_duration = min(remaining, SEGMENT_DURATION)
            num_frames = get_num_frames(seg_duration)
            # Actual playback length of this segment once snapped to the 4n+1
            # frame grid -- this is what actually ends up in the video, so the
            # chain must account for THIS, not the requested seg_duration, or
            # the accumulated rounding makes the final video come up short
            # (e.g. asking for 18s and getting ~15s).
            actual_seg_duration = num_frames / FIXED_FPS
            seg_index += 1
            is_last_segment = (remaining - actual_seg_duration) <= 0.01

            seg_end = processed_end if is_last_segment else None

            # Per-step progress: map this segment's denoising steps into its
            # slice of the overall [_P_START, _P_END] band across all segments.
            _seg_i0 = seg_index - 1

            def _seg_progress(step_frac, _i0=_seg_i0, _label=seg_index):
                if progress is None:
                    return
                overall = (_i0 + max(0.0, min(1.0, step_frac))) / total_segments
                frac = _P_START + (_P_END - _P_START) * overall
                if total_segments > 1:
                    desc = f"Generating segment {_label} of {total_segments}"
                else:
                    desc = "Generating video"
                progress(min(frac, _P_END), desc=desc)

            _seg_progress(0.0)
            raw_frames = animate_frame(
                current_frame, seg_end, prompt, negative_prompt,
                num_frames, seg_seed, flow_shift, edit_steps,
                lora_selections,
                selected_loras_info,
                _seg_progress,
            )

            factor = max(1, int(frame_multiplier) // FIXED_FPS)
            if factor > 1:
                seg_frames = interpolate_bits(raw_frames, multiplier=factor)
            else:
                seg_frames = list(raw_frames)
            seg_fps = FIXED_FPS * factor

            seg_buf = encode_frames_to_bytes(seg_frames, fps=seg_fps, quality=int(export_quality))
            segment_bufs.append(seg_buf)
            print(f"Segment {seg_index} complete ({actual_seg_duration:.2f}s, "
                  f"{len(seg_frames)} frames @ {seg_fps} fps)")

            # --- STREAMING: yield a growing preview after every segment ------
            # Concatenate all segments so far into a tmp file and yield it so
            # Gradio updates the video component immediately.  Only do this when
            # there are more segments still to generate (is_last_segment will get
            # the final yield below after audio is added).
            if not is_last_segment and total_segments > 1:
                try:
                    preview_buf = concatenate_videos(segment_bufs)
                    preview_name = _media_name("vidgen_preview", ".mp4")
                    preview_path = _write_video_tmp(preview_buf, preview_name)
                    yield preview_path, preview_path, gr.update(visible=False, value="")
                    # Clean up the preview tmp file — the browser has already
                    # received it and the final video will replace it.
                    _delete_video_tmp(preview_path)
                except Exception as _prev_e:
                    print(f"[Streaming] preview yield failed (non-fatal): {_prev_e}")
            # ------------------------------------------------------------------

            remaining -= actual_seg_duration
            if remaining <= 0.01:
                break

            nxt = _last_frame_of(seg_buf)
            if nxt is None:
                print("Could not read segment tail frame  stopping chain here.")
                break
            current_frame = nxt
            seg_seed = random.randint(0, MAX_SEED)

        if not segment_bufs:
            raise gr.Error("No video segments were produced.")

        final_buf = concatenate_videos(segment_bufs)

        if add_audio_cb and _AUDIO_ENGINE_AVAILABLE:
            try:
                final_buf = add_audio_to_video(
                    final_buf, audio_prompt_tb, float(duration_seconds),
                    audio_negative_prompt_tb, ref_audio_path, dialogue_text,
                    voice_speed=float(voice_speed),
                )
            except Exception as e:
                print(f"[AudioEngine] error in generate_video: {e}")

        filename = _media_name("vidgen", ".mp4")
        filepath = _write_video_tmp(final_buf, filename)
        print(f"Done in {time.time() - started:.1f}s  {seg_index} segment(s), "
              f"seed {current_seed} -> {filename}")
        if progress is not None:
            progress(1.0, desc="Generation complete")
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        yield filepath, filepath, gr.update(visible=False, value="")

    except gr.Error:
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        raise
    except Exception as e:
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        print(f"Generation error: {e}")
        raise gr.Error(f"Generation failed: {e}")


def _generate_video_blocking(*args, **kwargs):
    """Blocking wrapper around generate_video for internal callers that expect
    a plain (filepath, filepath, status) tuple rather than a generator.

    Consumes every yielded value and returns the last one, which is always the
    final full video (with audio, all segments concatenated).  Intermediate
    preview yields are discarded — internal callers (autorun, push-autorun,
    lipsync) don't have a Gradio component to stream into anyway.
    """
    result = None
    for result in generate_video(*args, **kwargs):
        pass
    return result


def generate_sequence(
    scene_images,
    scene_prompts,
    scene_durations,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="quiet ambience, soft room tone",
    audio_negative_prompt_tb="music, noise, wind, crowd",
    ref_audio_path=None,
    dialogue_text="",
    voice_speed=0.8,
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Sequence mode: up to SEQUENCE_MAX_SLOTS (image, prompt, duration) parts.
    Each part is animated (chaining internally past SEGMENT_DURATION exactly
    like normal generation) so that its FINAL segment ends on the NEXT part's
    starting image - making that image both the last frame of the current
    part and the first frame of the next part. All resulting segments across
    every part are then concatenated into a single mp4.
    """
    global _current_input_image_path

    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info

    if selected_loras_info:
        edit_steps, flow_shift, _ = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    slots = []
    protected_slot_paths = []
    for i in range(SEQUENCE_MAX_SLOTS):
        raw = scene_images[i] if i < len(scene_images) else None
        try:
            img = _ensure_pil(raw)
        except Exception as e:
            raise gr.Error(f"Sequence part {i + 1}: could not read the image ({e})")
        if img is None:
            continue
        slot_path = raw if isinstance(raw, str) else (getattr(raw, "filename", None))
        if slot_path:
            protected_slot_paths.append(slot_path)
        prompt_i = (scene_prompts[i] if i < len(scene_prompts) else "") or ""
        dur_i = float(scene_durations[i]) if i < len(scene_durations) and scene_durations[i] else 0.0
        slots.append({"image": img, "prompt": prompt_i.strip(), "duration": dur_i})

    for p in protected_slot_paths:
        _generation_protect(p)   # generation-active pin (widget changes can't strip it)

    if len(slots) < 1:
        raise gr.Error("Add at least one image to the Sequence.")
    for idx, s in enumerate(slots, start=1):
        if not s["prompt"]:
            raise gr.Error(f"Sequence part {idx}: please enter a prompt.")
        if s["duration"] <= 0:
            raise gr.Error(f"Sequence part {idx}: please set a duration greater than 0.")

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    all_segment_bufs = []  # list[bytes] — in-memory MP4 segments

    try:
        n_slots = len(slots)
        total_segments = sum(
            max(1, math.ceil((s["duration"] - 0.01) / SEGMENT_DURATION)) for s in slots
        )
        seg_counter = 0

        for slot_idx, slot in enumerate(slots):
            if progress is not None:
                progress(min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.95),
                          desc=f"Sequence part {slot_idx + 1}/{n_slots}: preparing frame")

            if flow_shift_auto:
                d = slot["duration"]
                if d <= 6.0:
                    slot_flow_shift = 6.9
                elif d <= 10.0:
                    slot_flow_shift = 5.5
                elif d <= 20.0:
                    slot_flow_shift = 4.5
                else:
                    slot_flow_shift = 4.0
            else:
                slot_flow_shift = flow_shift if flow_shift is not None else WAN_FLOW_SHIFT

            sized = resize_image_for_wan(slot["image"], resolution)
            start_frame = edit_reference_frame(
                sized, MODE_SEQUENCE, slot["prompt"], current_seed, edit_steps, edit_guidance,
            )

            target_end = None
            if slot_idx < n_slots - 1:
                next_sized = resize_image_for_wan(slots[slot_idx + 1]["image"], resolution)
                target_end = resize_and_crop_to_match(next_sized, start_frame)

            remaining = float(slot["duration"])
            current_frame = start_frame
            seg_seed = current_seed
            part_seg_index = 0

            while remaining > 0.01:
                seg_duration = min(remaining, SEGMENT_DURATION)
                num_frames = get_num_frames(seg_duration)
                actual_seg_duration = num_frames / FIXED_FPS
                part_seg_index += 1
                seg_counter += 1
                is_last_segment_of_part = (remaining - actual_seg_duration) <= 0.01
                seg_end = target_end if is_last_segment_of_part else None

                if progress is not None:
                    progress(min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.97),
                              desc=f"Sequence part {slot_idx + 1}/{n_slots}, segment {part_seg_index}")

                raw_frames = animate_frame(
                    current_frame, seg_end, slot["prompt"], negative_prompt,
                    num_frames, seg_seed, slot_flow_shift, edit_steps,
                    lora_selections, selected_loras_info, progress,
                )

                factor = max(1, int(frame_multiplier) // FIXED_FPS)
                if factor > 1:
                    seg_frames = interpolate_bits(raw_frames, multiplier=factor)
                else:
                    seg_frames = list(raw_frames)
                seg_fps = FIXED_FPS * factor

                seg_buf = encode_frames_to_bytes(seg_frames, fps=seg_fps, quality=int(export_quality))
                all_segment_bufs.append(seg_buf)
                print(f"Sequence part {slot_idx + 1}/{n_slots} segment {part_seg_index} "
                      f"complete ({actual_seg_duration:.2f}s, {len(seg_frames)} frames @ {seg_fps} fps)")

                remaining -= actual_seg_duration
                if remaining <= 0.01:
                    break

                nxt = _last_frame_of(seg_buf)
                if nxt is None:
                    print("Could not read segment tail frame  stopping chain here.")
                    break
                current_frame = nxt
                seg_seed = random.randint(0, MAX_SEED)

        if not all_segment_bufs:
            raise gr.Error("No video segments were produced.")

        final_buf = concatenate_videos(all_segment_bufs)

        total_duration = sum(s["duration"] for s in slots)
        if add_audio_cb and _AUDIO_ENGINE_AVAILABLE:
            try:
                final_buf = add_audio_to_video(final_buf, audio_prompt_tb, float(total_duration),
                                               audio_negative_prompt_tb, ref_audio_path, dialogue_text,
                                               voice_speed=float(voice_speed))
            except Exception as e:
                print(f"[AudioEngine] error in generate_sequence: {e}")

        filename = _media_name("vidgen_sequence", ".mp4")
        filepath = _write_video_tmp(final_buf, filename)
        print(f"Sequence done in {time.time() - started:.1f}s  {n_slots} part(s) -> {filename}")
        if progress is not None:
            progress(1.0, desc="Sequence generation complete")
        for p in protected_slot_paths:
            _generation_release(p)
        return filepath, filepath, gr.update(visible=False, value="")

    except gr.Error:
        for p in protected_slot_paths:
            _generation_release(p)
        raise
    except Exception as e:
        for p in protected_slot_paths:
            _generation_release(p)
        print(f"Sequence generation error: {e}")
        raise gr.Error(f"Sequence generation failed: {e}")



def generate_custom_edit_sequence(
    reference_image,
    cs_motion_prompts,
    cs_picgen_prompts,
    cs_durations,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="quiet ambience, soft room tone",
    audio_negative_prompt_tb="music, noise, wind, crowd",
    ref_audio_path=None,
    dialogue_text="",
    voice_speed=0.8,
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Custom Edit Sequence: single starting image, N segments each with a
    motion prompt (for vidgen), a picgen prompt (to generate the last frame),
    and a duration. Each segment's generated last frame becomes the next
    segment's first frame.
    """
    global _current_input_image_path

    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info

    if selected_loras_info:
        edit_steps, flow_shift, _ = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    if isinstance(reference_image, str):
        _current_input_image_path = reference_image
    elif hasattr(reference_image, 'filename'):
        _current_input_image_path = getattr(reference_image, 'filename', None)
    else:
        _current_input_image_path = None
    _protect_path(_current_input_image_path)

    try:
        first_image = _ensure_pil(reference_image)
    except Exception as e:
        raise gr.Error(f"Could not read the reference image: {e}")
    if first_image is None:
        raise gr.Error("Please upload a reference photo.")

    slots = []
    for i in range(CUSTOM_SEQ_MAX_SLOTS):
        motion_p = cs_motion_prompts[i].strip() if i < len(cs_motion_prompts) else ""
        picgen_p = cs_picgen_prompts[i].strip() if i < len(cs_picgen_prompts) else ""
        dur = cs_durations[i] if i < len(cs_durations) else 0.0
        if not motion_p and not picgen_p:
            continue
        if not motion_p:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please enter a motion prompt.")
        if not picgen_p:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please enter a picgen (last image) prompt.")
        if dur <= 0:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please set a duration greater than 0.")
        slots.append({"motion": motion_p, "picgen": picgen_p, "duration": dur})

    if not slots:
        raise gr.Error("Add at least one segment to the Custom Edit Sequence.")

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    all_segment_bufs = []  # list[bytes] — in-memory MP4 segments
    n_slots = len(slots)
    total_segments = sum(
        max(1, math.ceil((s["duration"] - 0.01) / SEGMENT_DURATION)) for s in slots
    )
    seg_counter = 0

    current_first = first_image

    try:
        for slot_idx, slot in enumerate(slots):
            if progress is not None:
                progress(
                    min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.92),
                    desc=f"Custom seq segment {slot_idx + 1}/{n_slots}: generating last frame with picgen"
                )

            sized_first = resize_image_for_wan(current_first, resolution)
            activate_pic()
            torch.cuda.set_device(PIC_DEVICE)
            pic_generator = torch.Generator(device=PIC_DEVICE).manual_seed(current_seed)
            try:
                with torch.cuda.device(PIC_DEVICE):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pic_result = pic_pipe(
                            image=[current_first],
                            prompt=slot["picgen"],
                            negative_prompt=" ",
                            num_inference_steps=int(edit_steps),
                            true_cfg_scale=float(edit_guidance),
                            generator=pic_generator,
                        )
                generated_last = pic_result.images[0]
            except Exception as e:
                raise gr.Error(f"Custom edit sequence segment {slot_idx + 1}: picgen failed: {e}")

            processed_end = resize_and_crop_to_match(generated_last, sized_first)

            if progress is not None:
                progress(
                    min(0.05 + 0.9 * (seg_counter + 0.5) / max(1, total_segments), 0.95),
                    desc=f"Custom seq segment {slot_idx + 1}/{n_slots}: animating"
                )

            if flow_shift_auto:
                d = slot["duration"]
                if d <= 6.0:
                    slot_flow_shift = 6.9
                elif d <= 10.0:
                    slot_flow_shift = 5.5
                elif d <= 20.0:
                    slot_flow_shift = 4.5
                else:
                    slot_flow_shift = 4.0
            else:
                slot_flow_shift = flow_shift if flow_shift is not None else WAN_FLOW_SHIFT

            remaining = float(slot["duration"])
            current_frame = sized_first
            seg_seed = current_seed
            part_seg_index = 0

            while remaining > 0.01:
                seg_duration = min(remaining, SEGMENT_DURATION)
                num_frames = get_num_frames(seg_duration)
                actual_seg_duration = num_frames / FIXED_FPS
                part_seg_index += 1
                seg_counter += 1
                is_last_segment_of_part = (remaining - actual_seg_duration) <= 0.01
                seg_end = processed_end if is_last_segment_of_part else None

                if progress is not None:
                    progress(
                        min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.97),
                        desc=f"Custom seq {slot_idx + 1}/{n_slots}, vidgen seg {part_seg_index}"
                    )

                raw_frames = animate_frame(
                    current_frame, seg_end, slot["motion"], negative_prompt,
                    num_frames, seg_seed, slot_flow_shift, edit_steps,
                    lora_selections, selected_loras_info, progress,
                )

                factor = max(1, int(frame_multiplier) // FIXED_FPS)
                if factor > 1:
                    seg_frames = interpolate_bits(raw_frames, multiplier=factor)
                else:
                    seg_frames = list(raw_frames)
                seg_fps = FIXED_FPS * factor

                seg_buf = encode_frames_to_bytes(seg_frames, fps=seg_fps, quality=int(export_quality))
                all_segment_bufs.append(seg_buf)
                print(f"Custom seq {slot_idx + 1}/{n_slots} vidgen seg {part_seg_index} "
                      f"complete ({actual_seg_duration:.2f}s, {len(seg_frames)} frames @ {seg_fps} fps)")

                remaining -= actual_seg_duration
                if remaining <= 0.01:
                    break

                nxt = _last_frame_of(seg_buf)
                if nxt is None:
                    print("Could not read segment tail frame — stopping chain here.")
                    break
                current_frame = nxt
                seg_seed = random.randint(0, MAX_SEED)

            current_first = generated_last
            current_seed = random.randint(0, MAX_SEED)

        if not all_segment_bufs:
            raise gr.Error("No video segments were produced.")

        final_buf = concatenate_videos(all_segment_bufs)

        total_duration = sum(s["duration"] for s in slots)
        if add_audio_cb and _AUDIO_ENGINE_AVAILABLE:
            try:
                final_buf = add_audio_to_video(final_buf, audio_prompt_tb, float(total_duration),
                                               audio_negative_prompt_tb, ref_audio_path, dialogue_text,
                                               voice_speed=float(voice_speed))
            except Exception as e:
                print(f"[AudioEngine] error in generate_custom_edit_sequence: {e}")

        filename = _media_name("vidgen_custom_seq", ".mp4")
        filepath = _write_video_tmp(final_buf, filename)
        _cache_last_frame_from_video(filepath)
        print(f"Custom edit sequence done in {time.time() - started:.1f}s — "
              f"{n_slots} segment(s) -> {filename}")
        if progress is not None:
            progress(1.0, desc="Custom edit sequence complete")
        return filepath, filepath, gr.update(visible=False, value="")

    except gr.Error:
        raise
    except Exception as e:
        print(f"Custom edit sequence error: {e}")
        raise gr.Error(f"Custom edit sequence failed: {e}")



def autorun_generate(
    prompt,
    scene_mode,          # will be MODE_AUTORUN but we still accept it for wiring
    end_image,           # ignored in autorun
    duration_seconds,
    resolution,
    frame_multiplier,
    export_quality,
    seed,
    randomize_seed,
    add_audio_cb,
    audio_prompt_tb,
    audio_negative_prompt_tb,
    ref_audio_path=None,
    dialogue_text="",
    voice_speed=0.8,
    vid_negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Autorun mode: iterate every image in AUTORUN_DIR exactly once,
    generate one video per image using the current settings, and yield
    a (video_path, video_path, status_text) tuple after each generation
    so the Gradio event chain can download+cleanup between items.

    End frame is intentionally ignored; each autorun image is its own start frame.
    """
    global _current_input_image_path

    autorun_files = discover_autorun_images()   # raises gr.Error early if invalid
    total = len(autorun_files)
    completed = 0

    for idx, img_path in enumerate(autorun_files, start=1):
        status = f"Autorun: {idx}/{total} â€” processing {img_path.name}"
        print(f"\n[Autorun] {status}")

        _current_input_image_path = str(img_path)

        try:
            reference_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise gr.Error(
                f"Autorun stopped at {idx}/{total}: "
                f"failed to open {img_path.name} â€” {e}"
            )

        try:
            video_path, _, _ = _generate_video_blocking(
                reference_image,
                prompt,
                MODE_KEEP,        # Autorun always uses Keep (no Qwen edit)
                None,             # end_image ignored
                duration_seconds,
                resolution,
                frame_multiplier,
                export_quality,
                seed,
                randomize_seed,
                add_audio_cb,
                audio_prompt_tb,
                audio_negative_prompt_tb,
                ref_audio_path,
                dialogue_text,
                vid_negative_prompt,
                edit_steps,
                edit_guidance,
                flow_shift_auto,
                flow_shift,
                *lora_args,
                progress=progress,
            )
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(
                f"Autorun stopped at {idx}/{total}: "
                f"failed to process {img_path.name} â€” {e}"
            )

        completed += 1
        completion_status = (
            f"Autorun complete: {completed}/{total} videos generated"
            if completed == total
            else f"Autorun: {completed}/{total} done, downloading…"
        )

        yield None, video_path, completion_status

        if completed < total:
            _vkey = video_path.split("/")[2] if video_path and video_path.startswith("/media/") else None
            if _vkey:
                with _media_store_lock:
                    entry = _media_store.get(_vkey)
                    if entry:
                        _media_store[_vkey] = (entry[0], entry[1].replace("vidgen_", "_protected_vidgen_", 1))
            _current_input_image_path = str(img_path)  # protect next input's disk path
            time.sleep(5)                               # let browser Blob-cache the video
            if _vkey:
                with _media_store_lock:
                    entry = _media_store.get(_vkey)
                    if entry:
                        _media_store[_vkey] = (entry[0], entry[1].replace("_protected_vidgen_", "vidgen_", 1))
            _do_clear_storage()    # release _media_store entries + Gradio RAM uploads



PICGEN_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BASE_MODEL_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

if DUAL_GPU:
    print(f" DUAL GPU: Loading Wan -> {WAN_DEVICE} and Qwen -> {PIC_DEVICE} simultaneously...")
    
    def _load_wan_thread():
        global wan_pipe_primary
        t = time.time()
        torch.cuda.set_device(WAN_DEVICE)
        wan_pipe_primary = _load_wan(WAN_DEVICE)
        print(f" WAN ready on {WAN_DEVICE} in {time.time()-t:.1f}s")
    
    def _load_qwen_thread():
        global pic_pipe
        t = time.time()
        torch.cuda.set_device(PIC_DEVICE)
        print(" Loading Qwen Image Edit pipeline...")
        model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
        if not os.path.exists(model_index_path):
            print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
            os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=torch.bfloat16,
                cache_dir=BASE_MODEL_LOCAL_PATH,
                use_safetensors=True
            )
        else:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                BASE_MODEL_LOCAL_PATH,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                use_safetensors=True
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
        state_dict = load_file(v23_path)
        transformer_weights, vae_weights, text_encoder_weights = {}, {}, {}
        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                transformer_weights[k.replace("model.diffusion_model.", "")] = v
            elif k.startswith("transformer."):
                transformer_weights[k.replace("transformer.", "")] = v
            elif k.startswith("first_stage_model."):
                vae_weights[k.replace("first_stage_model.", "")] = v
            elif k.startswith("vae."):
                vae_weights[k.replace("vae.", "")] = v
            elif "text_encoder" in k or "conditioner" in k:
                if "conditioner.embedders.0." in k:
                    text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
                elif "text_encoder." in k:
                    text_encoder_weights[k.replace("text_encoder.", "")] = v
        if transformer_weights:
            pipe.transformer.load_state_dict(transformer_weights, strict=False)
        if vae_weights:
            pipe.vae.load_state_dict(vae_weights, strict=False)
        if text_encoder_weights:
            pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)
        del state_dict, transformer_weights, vae_weights, text_encoder_weights
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to(PIC_DEVICE)
        pic_pipe = pipe
        print(f" Qwen ready on {PIC_DEVICE} in {time.time()-t:.1f}s")

    t_wan = threading.Thread(target=_load_wan_thread, daemon=False)
    t_qwen = threading.Thread(target=_load_qwen_thread, daemon=False)
    t_wan.start()
    t_qwen.start()
    t_wan.join()
    t_qwen.join()
    _active_model = "both"
    print(f" DUAL GPU READY  Vidgen on {WAN_DEVICE}, Picgen on {PIC_DEVICE}")

elif STARTUP_MODE == "vidgen":
    print(" VIDGEN DEFAULT: Loading Wan to GPU first for immediate use...")
    start_primary = time.time()
    
    wan_pipe_primary = _load_wan(WAN_DEVICE)
    _active_model = "wan"
    primary_load_time = time.time() - start_primary
    print(f" WAN READY ON GPU in {primary_load_time:.1f}s - Vidgen functional!")
    
    pic_pipe = None
    def _bg_qwen_load():
        global pic_pipe
        print("Background: Loading Qwen to CPU for Replace/Custom modes...")
        t = time.time()
        try:
            model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
            if not os.path.exists(model_index_path):
                print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
                os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=torch.bfloat16,
                    cache_dir=BASE_MODEL_LOCAL_PATH,
                    use_safetensors=True
                )
            else:
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    BASE_MODEL_LOCAL_PATH,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    use_safetensors=True
                )

            if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
                print(f"Downloading NSFW weights...")
                os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
                v23_path = hf_hub_download(
                    repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                    filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
                    cache_dir=PICGEN_MODELS_DIR,
                    local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
                )
            else:
                v23_path = NSFW_WEIGHTS_LOCAL_PATH

            state_dict = load_file(v23_path)
            transformer_weights = {}
            vae_weights = {}
            text_encoder_weights = {}

            for k, v in state_dict.items():
                if k.startswith("model.diffusion_model."):
                    transformer_weights[k.replace("model.diffusion_model.", "")] = v
                elif k.startswith("transformer."):
                    transformer_weights[k.replace("transformer.", "")] = v
                elif k.startswith("first_stage_model."):
                    vae_weights[k.replace("first_stage_model.", "")] = v
                elif k.startswith("vae."):
                    vae_weights[k.replace("vae.", "")] = v
                elif "text_encoder" in k or "conditioner" in k:
                    if "conditioner.embedders.0." in k:
                        text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
                    elif "text_encoder." in k:
                        text_encoder_weights[k.replace("text_encoder.", "")] = v

            if transformer_weights:
                pipe.transformer.load_state_dict(transformer_weights, strict=False)
            if vae_weights:
                pipe.vae.load_state_dict(vae_weights, strict=False)
            if text_encoder_weights:
                pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

            del state_dict, transformer_weights, vae_weights, text_encoder_weights

            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            pipe.to("cpu")
            pic_pipe = pipe
            print(f" Qwen loaded to CPU in {time.time()-t:.1f}s  Replace/Custom modes ready!")
        except Exception as e:
            print(f" Background Qwen load failed: {e}")
            pic_pipe = None
    
    threading.Thread(target=_bg_qwen_load, daemon=True).start()
    
else:
    print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")
    
    print(" AGGRESSIVE LOADING: Qwen Image Edit pipeline...")
    start_qwen = time.time()

    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
    if not os.path.exists(model_index_path):
        print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            cache_dir=BASE_MODEL_LOCAL_PATH,
            use_safetensors=True
        )
    else:
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True
        )

    print("Loading NSFW weights for Qwen...")
    if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
        print(f"Downloading NSFW weights...")
        os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
        v23_path = hf_hub_download(
            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
            cache_dir=PICGEN_MODELS_DIR,
            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
        )
    else:
        v23_path = NSFW_WEIGHTS_LOCAL_PATH

    print("Loading NSFW state dict...")
    state_dict = load_file(v23_path)

    transformer_weights = {}
    vae_weights = {}
    text_encoder_weights = {}

    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model."):
            transformer_weights[k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("transformer."):
            transformer_weights[k.replace("transformer.", "")] = v
        elif k.startswith("first_stage_model."):
            vae_weights[k.replace("first_stage_model.", "")] = v
        elif k.startswith("vae."):
            vae_weights[k.replace("vae.", "")] = v
        elif "text_encoder" in k or "conditioner" in k:
            if "conditioner.embedders.0." in k:
                text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
            elif "text_encoder." in k:
                text_encoder_weights[k.replace("text_encoder.", "")] = v

    if transformer_weights:
        pic_pipe.transformer.load_state_dict(transformer_weights, strict=False)
    if vae_weights:
        pic_pipe.vae.load_state_dict(vae_weights, strict=False)
    if text_encoder_weights:
        pic_pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

    del state_dict, transformer_weights, vae_weights, text_encoder_weights
    torch.cuda.empty_cache()

    pic_pipe.vae.enable_tiling()
    pic_pipe.vae.enable_slicing()

    pic_pipe.transformer.to(PIC_DEVICE)
    pic_pipe.text_encoder.to(PIC_DEVICE) 
    pic_pipe.vae.to(PIC_DEVICE)
    
    qwen_time = time.time() - start_qwen
    print(f" QWEN READY ON GPU in {qwen_time:.1f}s - Picgen functional!")
    _active_model = "pic"

_swap_lock = threading.Lock()


def _concurrent_component_load(component_loader_fn, device, component_name):
    """Load a single component to device concurrently."""
    print(f"    Loading {component_name} to {device}...")
    start_time = time.time()
    component_loader_fn()
    load_time = time.time() - start_time
    print(f"    {component_name} loaded in {load_time:.1f}s")
    return component_name, load_time

def _aggressive_pipeline_load(repo_id, device, pipeline_name):
    """Aggressively load pipeline with concurrent components and memory optimization."""
    print(f" AGGRESSIVE LOADING: {pipeline_name} to {device}")
    start_time = time.time()
    
    if "wan" in pipeline_name.lower():
        pipeline = WanImageToVideoPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
    
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{pipeline_name}_loader") as executor:
        futures = []
        
        if hasattr(pipeline, 'transformer'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.transformer.to(device, non_blocking=True),
                device, "transformer"
            ))
        if hasattr(pipeline, 'transformer_2'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.transformer_2.to(device, non_blocking=True),
                device, "transformer_2"
            ))
        if hasattr(pipeline, 'text_encoder'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.text_encoder.to(device, non_blocking=True),
                device, "text_encoder"
            ))
        if hasattr(pipeline, 'vae'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.vae.to(device, non_blocking=True),
                device, "vae"
            ))
        
        total_components = len(futures)
        completed_times = []
        for future in as_completed(futures):
            component_name, load_time = future.result()
            completed_times.append(load_time)
            print(f"     {component_name} ready ({len(completed_times)}/{total_components})")
    
    torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    print(f" {pipeline_name} LOADED in {total_time:.1f}s (concurrent speedup: {sum(completed_times)/total_time:.1f}x)")
    
    return pipeline


def activate_wan():
    """Ensure Wan is on WAN_DEVICE and ready."""
    global _active_model

    if DUAL_GPU:
        if not _wan_loaded or wan_pipe is None:
            _load_wan(WAN_DEVICE)
        return

    if _active_model == "wan":
        return

    print(" Fast swap to Wan...")
    start_time = time.time()

    with _swap_lock:
        if _active_model == "wan":
            return

        if _active_model == "pic" and pic_pipe is not None:
            pic_pipe.to("cpu")

        torch.cuda.empty_cache()

        if _wan_loaded and wan_pipe is not None:
            wan_pipe.to(WAN_DEVICE)
        else:
            _load_wan(WAN_DEVICE)

        _active_model = "wan"
        swap_time = time.time() - start_time
        print(f" Wan active in {swap_time:.1f}s")


def activate_pic():
    """Ensure Qwen is on PIC_DEVICE and ready."""
    global _active_model

    if DUAL_GPU:
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline not loaded  dual GPU startup failed.")
        return

    if _active_model == "pic":
        return

    if pic_pipe is None:
        print("Waiting for Qwen to finish loading in background...")
        wait_start = time.time()
        while pic_pipe is None:
            time.sleep(0.5)
            if time.time() - wait_start > 120:
                raise RuntimeError("Qwen failed to load within 120 seconds")
        print(f" Qwen background load complete, proceeding with swap")

    print(" Fast swap to Qwen...")
    start_time = time.time()

    with _swap_lock:
        if _active_model == "pic":
            return

        if _active_model == "wan" and _wan_loaded and wan_pipe is not None:
            wan_pipe.to("cpu")

        torch.cuda.empty_cache()

        pic_pipe.to(PIC_DEVICE)

        _active_model = "pic"
        swap_time = time.time() - start_time
        print(f" Qwen active in {swap_time:.1f}s")

PICGEN_MAX_SEED = np.iinfo(np.int32).max

_decode_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="b64decode")

_picgen_cache = {
    "vae_latents": {},      # keyed by image hash
    "prompt_embeds": {},    # keyed by (prompt, neg_prompt, images_hash)
}
_picgen_cache_lock = threading.Lock()
MAX_CACHE_ENTRIES = 20  # Keep last 20 to avoid memory bloat


def _hash_images(images):
    """Create a stable hash from a list of PIL images."""
    hasher = hashlib.sha256()
    for img in images:
        hasher.update(f"{img.size}".encode())
        img_array = np.array(img.resize((64, 64), Image.LANCZOS))
        hasher.update(img_array.tobytes())
    return hasher.hexdigest()


def _get_cached_vae_latents(images):
    """Get cached VAE latents for images if available."""
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        return _picgen_cache["vae_latents"].get(img_hash)


def _cache_vae_latents(images, latents):
    """Cache VAE latents for images."""
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        cache = _picgen_cache["vae_latents"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[img_hash] = latents


def _get_cached_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt):
    """Get cached prompt embeddings if available."""
    img_hash = _hash_images(images)
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        return _picgen_cache["prompt_embeds"].get(key)


def _cache_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt, embeds_data):
    """Cache prompt embeddings."""
    img_hash = _hash_images(images)
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        cache = _picgen_cache["prompt_embeds"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = embeds_data


def _find_starter_path(starter_num: int):
    """Find a starter image file by number, trying .jpg / .png / .webp."""
    for ext in (".jpg", ".png", ".webp"):
        p = os.path.join(SCRIPT_DIR, f"starters/start{starter_num}{ext}")
        if os.path.exists(p):
            return p, ext
    return None, None


def add_starter_image(starter_num):
    path, ext = _find_starter_path(starter_num)
    if path is None:
        return ""
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _decode_single_b64(b64_str):
    """Decode a single base64 image string to PIL (used by thread pool)."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if b64_str.startswith("data:image"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def b64_to_pil_list(b64_json_str):
    """Decode base64 JSON array to PIL images using thread pool for speedup."""
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    
    if len(b64_list) == 1:
        img = _decode_single_b64(b64_list[0])
        return [img] if img is not None else []
    
    try:
        from concurrent.futures import as_completed
        futures = {_decode_executor.submit(_decode_single_b64, b64_str): idx 
                   for idx, b64_str in enumerate(b64_list)}
        pil_images = [None] * len(b64_list)
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            if result is not None:
                pil_images[idx] = result
        return [img for img in pil_images if img is not None]
    except Exception as e:
        print(f"Thread pool decode failed, falling back to sequential: {e}")
        pil_images = []
        for b64_str in b64_list:
            img = _decode_single_b64(b64_str)
            if img is not None:
                pil_images.append(img)
        return pil_images


def _do_clear_storage():
    """
    Module-level storage clear helper called by infer_with_preclear(),
    autorun_generate(), and autorun_push_generate().

    With the in-memory pipeline, 'clearing storage' means:
      1. Releasing all transient _media_store entries (videos, picgen images,
         extracted frames) so their RAM is freed.
      2. Cleaning up Gradio's session upload dir (/dev/shm/newgen/gradio/)
         for any uploaded input-image temp files, respecting the protection
         system so files still needed by a running generation are kept.

    Returns the count of released/deleted items.
    """
    import shutil as _shutil

    count = 0

    import glob as _glob
    _tmp = os.path.join(SCRIPT_DIR, "tmp", "gradio")
    import time as _time
    _now = _time.time()
    for _pat in (
        _tmp + "/picgen_*.png",
        _tmp + "/extracted_frame_*.jpg",
        _tmp + "/vidgen_*.mp4",
        _tmp + "/vidgen_sequence_*.mp4",
        _tmp + "/vidgen_custom_seq_*.mp4",
        _tmp + "/_co_*.mp4",
        _tmp + "/_cs_*.mp4",
        _tmp + "/_cl_*.txt",
    ):
        for _f in _glob.glob(_pat):
            try:
                import os as _oss
                if _now - _oss.path.getmtime(_f) > 60:
                    _oss.unlink(_f); count += 1
            except Exception:
                pass
    _media_store_release_prefix("vidgen_")
    _media_store_release_prefix("vidgen_sequence_")
    _media_store_release_prefix("vidgen_custom_seq_")
    _media_store_release_prefix("picgen_")
    _media_store_release_prefix("extracted_frame_")

    for gradio_dir in [
        Path(SCRIPT_DIR) / "tmp" / "gradio",
    ]:
        if gradio_dir.exists():
            for item in gradio_dir.iterdir():
                if item.name == "vibe_edit_history":
                    continue
                if _is_protected(item):
                    continue
                try:
                    if item.is_dir():
                        _shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
            break

    return count


def infer_with_preclear(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generator wrapper around infer() that:
    1. Yields a gallery reset first so Gradio enters streaming/progress mode on
       EVERY run (not just the first).
    2. Clears old storage before generation.
    3. Yields the final result once infer() completes.
    """
    # Reset gallery -> Gradio enters streaming mode and shows progress bar
    yield gr.update(value=None), gr.update(), gr.update(value="")

    try:
        n = _do_clear_storage()
        print(f"[picgen pre-clear] cleared {n} item(s)")
    except Exception as _e:
        print(f"[picgen pre-clear] storage clear failed (non-fatal): {_e}")

    try:
        filepaths, seed_out, urls_json = infer(
            images_b64_json, prompt, negative_prompt, seed, randomize_seed,
            true_guidance_scale, num_inference_steps, height, width,
            num_images_per_prompt, progress,
        )
        yield filepaths, seed_out, urls_json
    except gr.Error:
        # Re-raise gr.Error cleanly — Gradio shows the message in the UI.
        # The exception is intentional (e.g. "no images uploaded") so we
        # don't need the full Python traceback in the terminal.
        raise
    except Exception as _infer_e:
        raise gr.Error(f"Generation failed: {_infer_e}") from None


def infer(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, PICGEN_MAX_SEED)

    _t_enter = time.time()
    activate_pic()
    _t_active = time.time()

    torch.cuda.set_device(PIC_DEVICE)
    generator = torch.Generator(device=PIC_DEVICE).manual_seed(seed)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image.") from None
    _t_decoded = time.time()

    if height == 256 and width == 256:
        height, width = None, None

    print(f"Seed: {seed} | Steps: {num_inference_steps}")
    print(f"  input images: {[im.size for im in pil_images]}")
    
    cached_embeds = _get_cached_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt)
    cache_status = "cached" if cached_embeds else "computing"
    
    print(f"  timing: activate {_t_active - _t_enter:.2f}s, "
          f"decode {_t_decoded - _t_active:.2f}s, embeds: {cache_status} "
          f"(active model: {_active_model}, dual_gpu: {DUAL_GPU})")
    _t_pipe = time.time()

    original_encode_prompt = pic_pipe.encode_prompt
    original_prepare_latents = pic_pipe.prepare_latents
    
    encode_called = [False]
    prepare_called = [False]
    
    def cached_encode_prompt(*args, **kwargs):
        encode_called[0] = True
        if cached_embeds is not None:
            return cached_embeds["prompt_embeds"], cached_embeds["prompt_embeds_mask"]
        result = original_encode_prompt(*args, **kwargs)
        embeds_data = {
            "prompt_embeds": result[0],
            "prompt_embeds_mask": result[1]
        }
        _cache_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt, embeds_data)
        return result
    
    def cached_prepare_latents(images, *args, **kwargs):
        prepare_called[0] = True
        result = original_prepare_latents(images, *args, **kwargs)
        if images is not None and result[1] is not None:
            pass
        return result
    
    pic_pipe.encode_prompt = cached_encode_prompt
    pic_pipe.prepare_latents = cached_prepare_latents
    
    try:
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image = pic_pipe(
                    image=pil_images if pil_images else None,
                    prompt=prompt,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    true_cfg_scale=true_guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                ).images
    finally:
        pic_pipe.encode_prompt = original_encode_prompt
        pic_pipe.prepare_latents = original_prepare_latents

    print(f"  pipeline call took {time.time() - _t_pipe:.2f}s")

    import os as _os
    _shm_dir = os.path.join(SCRIPT_DIR, "tmp", "gradio")
    _os.makedirs(_shm_dir, exist_ok=True)
    if not _os.path.isdir(_shm_dir):
        _shm_dir = _os.path.join(_os.environ.get("TMPDIR", "/tmp"), "picgen_out")
        _os.makedirs(_shm_dir, exist_ok=True)
    multiple = len(image) > 1
    filepaths = []
    for i, img in enumerate(image, start=1):
        filename = _media_name("picgen", ".png", index=i if multiple else None)
        fpath = _os.path.join(_shm_dir, filename)
        img.save(fpath, format="PNG")
        filepaths.append(fpath)
    print(f"  saved: {[_os.path.basename(p) for p in filepaths]}")
    urls_json = json.dumps(filepaths)

    return filepaths, seed, urls_json


gallery_js = r"""
() => {
// Keep the image array alive across re-inits so images added before a
// Gradio DOM refresh are not lost.
if (!window.__picgenImages) window.__picgenImages = [];
if (!window.__picgenSelected) window.__picgenSelected = { idx: -1 };

function init() {
    const galleryGrid  = document.getElementById('image-gallery-grid');
    const dropZone     = document.getElementById('gallery-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick  = document.getElementById('upload-click-area');
    const fileInput    = document.getElementById('custom-file-input');
    const btnUpload    = document.getElementById('tb-upload');
    const btnRemove    = document.getElementById('tb-remove');
    const btnClear     = document.getElementById('tb-clear');
    if (!galleryGrid || !fileInput || !dropZone) { setTimeout(init, 250); return; }

    // If the grid is already connected and has our listener marker, skip.
    // But if the element was replaced (e.g. Gradio re-rendered the tab),
    // we must re-attach all listeners to the new DOM nodes.
    if (galleryGrid._picgenListenerAttached && galleryGrid.isConnected) return;
    galleryGrid._picgenListenerAttached = true;

    let images = window.__picgenImages;
    let selectedIdx = window.__picgenSelected.idx;
    window.__uploadedImages = images;

    function syncToGradio() {
        if (window.__picgenSuppressSync) return;
        window.__uploadedImages = images;
        window.__picgenImages = images;
        window.__picgenSelected.idx = selectedIdx;
        const b64Array = images.map(img => img.b64);
        const container = document.getElementById('hidden-images-b64');
        if (!container) return;
        container.querySelectorAll('input,textarea').forEach(el => {
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, JSON.stringify(b64Array));
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }

    function addImage(b64, name) {
        images.push({id: Date.now() + Math.random(), b64: b64, name: name});
        renderGallery(); syncToGradio();
    }
    window.__addImage = addImage;

    // Replace all images with a single new one (used by outpaint popup)
    function replaceImages(b64, name) {
        images.length = 0;
        selectedIdx = -1;
        images.push({id: Date.now() + Math.random(), b64: b64, name: name});
        window.__uploadedImages = images;
        window.__picgenImages = images;
        renderGallery(); syncToGradio();
    }
    window.__replaceImages = replaceImages;

    function removeImage(idx) {
        images.splice(idx, 1);
        if (selectedIdx === idx) selectedIdx = -1;
        else if (selectedIdx > idx) selectedIdx--;
        renderGallery(); syncToGradio();
    }

    function clearAll() {
        images.length = 0;
        window.__uploadedImages = images;
        window.__picgenImages = images;
        selectedIdx = -1;
        renderGallery(); syncToGradio();
    }
    window.__clearAll = clearAll;

    function setSelected(idx) {
        selectedIdx = idx;
        window.__picgenSelected.idx = selectedIdx;
        // Re-render to reflect selection highlight without full DOM rebuild
        galleryGrid.querySelectorAll('.gallery-thumb').forEach((el) => {
            const i = parseInt(el.dataset.idx);
            if (i === selectedIdx) el.classList.add('selected');
            else el.classList.remove('selected');
        });
    }

    function renderGallery() {
        if (images.length === 0) {
            galleryGrid.innerHTML = ''; galleryGrid.style.display = 'none';
            if (uploadPrompt) uploadPrompt.style.display = '';
            return;
        }
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        galleryGrid.style.display = 'grid';
        let html = '';
        images.forEach((img, i) => {
            const sel = i === selectedIdx ? ' selected' : '';
            html += '<div class="gallery-thumb' + sel + '" draggable="true" data-idx="' + i + '">'
                  + '<img src="' + img.b64 + '" alt="' + (img.name||'image') + '">'
                  + '<span class="thumb-badge">#' + (i+1) + '</span>'
                  + '<button class="thumb-remove" data-remove="' + i + '" '
                  + 'style="position:absolute;top:4px;right:4px;z-index:100;'
                  + 'width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,0.8);'
                  + 'color:#fff;border:1px solid rgba(255,255,255,0.5);cursor:pointer;'
                  + 'display:flex;align-items:center;justify-content:center;font-size:11px;'
                  + 'pointer-events:auto;">\u2715</button>'
                  + '</div>';
        });
        html += '<div class="gallery-add-card" id="gallery-add-card" style="pointer-events:auto;">'
              + '<span class="add-icon">+</span><span class="add-text">Add</span></div>';
        galleryGrid.innerHTML = html;
    }

    function showLightbox(b64) {
        let lb = document.getElementById('picgen-lightbox');
        if (!lb) {
            lb = document.createElement('div');
            lb.id = 'picgen-lightbox';
            lb.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.92);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out;';
            lb.innerHTML = '<div style="position:relative;max-width:92vw;max-height:92vh;display:flex;align-items:center;justify-content:center;">'
                + '<img id="picgen-lb-img" style="max-width:92vw;max-height:92vh;width:auto;height:auto;border-radius:6px;display:block;object-fit:contain;image-rendering:auto;box-shadow:0 8px 40px rgba(0,0,0,0.7);">'
                + '<button id="picgen-lb-close" style="position:fixed;top:16px;right:20px;width:36px;height:36px;border-radius:50%;background:rgba(0,0,0,0.7);color:#fff;border:1px solid rgba(255,255,255,0.3);cursor:pointer;font-size:20px;line-height:1;display:flex;align-items:center;justify-content:center;z-index:10000;">\u00d7</button>'
                + '</div>';
            document.body.appendChild(lb);
            lb.addEventListener('click', () => { lb.style.display = 'none'; });
            document.getElementById('picgen-lb-close').addEventListener('click', (e) => { e.stopPropagation(); lb.style.display = 'none'; });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && lb.style.display === 'flex') lb.style.display = 'none'; });
        }
        document.getElementById('picgen-lb-img').src = b64;
        lb.style.display = 'flex';
    }

    function processFiles(files) {
        const maxSize = window.__picgenMaxSize || 512;
        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    let width = img.width, height = img.height;
                    if (width > height) { height = Math.round((height * maxSize) / width); width = maxSize; }
                    else { width = Math.round((width * maxSize) / height); height = maxSize; }
                    const canvas = document.createElement('canvas');
                    canvas.width = width; canvas.height = height;
                    canvas.getContext('2d').drawImage(img, 0, 0, width, height);
                    addImage(canvas.toDataURL('image/jpeg', 0.95), file.name);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    // ── Quality selector buttons ──────────────────────────────────────────────
    if (!window.__picgenMaxSize) window.__picgenMaxSize = 512;
    document.querySelectorAll('.tb-quality').forEach(btn => {
        btn.addEventListener('click', () => {
            window.__picgenMaxSize = parseInt(btn.dataset.size, 10);
            document.querySelectorAll('.tb-quality').forEach(b => b.classList.remove('tb-quality-active'));
            btn.classList.add('tb-quality-active');
        });
    });

    fileInput.addEventListener('change', (e) => { processFiles(e.target.files); e.target.value = ''; });
    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload)   btnUpload.addEventListener('click',   () => fileInput.click());

    // ── Single capture-phase delegated listener on the grid ──────────────────
    // Fires before any bubble-phase handlers on children. Handles X remove,
    // Add card, and thumb selection. Using capture ensures the dropZone's
    // bubble-phase click handler (which opens the file picker) never receives
    // events that originated inside the grid.
    galleryGrid.addEventListener('click', (e) => {
        // X button — remove by index
        const removeBtn = e.target.closest('.thumb-remove');
        if (removeBtn) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            removeImage(parseInt(removeBtn.dataset.remove, 10));
            return;
        }
        // Add card — open file picker
        if (e.target.closest('.gallery-add-card')) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            fileInput.click();
            return;
        }
        // Thumb body — toggle selection (single click) / lightbox (dblclick)
        const thumb = e.target.closest('.gallery-thumb');
        if (thumb) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            const idx = parseInt(thumb.dataset.idx, 10);
            if (!isNaN(idx)) {
                if (selectedIdx === idx) {
                    // Second click on same thumb opens lightbox
                    if (images[idx]) showLightbox(images[idx].b64);
                } else {
                    setSelected(idx);
                }
            }
            return;
        }
    }, true); // capture phase

    // Drop-zone background click opens file picker — but only when the click
    // did NOT originate inside the grid (grid events are fully stopped above).
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#image-gallery-grid')) return;
        if (e.target.closest('#upload-click-area')) return;
        fileInput.click();
    });

    // ── Drag-to-reorder (bubble phase — separate from click capture above) ────
    let _dragSrcIdx = -1;
    galleryGrid.addEventListener('dragstart', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        _dragSrcIdx = parseInt(thumb.dataset.idx, 10);
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(_dragSrcIdx));
        // Slight delay so the browser snapshot doesn't show the highlight
        setTimeout(() => thumb.classList.add('dragging'), 0);
    });
    galleryGrid.addEventListener('dragend', (e) => {
        _dragSrcIdx = -1;
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(t => {
            t.classList.remove('drag-over', 'dragging');
        });
    });
    galleryGrid.addEventListener('dragover', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        const overIdx = parseInt(thumb.dataset.idx, 10);
        if (overIdx === _dragSrcIdx) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(t => t.classList.remove('drag-over'));
        thumb.classList.add('drag-over');
    });
    galleryGrid.addEventListener('dragleave', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (thumb) thumb.classList.remove('drag-over');
    });
    galleryGrid.addEventListener('drop', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        e.preventDefault();
        const destIdx = parseInt(thumb.dataset.idx, 10);
        if (_dragSrcIdx < 0 || destIdx === _dragSrcIdx) return;
        // Move image from srcIdx to destIdx
        const moved = images.splice(_dragSrcIdx, 1)[0];
        images.splice(destIdx, 0, moved);
        // Keep selection tracking consistent
        if (selectedIdx === _dragSrcIdx) selectedIdx = destIdx;
        else if (_dragSrcIdx < selectedIdx && destIdx >= selectedIdx) selectedIdx--;
        else if (_dragSrcIdx > selectedIdx && destIdx <= selectedIdx) selectedIdx++;
        window.__picgenSelected.idx = selectedIdx;
        renderGallery();
        syncToGradio();
    });

    if (btnRemove) btnRemove.addEventListener('click', () => { if (selectedIdx >= 0) removeImage(selectedIdx); });
    if (btnClear)  btnClear.addEventListener( 'click', clearAll);
    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop',      (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files); });
    // Render any images that were already in the persistent store before this re-init
    renderGallery();
    if (images.length > 0) syncToGradio();
}
init();
// Re-run init on DOM mutations in case Gradio remounts the picgen tab's HTML
let _picgenObserverTimer = null;
const _picgenObserver = new MutationObserver(() => {
    // Debounce: only act after mutations settle (50ms quiet period)
    clearTimeout(_picgenObserverTimer);
    _picgenObserverTimer = setTimeout(() => {
        const g = document.getElementById('image-gallery-grid');
        if (g && !g._picgenListenerAttached) init();
    }, 50);
});
_picgenObserver.observe(document.body, { childList: true, subtree: true });
}
"""

css = """
body, .gradio-container { margin: 0 !important; padding: 0 !important; max-width: 100% !important; }
#col-container { margin: 0 !important; max-width: 100% !important; padding: 0 !important; }
.contain { padding: 0 !important; }
#preset-row { display: flex !important; align-items: center !important; gap: 8px !important; }
#preset-row > * { flex: 1 !important; }
#preset-row button { flex: 0 0 auto !important; min-width: 80px !important; }
#preset-row input[type="text"] { pointer-events: none !important; user-select: none !important; }
.hidden-input { display: none !important; height: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; }
#gallery-drop-zone { position: relative; min-height: 320px; overflow: auto; border: 1px solid var(--border-color-primary); border-radius: 8px; margin-bottom: 8px; }
#gallery-drop-zone.drag-over { outline: 2px solid var(--color-accent); outline-offset: -2px; }
.upload-prompt-modern { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1; }
/* The upload prompt is a centered absolute overlay. Only its inner click area
   should capture clicks; the rest must let clicks through to the thumbnails
   and their remove/add buttons underneath. */
.upload-prompt-modern { pointer-events: none; }
.upload-prompt-modern #upload-click-area, #upload-click-area { pointer-events: auto; }
/* The thumbnail grid must sit ABOVE the absolute upload prompt so its X and
   Add buttons are always clickable (previously they shared/were under the
   prompt's stacking context and got blocked). */
#image-gallery-grid { position: relative; z-index: 5; }
.gallery-thumb, .gallery-add-card { position: relative; z-index: 5; }
.upload-click-area { display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; padding: 36px 52px; border: 2px dashed var(--border-color-primary); border-radius: 16px; transition: all .2s ease; gap: 8px; }
.upload-click-area:hover { border-color: var(--color-accent); transform: scale(1.03); }
.upload-click-area svg { width: 64px; height: 64px; }
.upload-main-text { font-size: 14px; font-weight: 500; margin-top: 4px; }
.upload-sub-text { font-size: 12px; color: var(--body-text-color-subdued); }
.image-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; padding: 12px; align-content: start; }
/* Picgen tool buttons: three side by side, spaced, together full width. */
#picgen-tool-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 10px !important; width: 100% !important; margin-bottom: 8px !important; }
#picgen-tool-row > * { flex: 1 1 0 !important; min-width: 0 !important; }
#merge-bg-radio .wrap { display: flex !important; flex-wrap: nowrap !important; flex-direction: row !important; gap: 4px 8px !important; align-items: center !important; }
#merge-bg-radio .wrap label { white-space: nowrap !important; flex: 1 1 0 !important; font-size: 12px !important; }
/* Custom-mode results: 4 large landscape images side by side, full width.
   Keep this minimal — let Gradio render the gallery natively (heavy DOM
   overrides collapsed the images into blank space). We only force full width
   and let images size to their container. */
#merge-custom-gallery { width: 100% !important; }
#merge-custom-gallery img { object-fit: contain !important; }
.gallery-thumb { position: relative; aspect-ratio: 1; border-radius: 8px; overflow: visible; cursor: grab; border: 2px solid var(--border-color-primary); transition: border-color .2s ease, box-shadow .2s ease; background: var(--background-fill-primary); display: flex; align-items: center; justify-content: center; }
.gallery-thumb > img { border-radius: 6px; overflow: hidden; }
.gallery-thumb:hover { border-color: var(--color-accent); }
.gallery-thumb.selected { border-color: var(--color-accent) !important; box-shadow: 0 0 0 3px rgba(var(--color-accent-soft), .3); }
.gallery-thumb.dragging { opacity: 0.4; cursor: grabbing; }
.gallery-thumb.drag-over { border-color: #fff !important; box-shadow: 0 0 0 3px rgba(255,255,255,0.5) !important; }
.gallery-thumb img { width: 100%; height: 100%; object-fit: contain; display: block !important; }
.thumb-badge { position: absolute; top: 5px; left: 5px; background: var(--color-accent); color: #fff; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; display: block; text-align: left; }
.thumb-remove { position: absolute; top: 5px; right: 5px; width: 24px; height: 24px; background: rgba(0,0,0,.75); color: #fff; border: 1px solid rgba(255,255,255,.35); border-radius: 50%; cursor: pointer; display: flex !important; align-items: center; justify-content: center; font-size: 12px; transition: background .15s; line-height: 1; z-index: 50; padding: 0; pointer-events: auto !important; }
.gallery-thumb:hover .thumb-remove { display: flex !important; }
.thumb-remove:hover { background: #e53e3e !important; }
.gallery-add-card { aspect-ratio: 1; border-radius: 8px; border: 2px dashed var(--border-color-primary); display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; transition: all .2s ease; gap: 4px; }
.gallery-add-card:hover { border-color: var(--color-accent); }
.gallery-add-card .add-icon { font-size: 26px; font-weight: 300; }
.gallery-add-card .add-text { font-size: 12px; font-weight: 500; }
.uploader-toolbar { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.quality-toolbar { display: flex; gap: 5px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.quality-label { font-size: 11px; color: var(--body-text-color-subdued); white-space: nowrap; margin-right: 2px; }
.tb-btn { display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px; border: 1px solid var(--border-color-primary); border-radius: 6px; background: var(--background-fill-secondary); cursor: pointer; font-size: 12px; font-weight: 500; transition: all .15s; }
.tb-btn:hover { border-color: var(--color-accent); }
.tb-quality { font-size: 11px; padding: 4px 9px; }
.tb-quality-active { border-color: var(--color-accent) !important; background: var(--color-accent) !important; color: #fff !important; }
/* Input photo fits inside its container without expanding it */
#vidgen-reference { overflow: hidden !important; }
#vidgen-reference img, #vidgen-reference video { max-height: 320px !important; max-width: 100% !important; width: auto !important; height: auto !important; object-fit: contain !important; display: block !important; margin: 0 auto !important; }
/* Output video fits fully without cropping */
#generated-video { overflow: hidden !important; }
#generated-video video { max-height: 360px !important; max-width: 100% !important; width: auto !important; height: auto !important; object-fit: contain !important; display: block !important; margin: 0 auto !important; }
/* Merge result fixed height, no resize when image appears */
#merge-output-img { overflow: hidden !important; }
#merge-output-img img { max-height: 200px !important; max-width: 100% !important; width: auto !important; height: auto !important; object-fit: contain !important; display: block !important; margin: 0 auto !important; }
/* Keep the hidden video upload widget completely out of the layout */
#last-frame-upload-file { height: 0 !important; overflow: hidden !important; opacity: 0 !important; position: absolute !important; pointer-events: none !important; }
/* Show Media = off: hide actual media pixels but keep every container/upload-zone intact.
   Uses visibility:hidden so the container box stays; only the rendered image/video disappears. */
body.hide-media #vidgen-reference img,
body.hide-media #vidgen-reference video,
body.hide-media #generated-video video,
body.hide-media #end-frame-image img,
body.hide-media #end-frame-image video,
body.hide-media #image-gallery-grid img,
body.hide-media #picgen-result-gallery img,
body.hide-media #picgen-result-gallery video { visibility: hidden !important; }
/* LoRA cards grid - 3 per row, compact */
.lora-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 10px !important; padding: 4px 0 !important; }
.lora-card { border: 1px solid var(--border-color-primary) !important; border-radius: 8px !important; padding: 10px 12px !important; background: var(--background-fill-secondary) !important; display: flex !important; flex-direction: column !important; gap: 4px !important; }
.lora-card-header { display: flex !important; align-items: center !important; justify-content: space-between !important; gap: 6px !important; }
.lora-status-badges { display: flex !important; gap: 4px !important; flex-shrink: 0 !important; }
.lora-badge { font-size: 10px !important; padding: 1px 5px !important; border-radius: 3px !important; font-weight: 600 !important; white-space: nowrap !important; }
.lora-badge-ok { background: #22543d !important; color: #9ae6b4 !important; }
.lora-badge-dl { background: #744210 !important; color: #fbd38d !important; }
.lora-badge-miss { background: #742a2a !important; color: #feb2b2 !important; }
.lora-desc { font-size: 12px !important; color: var(--body-text-color-subdued) !important; line-height: 1.4 !important; margin: 2px 0 !important; }
.lora-notes { font-size: 11px !important; color: var(--body-text-color-subdued) !important; line-height: 1.35 !important; border-top: 1px solid var(--border-color-primary) !important; padding-top: 5px !important; margin-top: 2px !important; }
.lora-notes summary { cursor: pointer !important; font-size: 11px !important; font-weight: 500 !important; user-select: none !important; margin-bottom: 3px !important; }
.lora-card .gr-checkbox-label { font-size: 13px !important; font-weight: 600 !important; }
@media (max-width: 900px) { .lora-grid { grid-template-columns: repeat(2, 1fr) !important; } }
@media (max-width: 600px) { .lora-grid { grid-template-columns: 1fr !important; } }
/* Make picgen prompt textareas manually resizable */
#col-container textarea { resize: vertical !important; min-height: 60px !important; touch-action: pan-y !important; }
/* Center labels, info/description text, and dropdown option text on all components.
   Scoped away from #gallery-drop-zone so .thumb-badge/.thumb-remove are not affected. */
.gradio-container label > span:first-child,
.gradio-container .label-wrap > span,
.gradio-container legend > span { text-align: center !important; width: 100% !important; }
.gradio-container .info { text-align: center !important; }
.gradio-container select option { text-align: center !important; }
.gradio-container .wrap-inner input,
.gradio-container .secondary-wrap span { text-align: center !important; }
/* Center the Clear Storage button and Show Media checkbox at the top */
#top-bar-row { justify-content: center !important; }
#top-bar-row button { min-width: 140px !important; }
#show-media-row { justify-content: center !important; }
#show-media-row > * { flex: 0 0 auto !important; }
/* Tab bar: full width, two equal tabs, Photo Editor left, Video Generator right */
.gradio-container .tab-nav { display: flex !important; width: 100% !important; }
.gradio-container .tab-nav button { flex: 1 1 50% !important; text-align: center !important; justify-content: center !important; }
.gradio-container .tab-nav button:nth-child(1) { order: 2 !important; }
.gradio-container .tab-nav button:nth-child(2) { order: 1 !important; }
/* Center all section title / heading text (Markdown headings used as
   section titles, e.g. "### F5-TTS Voice Controls") on both tabs. */
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose h5,
.gradio-container .prose h6 { text-align: center !important; width: 100% !important; }
/* Accordion (expandable section) header bars: keep the title text centered
   but pin the expand/collapse arrow icon fully to the right edge of the
   clickable title bar. */
.gradio-container .label-wrap { display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; position: relative !important; }
.gradio-container .label-wrap > span { flex: 1 1 auto !important; text-align: center !important; }
.gradio-container .label-wrap svg { flex: 0 0 auto !important; margin-left: auto !important; position: relative !important; right: 0 !important; }
"""


_PUSH_API_PORT = 7861

_push_state = "idle"          # current state string
_push_lock   = threading.Lock()
_push_image_queue: "_queue.Queue[tuple]" = _queue.Queue(maxsize=1)  # (filename, PIL.Image)
_push_cancel = threading.Event()
_push_pending_video: dict = {"bytes": None, "name": None, "ready": False}


def _set_push_state(s: str):
    global _push_state
    with _push_lock:
        _push_state = s
    print(f"[AutorunAPI] state -> {s}")


def _start_push_api():
    """Run a minimal HTTP server for the push API in a daemon thread."""
    import http.server
    import urllib.parse

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence default access log

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/autorun/status":
                with _push_lock:
                    s = _push_state
                self._send_json(200, {"state": s})

            elif self.path == "/autorun/download":
                with _push_lock:
                    ready  = _push_pending_video.get("ready", False)
                    vbytes = _push_pending_video.get("bytes")
                    vname  = _push_pending_video.get("name", "vidgen.mp4")
                if not ready or not vbytes:
                    self._send_json(404, {"error": "no video pending"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(vbytes)))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{vname}"')
                self.end_headers()
                self.wfile.write(vbytes)
                with _push_lock:
                    _push_pending_video["bytes"] = None
                    _push_pending_video["ready"] = False
                print(f"[AutorunAPI] /autorun/download served {vname} "
                      f"({len(vbytes)//1024} KB) to local feeder")

            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/autorun/cancel":
                _push_cancel.set()
                _set_push_state("idle")
                self._send_json(200, {"cancelled": True})
                return

            if self.path == "/autorun/ready":
                with _push_lock:
                    state = _push_state
                if state not in ("busy", "done"):
                    self._send_json(409, {"error": f"not awaiting completion (state={state})"})
                    return
                _set_push_state("ready")
                self._send_json(200, {"ready": True})
                return

            if self.path != "/autorun/push":
                self._send_json(404, {"error": "not found"})
                return

            with _push_lock:
                state = _push_state
            if state != "ready":
                self._send_json(409, {"error": f"not ready (state={state})"})
                return

            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._send_json(400, {"error": "expected multipart/form-data"})
                return

            import cgi
            length = int(self.headers.get("Content-Length", 0))
            fs = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": str(length),
                },
            )
            file_item = fs.getvalue("file")
            filename   = fs["file"].filename if "file" in fs else "image.jpg"

            if file_item is None:
                self._send_json(400, {"error": "missing 'file' field"})
                return

            try:
                img = Image.open(BytesIO(file_item if isinstance(file_item, bytes) else file_item.read())).convert("RGB")
            except Exception as e:
                self._send_json(400, {"error": f"cannot decode image: {e}"})
                return

            _set_push_state("busy")
            _push_image_queue.put((filename, img))
            self._send_json(200, {"accepted": True, "filename": filename})

    server = http.server.HTTPServer(("127.0.0.1", _PUSH_API_PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[AutorunAPI] listening on 127.0.0.1:{_PUSH_API_PORT}  (SSH-tunnel only)")


def autorun_push_generate(
    prompt,
    duration_seconds,
    resolution,
    frame_multiplier,
    export_quality,
    seed,
    randomize_seed,
    add_audio_cb,
    audio_prompt_tb,
    audio_negative_prompt_tb,
    ref_audio_path=None,
    dialogue_text="",
    voice_speed=0.8,
    vid_negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Push-mode autorun: waits for images sent from the local machine via the
    push API, processes each one, yields status, then signals ready for next.
    Runs until cancelled or until the local feeder sends no more images
    (detected by a 60-second timeout after the last 'ready' signal).
    """
    global _current_input_image_path
    _push_cancel.clear()
    completed = 0

    _set_push_state("ready")

    while not _push_cancel.is_set():
        try:
            filename, img = _push_image_queue.get(timeout=900)
        except _queue.Empty:
            _set_push_state("idle")
            yield None, None, f"Push autorun complete: {completed} video(s) generated (timed out waiting for next image)"
            return

        if _push_cancel.is_set():
            _set_push_state("idle")
            return

        completed += 1
        status = f"Push autorun: processing {filename} (#{completed})"
        print(f"\n[AutorunPush] {status}")
        _current_input_image_path = None  # in-memory image, no path to protect

        try:
            video_path, _, _ = _generate_video_blocking(
                img,
                prompt,
                MODE_KEEP,
                None,
                duration_seconds,
                resolution,
                frame_multiplier,
                export_quality,
                seed,
                randomize_seed,
                add_audio_cb,
                audio_prompt_tb,
                audio_negative_prompt_tb,
                ref_audio_path,
                dialogue_text,
                vid_negative_prompt,
                edit_steps,
                edit_guidance,
                flow_shift_auto,
                flow_shift,
                *lora_args,
                progress=progress,
            )
        except gr.Error:
            _set_push_state("idle")
            raise
        except Exception as e:
            _set_push_state("idle")
            raise gr.Error(f"Push autorun failed on {filename}: {e}")

        push_status = f"Push autorun: {completed} done, ready for local download"
        if video_path and video_path.startswith("/media/"):
            _vkey = video_path.split("/")[2]
            _entry = _media_store_get(_vkey)
            if _entry is not None:
                _vbytes, out_name = _entry
                with _push_lock:
                    _push_pending_video["bytes"] = _vbytes
                    _push_pending_video["name"]  = out_name
                    _push_pending_video["ready"] = True
                print(f"[AutorunPush] video queued for local pull ({len(_vbytes)//1024} KB): {out_name}")
            else:
                print("[AutorunPush] could not queue video: store key not found")

        _set_push_state("done")
        yield None, video_path, push_status

        time.sleep(1)
        _do_clear_storage()
        print(f"[AutorunPush] storage cleared after #{completed} — waiting for /autorun/ready")

        while not _push_cancel.is_set():
            with _push_lock:
                s = _push_state
            if s == "ready":
                break
            time.sleep(0.25)

    _set_push_state("idle")




with gr.Blocks(css=css) as demo:
    with gr.Row(elem_id="top-bar-row"):
        clear_storage_btn = gr.Button("Clear Storage", variant="secondary", size="sm")
    clear_storage_status = gr.Textbox(visible=False, label="")

    def protect_current_inputs(reference_image, end_image, merge_a, merge_b):
        """Explicit pre-clear protection step for the automatic storage clear.

        Runs as a .then() step BEFORE clear_storage() in the generate/download
        chains: reads the first frame (reference), last frame (end), merge-photo
        inputs, and the merged result, and registers their on-disk paths in the
        protected set so the full storage wipe that follows cannot delete them
        out from under the widgets.

        Two protection layers are applied:
          1. Full path registered in _protected_image_paths (existing behaviour).
          2. Basename registered in _protected_image_filenames so that even if
             Gradio has moved or re-indexed the file, anything sharing the same
             filename is guaranteed to survive the clear.

        Accepts the same image component values Gradio hands to change
        handlers: None, "", filepath str, or objects with .name/.filename.
        """

        def _path_of(img):
            if img is None or img == "":
                return None
            if isinstance(img, str):
                return img
            for attr in ("name", "filename"):
                v = getattr(img, attr, None)
                if isinstance(v, str) and v:
                    return v
            return None

        with _protected_filenames_lock:
            _protected_image_filenames.clear()

        for img in (reference_image, end_image, merge_a, merge_b):
            p = _path_of(img)
            if p and not p.startswith("/media/"):
                _protect_path(p)
                _protect_filename(p)  # layer 2: basename protection
        return None

    def clear_storage():
        """Release in-memory media store entries and clean Gradio's RAM upload dir."""
        import shutil as _shutil

        _media_store_release_prefix("vidgen_")
        _media_store_release_prefix("vidgen_sequence_")
        _media_store_release_prefix("vidgen_custom_seq_")
        _media_store_release_prefix("picgen_")
        _media_store_release_prefix("extracted_frame_")

        cleaned = 0
        for gradio_dir in [
            Path(SCRIPT_DIR) / "tmp" / "gradio",
        ]:
            if gradio_dir.exists():
                for item in gradio_dir.iterdir():
                    if item.name == "vibe_edit_history":
                        continue
                    if _is_protected(item):
                        continue
                    try:
                        if item.is_dir():
                            _shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                        cleaned += 1
                    except Exception:
                        pass
                break

        return gr.update(visible=True, value=f"✓ Cleared {cleaned} upload(s).")




    clear_storage_btn.click(
        fn=clear_storage,
        inputs=[],
        outputs=[clear_storage_status],
    )

    with gr.Row(elem_id="show-media-row"):
        show_media_cb = gr.Checkbox(
            label="Show Media",
            value=True,
            info="When checked, displays input images and generated output on screen.",
        )

    with gr.Tabs(selected=("picgen" if STARTUP_MODE == "picgen" else "vidgen")):

        with gr.Tab(" Video Generator", id="vidgen"):
            gr.Markdown(model_title())

            with gr.Row():
                with gr.Column(scale=1):
                    reference_image = gr.Image(
                        label="Reference Photo",
                        type="filepath",
                        elem_id="vidgen-reference",
                        # PNG (not Gradio's default webp) so right-clicking the
                        # imported preview > "open image in new tab" displays it
                        # inline full size instead of downloading a .webp.
                        format="png",
                    )
                    last_frame_from_video_btn = gr.Button(
                        "Last Frame from Video", size="sm",
                    )
                    vid_prompt = gr.Textbox(
                        label="Motion & Scene Prompt",
                        value=default_video_prompt,
                        lines=4,
                        placeholder=(
                            "Describe the motion, action, lighting and camera. "
                            "In Replace-background mode this also describes the new setting."
                        ),
                    )
                    vid_negative_prompt = gr.Textbox(
                        label="Negative Prompt", value=default_negative_prompt, lines=2,
                    )

                    scene_mode = gr.Radio(
                        choices=SCENE_MODES,
                        value=MODE_KEEP,
                        label="Scene Handling",
                        info=(
                            "Keep = photo animated untouched, background and bodies "
                            "identical. Replace = new environment with motion prompt. "
                            "Custom = motion prompt applied as direct edit instruction. "
                            "Autorun = process all images in the 'autorun' folder "
                            "sequentially using the current prompt and settings. "
                            "Sequence = chain up to 10 of your own image/prompt/duration "
                            "parts into one continuous video. "
                            "Custom Edit Sequence = single reference photo, N segments each "
                            "with a picgen prompt (Qwen generates the last frame) and a "
                            "motion prompt (Wan animates from current to generated frame); "
                            "uses Generation Steps and Frame-Edit Guidance sliders. "
                            "Lip-Synced Speaking = generate video then run MuseTalk "
                            "(TMElyralab) to drive mouth movements from your Dialogue Script — "
                            "requires Voice Reference Clip + Dialogue Script in the Sound section."
                        ),
                    )

                    with gr.Row():
                        duration_seconds = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=6.0, step=0.5,
                            label="Duration (seconds)",
                            info=f"Over {SEGMENT_DURATION}s is produced by chaining segments.",
                            scale=2,
                        )
                        with gr.Column(scale=1, min_width=200):
                            flow_shift_auto = gr.Checkbox(
                                label="Auto Flow Shift",
                                value=True,
                            )
                            flow_shift = gr.Slider(
                                1.0, 12.0, value=WAN_FLOW_SHIFT, step=0.1,
                                label="Flow Shift",
                                info="Lower = better prompt. Higher = more motion.",
                                interactive=False,
                            )

                    def _make_duration_setter(value):
                        def _set_duration():
                            return gr.update(value=value)
                        return _set_duration

                    with gr.Row():
                        for _dur_preset in (3.5, 6, 12, 18, 24, 30):
                            _dur_btn = gr.Button(
                                str(_dur_preset), size="sm", min_width=40,
                            )
                            _dur_btn.click(
                                fn=_make_duration_setter(_dur_preset),
                                inputs=[],
                                outputs=[duration_seconds],
                            )
                    
                    with gr.Row():
                        edit_steps = gr.Slider(
                            1, 20, value=4, step=1,
                            label="Generation Steps",
                            info="Auto-set by LoRAs or manually override",
                        )

                    with gr.Group():
                        resolution = gr.Radio(
                            choices=["240p", "480p", "600p", "720p", "1080p"], value="480p",
                            label="Resolution",
                            info="240p=ultra-fast/tiny, 480p=fast, 600p=balanced, 720p=high quality, 1080p=max (slow, VRAM heavy)",
                        )
                    
                    def update_flow_shift_interactivity(auto_enabled):
                        if auto_enabled:
                            return gr.update(interactive=False, value=WAN_FLOW_SHIFT)
                        else:
                            return gr.update(interactive=True)
                    
                    flow_shift_auto.change(
                        fn=update_flow_shift_interactivity,
                        inputs=[flow_shift_auto],
                        outputs=[flow_shift],
                    )


                with gr.Column(scale=1):
                    vidgen_progress = gr.Markdown(
                        "", visible=False, elem_id="vidgen-progress"
                    )
                    video_output = gr.Video(
                        label="Generated Video / Upload Video",
                        elem_id="generated-video",
                        autoplay=True,
                        interactive=True,
                    )
                    
                    generate_btn = gr.Button(
                        "Generate Video", variant="primary", size="lg", elem_id="generate-btn"
                    )
                    clear_storage_btn_vid = gr.Button(
                        "Clear Storage", variant="secondary", size="lg",
                    )
                    
                    with gr.Row():
                        frame_time_input = gr.Number(
                            label="Frame time (seconds) - auto-updates as video plays",
                            value=0.0,
                            minimum=0.0,
                            step=0.1,
                            scale=1,
                            elem_id="frame-time-input",
                        )
                    with gr.Row():
                        use_as_reference_btn = gr.Button(
                            "Use Frame as Reference",
                            "Use Frame as Reference",
                        )
                        download_frame_btn = gr.Button(
                            "Download Frame",
                            "Download Frame",
                            visible=False,
                        )
                    
                    with gr.Row():
                        vid_preset_dropdown = gr.Dropdown(
                            label="Solo Prompts",
                            choices=list(vid_solo_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown2 = gr.Dropdown(
                            label="Couple Prompts",
                            choices=list(vid_couple_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown3 = gr.Dropdown(
                            label="Multiple Prompts",
                            choices=list(vid_multiple_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown4 = gr.Dropdown(
                            label="Multi-Step Prompts",
                            choices=list(vid_multistep_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown5 = gr.Dropdown(
                            label="Environment Prompts",
                            choices=list(vid_environment_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown6 = gr.Dropdown(
                            label="Custom Prompts",
                            choices=list(vid_custom_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown7 = gr.Dropdown(
                            label="Multiple (Man Unseen)",
                            choices=list(vid_multiple_man_unseen_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown8 = gr.Dropdown(
                            label="Multiple (Man Seen)",
                            choices=list(vid_multiple_man_seen_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                    
                    end_image = gr.Image(
                        label="End Frame (optional - used as the true last frame, "
                              "even when the duration requires chaining multiple segments)",
                        type="filepath",
                        elem_id="end-frame-image",
                        format="png",
                    )
                    use_last_as_first_btn = gr.Button(
                        "Use as first frame",
                        variant="secondary",
                        size="sm",
                        elem_id="use-last-as-first-btn",
                    )

                    with gr.Group():
                        edit_guidance = gr.Slider(
                            1.0, 10.0, value=1.0, step=0.1,
                            label="Frame-Edit Guidance (Qwen stage)",
                        )

            with gr.Group():
                add_audio_cb = gr.Checkbox(label="Add Audio (F5-TTS + HunyuanVideo-Foley)", value=True)
                # ── Top row: all four panels side-by-side ──────────────────────
                with gr.Row():
                    audio_prompt_tb = gr.Textbox(
                        label="Sound Effects / Foley Prompt", value="quiet ambience, soft room tone",
                        lines=4, scale=1,
                    )
                    audio_negative_prompt_tb = gr.Textbox(
                        label="Audio Negative Prompt", value="music, noise, wind, crowd",
                        placeholder="e.g. music, noise, wind, crowd",
                        lines=4, scale=1,
                    )
                    with gr.Column(scale=1):
                        gr.Markdown("**Voice Cloning (optional)** — upload a 5-10s reference clip and type the dialogue to speak. Leave blank to skip.")
                        ref_audio_input = gr.File(
                            label="Voice Reference Clip (upload .wav/.mp3, 5-10s)",
                            file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                            file_count="single",
                        )
                    dialogue_text_tb = gr.Textbox(
                        label="Dialogue Script",
                        placeholder="Leave blank to skip voice cloning",
                        lines=4, scale=1,
                    )
                # ── Voice Speed ────────────────────────────────────────────────
                gr.Markdown("**Voice Speed** — controls how fast the cloned voice speaks. 1.0 is natural; try 0.75–0.85 if she sounds rushed.")
                with gr.Row():
                    voice_speed_slider = gr.Slider(
                        label="Voice Speed",
                        minimum=0.5,
                        maximum=1.5,
                        step=0.05,
                        value=0.8,
                        scale=3,
                    )
                with gr.Row():
                    voice_speed_very_slow_btn = gr.Button("🐢 Very Slow (0.6)", size="sm", scale=1)
                    voice_speed_slow_btn      = gr.Button("🐌 Slow (0.75)",     size="sm", scale=1)
                    voice_speed_normal_btn    = gr.Button("🎙️ Normal (1.0)",    size="sm", scale=1)
                    voice_speed_fast_btn      = gr.Button("⚡ Fast (1.2)",       size="sm", scale=1)
                voice_speed_very_slow_btn.click(fn=lambda: 0.6,  outputs=[voice_speed_slider])
                voice_speed_slow_btn.click(     fn=lambda: 0.75, outputs=[voice_speed_slider])
                voice_speed_normal_btn.click(   fn=lambda: 1.0,  outputs=[voice_speed_slider])
                voice_speed_fast_btn.click(     fn=lambda: 1.2,  outputs=[voice_speed_slider])

                # ── F5-TTS voice control quick-insert buttons ──────────────────
                gr.Markdown(
                    "### 🎙️ F5-TTS Voice Controls — click any button to insert into Dialogue Script"
                )
                gr.Markdown("**Punctuation for pacing & breath:**")
                with gr.Row():
                    _ins_comma  = gr.Button(", — short pause / breath beat",              size="sm")
                    _ins_period = gr.Button(". / ! / ? — longer pause, sentence boundary", size="sm")
                    _ins_ellip  = gr.Button("... — extended pause / trailing off",         size="sm")
                    _ins_emdash = gr.Button("— (em-dash) — abrupt cut / interruption",    size="sm")

                def _append_to_dialogue(current_text, insert_str):
                    """Append insert_str to whatever is already in the dialogue box."""
                    return (current_text or "") + insert_str

                _ins_comma.click(
                    fn=lambda t: _append_to_dialogue(t, ", "),
                    inputs=[dialogue_text_tb], outputs=[dialogue_text_tb],
                )
                _ins_period.click(
                    fn=lambda t: _append_to_dialogue(t, ". "),
                    inputs=[dialogue_text_tb], outputs=[dialogue_text_tb],
                )
                _ins_ellip.click(
                    fn=lambda t: _append_to_dialogue(t, "... "),
                    inputs=[dialogue_text_tb], outputs=[dialogue_text_tb],
                )
                _ins_emdash.click(
                    fn=lambda t: _append_to_dialogue(t, " — "),
                    inputs=[dialogue_text_tb], outputs=[dialogue_text_tb],
                )

                gr.Markdown(
                    "**Capitalization for emphasis:** F5 is sensitive to capitalization for stress — "
                    "`I LOVE this` will make LOVE land harder than `I love this`. Not perfect, but noticeable.\n\n"
                    "**Repetition for elongation:** Stretch a vowel by writing it as you'd say it — "
                    "`nooo`, `pleease`, `yeeees`. F5 treats the extra letters as held phonemes.\n\n"
                    "**Phonetic spelling for tricky pronunciation:** If a word sounds wrong, spell it "
                    "phonetically — `Anth-ro-pic` instead of `Anthropic`, `eye-kon` instead of `icon`. "
                    "Hyphens help segment syllables.\n\n"
                    "**Whispered / breathy tone:** F5 clones the reference voice's character — so the "
                    "clearest lever is your reference clip itself. A breathy 5-second clip → breathy output. "
                    "An excited clip → excited clone. Record different reference clips for different moods."
                )

            with gr.Group(visible=False) as sequence_group:
                gr.Markdown(
                    "**Sequence** — fill in as many of the parts below as you need "
                    "(others left empty are skipped). Each part's image is animated "
                    "with its own prompt and duration (chaining internally past "
                    f"{SEGMENT_DURATION}s automatically) and ends exactly on the next "
                    "part's image, so the whole thing plays as one continuous clip. "
                    "Uses the Generation Steps / resolution / quality settings above."
                )
                sequence_images = []
                sequence_prompts = []
                sequence_durations = []
                for _seq_i in range(SEQUENCE_MAX_SLOTS):
                    with gr.Row():
                        _seq_img = gr.Image(
                            label=f"Part {_seq_i + 1} image",
                            type="filepath",
                            scale=1,
                            elem_id=f"sequence-image-{_seq_i}",
                            format="png",
                        )
                        _seq_prompt = gr.Textbox(
                            label=f"Part {_seq_i + 1} prompt",
                            lines=2,
                            scale=2,
                            elem_id=f"sequence-prompt-{_seq_i}",
                        )
                        _seq_dur = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=6.0, step=0.5,
                            label=f"Part {_seq_i + 1} duration (s)",
                            scale=1,
                            elem_id=f"sequence-duration-{_seq_i}",
                        )
                    sequence_images.append(_seq_img)
                    sequence_prompts.append(_seq_prompt)
                    sequence_durations.append(_seq_dur)

                sequence_status = gr.Textbox(visible=False, label="")

            with gr.Group(visible=False) as custom_seq_group:
                gr.Markdown(
                    "**Custom Edit Sequence** — uses your Reference Photo (top-left) as the "
                    "starting image. For each segment below, picgen generates the last frame "
                    "from the current image using your picgen prompt, then vidgen animates from "
                    "the current image to that generated last frame using your motion prompt. "
                    "The generated last frame becomes the next segment's first image. "
                    "The main Motion & Scene Prompt above is **ignored** — each segment uses its own motion prompt."
                )
                custom_seq_motion_prompts = []
                custom_seq_picgen_prompts = []
                custom_seq_durations = []
                for _csi in range(CUSTOM_SEQ_MAX_SLOTS):
                    with gr.Row():
                        _cs_motion = gr.Textbox(
                            label=f"Seg {_csi + 1} motion prompt (vidgen)",
                            lines=2,
                            scale=2,
                            placeholder="Describe the motion/animation for this segment",
                            elem_id=f"custom-seq-motion-{_csi}",
                        )
                        _cs_picgen = gr.Textbox(
                            label=f"Seg {_csi + 1} picgen prompt (last frame)",
                            lines=2,
                            scale=2,
                            placeholder="Describe what to generate as the final image of this segment",
                            elem_id=f"custom-seq-picgen-{_csi}",
                        )
                        _cs_dur = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=6.0, step=0.5,
                            label=f"Seg {_csi + 1} duration (s)",
                            scale=1,
                            elem_id=f"custom-seq-duration-{_csi}",
                        )
                    custom_seq_motion_prompts.append(_cs_motion)
                    custom_seq_picgen_prompts.append(_cs_picgen)
                    custom_seq_durations.append(_cs_dur)

            with gr.Group(visible=False) as lip_sync_group:
                gr.Markdown(
                    "**Lip-Synced Speaking** — generates video with subject speaking, "
                    "then runs **MuseTalk** (TMElyralab) to replace the mouth region "
                    "with audio-driven synthesis locked to your Dialogue Script. "
                    "Requires a Voice Reference Clip + Dialogue Script in the Sound section above. "
                    "MuseTalk downloads automatically (several GB) on first use.\n\n"
                    "Pipeline: Wan generates video → F5-TTS generates voice WAV → "
                    "MuseTalk drives mouth → Foley SFX mixed on top."
                )
                with gr.Row():
                    lipsync_steps_sl = gr.Slider(
                        label="MuseTalk Mouth Openness (bbox_shift)",
                        minimum=-9, maximum=9, step=1, value=0,
                        info="Positive = mouth opens more, negative = opens less. 0 is a good default.",
                        scale=2,
                    )
                    lipsync_cfg_sl = gr.Slider(
                        label="MuseTalk Batch Size",
                        minimum=1, maximum=16, step=1, value=4,
                        info="Higher = faster, more VRAM. Lower this if you see CUDA out-of-memory.",
                        scale=2,
                    )

            video_file = gr.File(visible=False)
            
            lora_checkboxes = {}
            lora_download_btns = {}
            lora_example_dropdowns = {}
            with gr.Accordion("LoRA Models (Optional)", open=False):
                if AVAILABLE_LORAS:
                    gr.Markdown(
                        "**Select LoRAs to apply.** "
                        "Checkboxes enable/disable. Download missing files. "
                        "Select example prompt to load it into the prompt box."
                    )
                    
                    sorted_loras = sorted(
                        AVAILABLE_LORAS.items(),
                        key=lambda x: x[1].get('display_name', x[0]).lower()
                    )
                    
                    lora_download_status = gr.Markdown("", visible=False)
                    
                    def download_lora_handler(lora_id):
                        """Download missing LoRA files for a specific LoRA, then
                        make it usable IMMEDIATELY (no app restart / page refresh):
                        we re-discover the LoRAs so the backend sees the new
                        files, and return UI updates that enable this LoRA's
                        checkbox and hide its download button right away.

                        Returns: (status_markdown, checkbox_update, download_btn_update)
                        """
                        global AVAILABLE_LORAS, LORA_STATUS
                        config = LORA_CONFIG.get(lora_id, {})
                        results = []
                        success_count = 0
                        if config.get('high_url') and config.get('high_filename'):
                            high_path = LORA_DIR / config['high_filename']
                            if not high_path.exists():
                                success, msg = download_lora_file(config['high_url'], config['high_filename'])
                                results.append(f"**High:** {msg}")
                                if success:
                                    success_count += 1
                            else:
                                results.append(f"**High:** Already downloaded")
                        if config.get('low_url') and config.get('low_filename'):
                            low_path = LORA_DIR / config['low_filename']
                            if not low_path.exists():
                                success, msg = download_lora_file(config['low_url'], config['low_filename'])
                                results.append(f"**Low:** {msg}")
                                if success:
                                    success_count += 1
                            else:
                                results.append(f"**Low:** Already downloaded")
                        if not results:
                            return (gr.update(visible=True, value="Nothing to download"),
                                    gr.update(), gr.update())

                        # Refresh backend view so generation picks up the new files.
                        AVAILABLE_LORAS = discover_loras()
                        LORA_STATUS = check_lora_status(LORA_CONFIG)

                        # Is this LoRA now usable (at least one file present)?
                        info_now = AVAILABLE_LORAS.get(lora_id, {})
                        can_use_now = bool(info_now.get('high')) or bool(info_now.get('low'))

                        status_msg = "\n".join(results)
                        if success_count > 0 and can_use_now:
                            status_msg += f"\n\n**{success_count} file(s) downloaded — ready to use now.**"

                        # Enable the checkbox and hide the download button immediately.
                        checkbox_update = gr.update(interactive=can_use_now) if can_use_now else gr.update()
                        btn_update = gr.update(visible=False) if can_use_now else gr.update()
                        return (gr.update(visible=True, value=status_msg),
                                checkbox_update, btn_update)
                    
                    card_index = 0
                    current_row = None
                    for lora_id, lora_info in sorted_loras:
                        status = LORA_STATUS.get(lora_id, {})
                        display_name = lora_info.get('display_name', lora_id)
                        description = lora_info.get('description', '')
                        notes = lora_info.get('notes', '')
                        trigger = lora_info.get('trigger_prompt', '')
                        
                        high_exists = bool(lora_info['high'])
                        low_exists = bool(lora_info['low'])
                        high_downloadable = status.get('high_downloadable', False)
                        low_downloadable = status.get('low_downloadable', False)
                        needs_download = (not high_exists and high_downloadable) or (not low_exists and low_downloadable)
                        can_use = high_exists or low_exists
                        
                        high_status = "âœ“" if high_exists else ("â†“" if high_downloadable else "âœ—")
                        low_status = "âœ“" if low_exists else ("â†“" if low_downloadable else "âœ—")
                        high_cls = "lora-badge-ok" if high_exists else ("lora-badge-dl" if high_downloadable else "lora-badge-miss")
                        low_cls = "lora-badge-ok" if low_exists else ("lora-badge-dl" if low_downloadable else "lora-badge-miss")
                        
                        notes_html = ""
                        if notes:
                            import html as _html
                            notes_escaped = _html.escape(notes)
                            notes_html = f'<details class="lora-notes"><summary>Notes</summary><span>{notes_escaped}</span></details>'
                        
                        trigger_html = ""
                        if trigger:
                            import html as _html
                            trigger_escaped = _html.escape(trigger[:80] + ("..." if len(trigger) > 80 else ""))
                            trigger_html = f'<div class="lora-desc" style="font-size:10px;opacity:0.7;">Trigger: <code>{trigger_escaped}</code></div>'
                        
                        if card_index % 3 == 0:
                            current_row = gr.Row(elem_classes="lora-grid-row")
                            current_row.__enter__()
                        
                        with gr.Column(elem_classes="lora-card", min_width=200):
                            gr.HTML(
                                f'<div class="lora-status-badges" style="margin-bottom:2px;">'
                                f'<span class="lora-badge {high_cls}">H:{high_status}</span>'
                                f'<span class="lora-badge {low_cls}">L:{low_status}</span>'
                                f'</div>'
                                f'<div class="lora-desc">{description}</div>'
                                f'{trigger_html}'
                                f'{notes_html}'
                            )
                            
                            lora_checkboxes[lora_id] = gr.Checkbox(
                                label=display_name,
                                value=False,
                                interactive=can_use,
                            )
                            
                            if needs_download:
                                dl_parts = []
                                if not high_exists and high_downloadable:
                                    dl_parts.append("High")
                                if not low_exists and low_downloadable:
                                    dl_parts.append("Low")
                                lora_download_btns[lora_id] = gr.Button(
                                    f"â†“ Download {' + '.join(dl_parts)}",
                                    size="sm",
                                    variant="secondary",
                                )
                            
                            example_prompts = lora_info.get('example_prompts', [])
                            if example_prompts:
                                prompt_choices = {ex['name']: ex['prompt'] for ex in example_prompts}
                                lora_example_dropdowns[lora_id] = gr.Dropdown(
                                    label="Example prompt",
                                    choices=list(prompt_choices.keys()),
                                    value=None,
                                    interactive=True,
                                )
                                
                                def create_example_handler(prompts_dict):
                                    def handler(selected):
                                        if selected and selected in prompts_dict:
                                            return prompts_dict[selected]
                                        return gr.update()
                                    return handler
                                
                                lora_example_dropdowns[lora_id].change(
                                    fn=create_example_handler(prompt_choices),
                                    inputs=[lora_example_dropdowns[lora_id]],
                                    outputs=[vid_prompt],
                                )
                        
                        card_index += 1
                        if card_index % 3 == 0 or card_index == len(sorted_loras):
                            current_row.__exit__(None, None, None)
                    
                    for lora_id, btn in lora_download_btns.items():
                        # Output to: status message + this LoRA's checkbox (to
                        # enable it) + this download button (to hide it) — so the
                        # LoRA becomes usable immediately with no restart/refresh.
                        _cb = lora_checkboxes.get(lora_id)
                        _outs = [lora_download_status]
                        _outs.append(_cb if _cb is not None else lora_download_status)
                        _outs.append(btn)
                        btn.click(
                            fn=lambda lid=lora_id: download_lora_handler(lid),
                            inputs=[],
                            outputs=_outs,
                        )
                
                else:
                    gr.Markdown(
                        "**No LoRAs configured.**  \n"
                        f"Add LoRA entries to `{LORA_CONFIG_FILE}` or place `.safetensors` files in `{LORA_DIR}` and restart."
                    )
            
            lora_compat_status = gr.Markdown(
                "", 
                visible=False,
                elem_classes="lora-compat-status"
            )
            
            def update_lora_compatibility_and_steps(*checkbox_states):
                """Show compatibility status and update steps slider when LoRAs are selected."""
                if not AVAILABLE_LORAS:
                    return gr.update(visible=False, value=""), gr.update()
                
                selected = {}
                for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), checkbox_states):
                    if is_enabled:
                        selected[lora_id] = lora_info
                
                if not selected:
                    return gr.update(visible=False, value=""), gr.update(value=4)
                
                is_compatible, message, settings = check_lora_compatibility(selected)
                
                status_lines = [f"### Active LoRAs: {len(selected)}"]
                status_lines.append(message)
                
                if settings.get('recommended_steps'):
                    status_lines.append(f"**Recommended Steps:** {settings['recommended_steps']}")
                if settings.get('recommended_flow_shift'):
                    status_lines.append(f"**Recommended Flow Shift:** {settings['recommended_flow_shift']}")
                
                if len(selected) > 1:
                    status_lines.append(f"**Average Weights:** High={settings['high_weight']:.2f}, Low={settings['low_weight']:.2f}")
                
                if not is_compatible:
                    status_lines.append("\n**Note:** Settings conflict - using defaults. You can manually override.")
                
                recommended_steps = settings.get('recommended_steps')
                
                return gr.update(visible=True, value="\n".join(status_lines)), gr.update(value=recommended_steps) if recommended_steps is not None else gr.update()
            
            if lora_checkboxes:
                for checkbox in lora_checkboxes.values():
                    checkbox.change(
                        fn=update_lora_compatibility_and_steps,
                        inputs=[lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes],
                        outputs=[lora_compat_status, edit_steps],
                    )

            download_file_output = gr.File(label="Click to Download Frame", visible=False)

            with gr.Accordion("Merge Photos", open=True):
                gr.Markdown(
                    "Upload two photos. Backgrounds are removed and both subjects are "
                    "placed side by side on a white canvas (1280×720), ready to use as "
                    "a first frame for video generation."
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=140):
                        merge_img_a = gr.Image(
                            label="Person A",
                            type="pil",
                            sources=["upload", "clipboard"],
                            elem_id="merge-img-a",
                            height=200,
                            format="png",
                        )
                        merge_gender_a = gr.Radio(
                            choices=["Man", "Woman"],
                            value=None,
                            label="Person A gender (optional)",
                            info="Select Man to preserve explicit male anatomy unchanged.",
                        )
                    with gr.Column(scale=1, min_width=140):
                        merge_img_b = gr.Image(
                            label="Person B",
                            type="pil",
                            sources=["upload", "clipboard"],
                            elem_id="merge-img-b",
                            height=200,
                            format="png",
                        )
                        merge_gender_b = gr.Radio(
                            choices=["Man", "Woman"],
                            value=None,
                            label="Person B gender (optional)",
                            info="Select Man to preserve explicit male anatomy unchanged.",
                        )
                    with gr.Column(scale=2):
                        merge_output = gr.Image(
                            label="Merged Result (1280×720)",
                            type="pil",
                            interactive=False,
                            elem_id="merge-output-img",
                            height=200,
                            # Serve full-quality PNG instead of Gradio's default
                            # webp so "open image in new tab" displays it inline
                            # full size rather than downloading a .webp.
                            format="png",
                        )
                with gr.Row():
                    merge_bg_radio = gr.Radio(
                        choices=MERGE_BG_LABELS,
                        value=None,
                        label="Background  (leave unselected for all-white)",
                        elem_id="merge-bg-radio",
                    )
                with gr.Row():
                    merge_side_radio = gr.Radio(
                        choices=["Right", "Left"],
                        value="Right",
                        label="Which side to add the other person to (Person A / Person B options only)",
                        info="Right = other person appears on the right of the base photo. Left = on the left.",
                    )
                with gr.Column(visible=True) as merge_custom_prompt_row:
                    merge_custom_prompt = gr.Textbox(
                        label="Custom prompt (optional — added to whichever option you pick, or used alone with the Custom option)",
                        placeholder="Describe the background / scene or any extra detail you want…",
                        lines=2,
                    )
                # Gallery shown only for Custom mode (4 images, user picks one)
                merge_custom_gallery = gr.Gallery(
                    show_label=False,
                    type="pil",
                    columns=4,
                    rows=1,
                    # Fixed height so the 4 landscape results render as a proper
                    # side-by-side row (height=None collapsed them to blank).
                    height=260,
                    object_fit="contain",
                    visible=False,
                    elem_id="merge-custom-gallery",
                    interactive=False,
                    allow_preview=False,
                    # Serve full-quality PNG instead of Gradio's default webp so
                    # "open image in new tab" shows the image inline full size.
                    format="png",
                )
                # Tracks which gallery image the user has selected (0-indexed)
                merge_gallery_selection = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=2):
                        merge_btn = gr.Button("Merge", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        use_as_first_frame_btn = gr.Button(
                            "Use as First Frame",
                            variant="secondary",
                            size="lg",
                        )

                # The custom prompt box is ALWAYS visible now (it can be added to
                # any option, or used alone with the Custom option). On mode
                # change we only reset the gallery/selection.
                def _on_merge_bg_change(choice):
                    return (
                        gr.update(visible=True),         # merge_custom_prompt_row (always shown)
                        gr.update(visible=False),        # hide gallery on mode change
                        None,                            # reset selection state
                    )

                merge_bg_radio.change(
                    fn=_on_merge_bg_change,
                    inputs=[merge_bg_radio],
                    outputs=[merge_custom_prompt_row, merge_custom_gallery, merge_gallery_selection],
                )

                def _do_merge(a, b, bg_choice, custom_prompt, gender_a, gender_b,
                              merge_side,
                              progress=gr.Progress(track_tqdm=True)):
                    """Merge two PIL images onto the chosen background.

                    For Custom mode: generate 4 variants with pic_pipe using the
                    user-supplied prompt, return them as a gallery list.
                    For all other modes: single result returned to merge_output.

                    `progress` drives Gradio's built-in progress bar on the
                    event's output components (merge_output on the right for the
                    8 preset/Person options, merge_custom_gallery below for
                    Custom), matching how vidgen shows progress.
                    """
                    global _current_merge_output_path
                    if a is None or b is None:
                        raise gr.Error("Please upload both Person A and Person B photos.")

                    if progress is not None:
                        progress(0.02, desc="Preparing photos")

                    if bg_choice == "Custom":
                        prompt_text = (custom_prompt or "").strip()
                        if not prompt_text:
                            raise gr.Error("Please enter a custom background prompt.")

                        # Build the merged subjects on a white canvas. With
                        # bg_choice=None, merge_photos_fn already returns a
                        # flattened RGB 1280x720 image of both fitted subjects
                        # on white — exactly the base pic_pipe needs. (It is
                        # NOT an RGBA image, so we must not call .split()[3] on
                        # it — that was the "tuple index out of range" crash.)
                        OUT_W, OUT_H = 1280, 720
                        if progress is not None:
                            progress(0.10, desc="Merging subjects")
                        base = merge_photos_fn(a, b, None, gender_a=gender_a, gender_b=gender_b,
                                               add_side=(merge_side or "Right").lower())
                        if base is None:
                            raise gr.Error("Merge failed — could not process images.")
                        base = base.convert("RGB")
                        if base.size != (OUT_W, OUT_H):
                            base = base.resize((OUT_W, OUT_H), Image.LANCZOS)

                        # The user's custom prompt only needs to describe the
                        # scene — the identity-preserving wrapper AND the
                        # fit/placement instructions (both full bodies visible,
                        # correctly sized, posed naturally in the scene) are
                        # appended automatically so they don't have to include
                        # any of that themselves.
                        # Add gender-specific anatomy preservation if requested.
                        _custom_gender_notes = []
                        if gender_a and gender_a.lower() == "man":
                            _custom_gender_notes.append("the man's penis remains the same in every way, completely unchanged and still fully visible")
                        if gender_b and gender_b.lower() == "man":
                            _custom_gender_notes.append("the man's penis remains the same in every way, completely unchanged and still fully visible")
                        _custom_gender_clause = (". " + "; ".join(_custom_gender_notes)) if _custom_gender_notes else ""
                        instruction = MERGE_BG_INSTRUCTION.format(prompt=prompt_text) + MERGE_FIT_INSTRUCTION + _custom_gender_clause

                        activate_pic()
                        torch.cuda.set_device(PIC_DEVICE)
                        images = []
                        with torch.cuda.device(PIC_DEVICE):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                for _i in range(4):
                                    if progress is not None:
                                        progress(0.20 + 0.75 * (_i / 4.0),
                                                 desc=f"Generating result {_i + 1} of 4")
                                    gen = torch.Generator(device=PIC_DEVICE).manual_seed(
                                        random.randint(0, np.iinfo(np.int32).max)
                                    )
                                    result = pic_pipe(
                                        image=[base],
                                        prompt=instruction,
                                        negative_prompt=" ",
                                        num_inference_steps=4,
                                        true_cfg_scale=1.0,
                                        generator=gen,
                                    )
                                    img = result.images[0]
                                    if img.size != (OUT_W, OUT_H):
                                        img = img.resize((OUT_W, OUT_H), Image.LANCZOS)
                                    images.append(img)
                        if progress is not None:
                            progress(1.0, desc="Done")

                        # Return: single output hidden, gallery shown, selection reset
                        return (
                            gr.update(visible=False),      # merge_output
                            gr.update(value=images, visible=True),  # merge_custom_gallery
                            None,                          # reset merge_gallery_selection
                        )

                    else:
                        if progress is not None:
                            progress(0.15, desc="Merging & generating background")
                        # Any text in the always-visible custom box is combined
                        # with the chosen option's scene prompt (or used alone if
                        # no option prompt applies).
                        extra = (custom_prompt or "").strip() or None
                        result = merge_photos_fn(a, b, bg_choice or None, extra_prompt=extra,
                                                 gender_a=gender_a, gender_b=gender_b,
                                                 add_side=(merge_side or "Right").lower())
                        if result is None:
                            raise gr.Error("Merge failed — could not process images.")
                        if progress is not None:
                            progress(0.95, desc="Finalizing")
                        buf = BytesIO()
                        result.save(buf, format="PNG")
                        filename = _media_name("merge", ".png")
                        url = _media_store_put(buf.getvalue(), filename)
                        with _merge_output_lock:
                            old_url = _current_merge_output_path
                            _current_merge_output_path = url
                        if old_url:
                            _media_store_release(old_url)
                        return (
                            gr.update(value=result, visible=True),  # merge_output
                            gr.update(value=None, visible=False),   # hide gallery
                            None,                                    # reset selection
                        )

                merge_btn.click(
                    fn=_do_merge,
                    inputs=[merge_img_a, merge_img_b, merge_bg_radio, merge_custom_prompt,
                            merge_gender_a, merge_gender_b, merge_side_radio],
                    outputs=[merge_output, merge_custom_gallery, merge_gallery_selection],
                )

                # Track gallery selection via the select event
                def _on_gallery_select(evt: gr.SelectData):
                    return evt.index

                merge_custom_gallery.select(
                    fn=_on_gallery_select,
                    outputs=[merge_gallery_selection],
                )

                def _use_merged_as_first_frame(merged_img, gallery_imgs, selection, bg_choice):
                    """Push the selected result into the reference_image widget."""
                    if bg_choice == "Custom":
                        if not gallery_imgs:
                            gr.Warning("No merged results yet — click Merge first.")
                            return gr.update()
                        if selection is None:
                            gr.Warning("Click one of the 4 images to select it, then click Use as First Frame.")
                            return gr.update()
                        idx = int(selection)
                        if idx < 0 or idx >= len(gallery_imgs):
                            gr.Warning("Selection out of range — please click an image to select it.")
                            return gr.update()
                        item = gallery_imgs[idx]
                        # gallery items are (PIL.Image, caption) tuples or plain PIL images
                        pil = item[0] if isinstance(item, (list, tuple)) else item
                        return gr.update(value=pil)
                    else:
                        if merged_img is None:
                            gr.Warning("No merged result yet — click Merge first.")
                            return gr.update()
                        return gr.update(value=merged_img)

                use_as_first_frame_btn.click(
                    fn=_use_merged_as_first_frame,
                    inputs=[merge_output, merge_custom_gallery, merge_gallery_selection, merge_bg_radio],
                    outputs=[reference_image],
                )

            export_quality = gr.Slider(
                1, 10, value=10, step=1,
                label="Export Quality (1=fastest/smallest, 10=best/largest)",
            )
            generate_btn_bottom = gr.Button(
                "Generate Video", variant="primary", size="lg",
            )
            frame_multiplier = gr.State(value=16)
            seed = gr.State(value=42)
            randomize_seed = gr.State(value=True)

            vid_preset_dropdown.change(fn=update_vid_prompt1, inputs=[vid_preset_dropdown], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown2.change(fn=update_vid_prompt2, inputs=[vid_preset_dropdown2], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown3.change(fn=update_vid_prompt3, inputs=[vid_preset_dropdown3], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown4.change(fn=update_vid_prompt4, inputs=[vid_preset_dropdown4], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown5.change(fn=update_vid_prompt5, inputs=[vid_preset_dropdown5], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown6.change(fn=update_vid_prompt6, inputs=[vid_preset_dropdown6], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown7.change(fn=update_vid_prompt7, inputs=[vid_preset_dropdown7], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown8.change(fn=update_vid_prompt8, inputs=[vid_preset_dropdown8], outputs=[vid_prompt], scroll_to_output=False)

            def _noop_download(f):
                """Pass-through function for download chain."""
                return f

            def _delete_video_after_download(video_file_val):
                """Delete the tmp video file and clear storage after browser downloads it.
                
                Uses a longer sleep (30s) to ensure Gradio has fully finished postprocessing
                and the browser has had time to download the file before we delete it.
                The FileNotFoundError was caused by deleting the file too soon (3s) while
                Gradio's gr.File postprocess was still calling Path(value).stat().st_size.
                """
                import time as _t
                _t.sleep(30)
                _delete_video_tmp(_extract_file_path(video_file_val))
                _do_clear_storage()
                return gr.update(visible=True, value="Storage cleared.")

            _VID_DOWNLOAD_JS = """
            (videoFile) => {
                if (!videoFile || !videoFile.url) return videoFile;
                const a = document.createElement('a');
                a.href = videoFile.url;
                a.download = videoFile.url.split('/').pop();
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                return videoFile;
            }
            """

            autorun_status = gr.Textbox(
                label="Autorun Status", visible=False, interactive=False,
            )

            with gr.Row(visible=False) as push_autorun_row:
                push_autorun_btn = gr.Button(
                    "â–¶ Start Push Autorun (waiting for local feeder)",
                    variant="primary", size="lg",
                )
                push_cancel_btn = gr.Button("â–  Cancel", variant="stop", size="lg")

            def _scene_mode_visibility(m):
                is_seq = (m == MODE_SEQUENCE)
                is_cseq = (m == MODE_CUSTOM_SEQ)
                is_autorun = (m == MODE_AUTORUN)
                is_lipsync = (m == MODE_LIP_SYNC)
                prompt_visible = not (is_seq or is_cseq)
                return (
                    gr.update(visible=is_autorun),    # autorun_status
                    gr.update(visible=is_autorun),    # push_autorun_row
                    gr.update(visible=is_seq),         # sequence_group
                    gr.update(visible=is_cseq),        # custom_seq_group
                    gr.update(visible=prompt_visible), # vid_prompt
                    gr.update(visible=is_lipsync),     # lip_sync_group
                )

            scene_mode.change(
                fn=_scene_mode_visibility,
                inputs=[scene_mode],
                outputs=[autorun_status, push_autorun_row, sequence_group, custom_seq_group, vid_prompt, lip_sync_group],
            )

            _PUSH_AUTORUN_INPUTS = [
                vid_prompt, duration_seconds, resolution, frame_multiplier,
                export_quality, seed, randomize_seed, add_audio_cb,
                audio_prompt_tb, audio_negative_prompt_tb,
                ref_audio_input, dialogue_text_tb, voice_speed_slider,
                vid_negative_prompt, edit_steps, edit_guidance,
                flow_shift_auto, flow_shift,
            ] + [lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes]


            def _dispatch_generate(scene_mode_val, ref_image, prompt,
                                   end_img, dur, res, fmul, qual, sd, rsd,
                                   audio_cb, audio_pt, audio_neg_pt,
                                   ref_aud, dlg_txt, v_speed,
                                   neg_pt, esteps, eguid,
                                   fsa, fs,
                                   lipsync_steps, lipsync_cfg,
                                   *rest_args):
                """Route to normal generate_video, folder-Autorun, Sequence, Custom Edit Sequence, or Lip-Synced Speaking."""
                _do_clear_storage()
                n = SEQUENCE_MAX_SLOTS
                c = CUSTOM_SEQ_MAX_SLOTS
                seq_imgs = list(rest_args[0:n])
                seq_prompts = list(rest_args[n:2 * n])
                seq_durs = list(rest_args[2 * n:3 * n])
                cs_motions = list(rest_args[3 * n: 3 * n + c])
                cs_picgens = list(rest_args[3 * n + c: 3 * n + 2 * c])
                cs_durs = list(rest_args[3 * n + 2 * c: 3 * n + 3 * c])
                lora_args_inner = rest_args[3 * n + 3 * c:]

                if scene_mode_val == MODE_LIP_SYNC:
                    result = generate_lip_sync_video(
                        ref_image, prompt, scene_mode_val,
                        end_img, dur, res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
                        neg_pt, esteps, eguid, fsa, fs,
                        lipsync_steps, lipsync_cfg,
                        *lora_args_inner,
                    )
                    # Check if result is a tuple (success case) or error
                    if result and len(result) >= 3:
                        yield result[0], result[1], result[2]
                    else:
                        yield None, None, "Lip-sync generation failed."
                    return

                if scene_mode_val == MODE_AUTORUN:
                    yield from autorun_generate(
                        prompt, scene_mode_val, None,
                        dur, res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
                        neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                elif scene_mode_val == MODE_SEQUENCE:
                    result = generate_sequence(
                        seq_imgs, seq_prompts, seq_durs,
                        res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
                        neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                    yield result[0], result[1], ""
                elif scene_mode_val == MODE_CUSTOM_SEQ:
                    result = generate_custom_edit_sequence(
                        ref_image, cs_motions, cs_picgens, cs_durs,
                        res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
                        neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                    yield result[0], result[1], ""
                else:
                    yield from generate_video(
                        ref_image, prompt, scene_mode_val,
                        end_img, dur, res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, audio_neg_pt, ref_aud, dlg_txt, v_speed,
                        neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )

            generate_btn.click(
                fn=_dispatch_generate,
                inputs=[
                    scene_mode, reference_image, vid_prompt,
                    end_image, duration_seconds, resolution, frame_multiplier,
                    export_quality, seed, randomize_seed, add_audio_cb,
                    audio_prompt_tb, audio_negative_prompt_tb,
                    ref_audio_input, dialogue_text_tb, voice_speed_slider,
                    vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift,
                    lipsync_steps_sl, lipsync_cfg_sl,
                ] + sequence_images + sequence_prompts + sequence_durations
                  + custom_seq_motion_prompts + custom_seq_picgen_prompts + custom_seq_durations
                  + [lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes],
                outputs=[video_output, video_file, autorun_status],
            ).then(
                fn=_cache_last_frame_from_video,
                inputs=[video_file],
                outputs=[],
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js=_VID_DOWNLOAD_JS,
            ).then(
                fn=protect_current_inputs,
                inputs=[reference_image, end_image, merge_img_a, merge_img_b],
                outputs=[],
            ).then(
                fn=_delete_video_after_download,
                inputs=[video_file],
                outputs=[clear_storage_status],
            )

            generate_btn_bottom.click(
                fn=_dispatch_generate,
                inputs=[
                    scene_mode, reference_image, vid_prompt,
                    end_image, duration_seconds, resolution, frame_multiplier,
                    export_quality, seed, randomize_seed, add_audio_cb,
                    audio_prompt_tb, audio_negative_prompt_tb,
                    ref_audio_input, dialogue_text_tb, voice_speed_slider,
                    vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift,
                    lipsync_steps_sl, lipsync_cfg_sl,
                ] + sequence_images + sequence_prompts + sequence_durations
                  + custom_seq_motion_prompts + custom_seq_picgen_prompts + custom_seq_durations
                  + [lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes],
                outputs=[video_output, video_file, autorun_status],
            ).then(
                fn=_cache_last_frame_from_video,
                inputs=[video_file],
                outputs=[],
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js=_VID_DOWNLOAD_JS,
            ).then(
                fn=protect_current_inputs,
                inputs=[reference_image, end_image, merge_img_a, merge_img_b],
                outputs=[],
            ).then(
                fn=_delete_video_after_download,
                inputs=[video_file],
                outputs=[clear_storage_status],
            )

            push_autorun_btn.click(
                fn=autorun_push_generate,
                inputs=_PUSH_AUTORUN_INPUTS,
                outputs=[video_output, video_file, autorun_status],
            ).then(
                fn=_cache_last_frame_from_video,
                inputs=[video_file],
                outputs=[],
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js=_VID_DOWNLOAD_JS,
            ).then(
                fn=protect_current_inputs,
                inputs=[reference_image, end_image, merge_img_a, merge_img_b],
                outputs=[],
            ).then(
                fn=_delete_video_after_download,
                inputs=[video_file],
                outputs=[clear_storage_status],
            ).then(
                fn=None,
                inputs=[],
                outputs=[],
                js="""
                () => {
                    fetch('http://127.0.0.1:7861/autorun/ready', {
                        method: 'POST'
                    }).catch(() => {});
                    return [];
                }
                """
            )
            push_cancel_btn.click(
                fn=lambda: (_push_cancel.set() or _set_push_state("idle") or "Cancelled."),
                inputs=[],
                outputs=[autorun_status],
            )

            show_media_cb.change(
                fn=None,
                inputs=[show_media_cb],
                outputs=[],
                js="(show) => { document.body.classList.toggle('hide-media', !show); }",
            )

            clear_storage_btn_vid.click(
                fn=clear_storage,
                inputs=[],
                outputs=[clear_storage_status],
            )

            def get_frame_as_file(video_source, timestamp):
                """Extract frame at given seconds from the video.

                video_source may be a /media/ URL (generated video in
                _media_store) or a file path (user-uploaded video).
                Returns a /media/ URL for the extracted frame JPEG, which
                Gradio serves to both reference_image and download_file_output.
                """
                print(f"\nget_frame_as_file called, ts={timestamp}")
                if not video_source:
                    raise gr.Error("No video available. Generate a video first.")

                if hasattr(video_source, 'name'):
                    source = video_source.name
                elif isinstance(video_source, dict):
                    source = (video_source.get('url') or video_source.get('name')
                              or video_source.get('path') or str(video_source))
                else:
                    source = str(video_source)

                ts = float(timestamp) if timestamp else 0.0
                frame_url = extract_frame(source, ts)
                if not frame_url:
                    raise gr.Error("Failed to extract frame from video.")
                print(f" Frame extracted: {frame_url}")
                return frame_url

            def _use_frame_from_video_output_or_file(video_out, video_f, timestamp):
                source = None
                if video_out:
                    if isinstance(video_out, dict):
                        candidate = video_out.get('name') or video_out.get('path') or video_out.get('url') or ''
                        if candidate and os.path.exists(str(candidate)):
                            source = candidate
                    elif isinstance(video_out, str) and os.path.exists(video_out):
                        source = video_out
                if not source:
                    source = video_f if video_f else video_out
                return get_frame_as_file(source, timestamp)

            use_as_reference_btn.click(
                fn=_use_frame_from_video_output_or_file,
                inputs=[video_output, video_file, frame_time_input],
                outputs=[reference_image],
            )

            download_frame_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[download_file_output],
            )

            def _copy_end_to_first(end_img):
                """Return the end frame value so it replaces the first frame widget."""
                return end_img

            use_last_as_first_btn.click(
                fn=_copy_end_to_first,
                inputs=[end_image],
                outputs=[reference_image],
            )

            last_frame_from_video_btn.click(
                fn=_use_last_generated_frame,
                inputs=[],
                outputs=[reference_image],
            )


            def _extract_img_path(img):
                if img is None or img == "":
                    return None
                if isinstance(img, str):
                    return img
                for attr in ("name", "filename"):
                    v = getattr(img, attr, None)
                    if isinstance(v, str) and v:
                        return v
                return None

            def _make_image_tracker(is_primary=False):
                """Returns a change-handler closure with its own 'last path' state.

                When the user replaces an input image while a generation is still
                running, the old file must NOT be unprotected — the in-flight job
                is still reading it.  We check _is_generation_active() before
                calling _unprotect_path() so the file stays protected until the
                generation finishes and _generation_release() is called.
                """
                state = {"last": None}

                def _tracker(img):
                    global _current_input_image_path
                    new_path = _extract_img_path(img)
                    old_path = state["last"]
                    if new_path and not new_path.startswith("/media/"):
                        _protect_path(new_path)
                        _protect_filename(new_path)   # filename-layer too
                    if is_primary:
                        if not new_path or not new_path.startswith("/media/"):
                            _current_input_image_path = new_path
                    if old_path and old_path != new_path:
                        if not new_path or not old_path.startswith("/media/"):
                            if not _is_generation_active(old_path):
                                _unprotect_path(old_path)
                                _unprotect_filename(old_path)
                    state["last"] = new_path
                    return None

                return _tracker

            reference_image.change(
                fn=_make_image_tracker(is_primary=True),
                inputs=[reference_image],
                outputs=[],
            )
            end_image.change(
                fn=_make_image_tracker(),
                inputs=[end_image],
                outputs=[],
            )
            for _seq_img_comp in sequence_images:
                _seq_img_comp.change(
                    fn=_make_image_tracker(),
                    inputs=[_seq_img_comp],
                    outputs=[],
                )

            merge_img_a.change(
                fn=_make_image_tracker(),
                inputs=[merge_img_a],
                outputs=[],
            )
            merge_img_b.change(
                fn=_make_image_tracker(),
                inputs=[merge_img_b],
                outputs=[],
            )

        with gr.Tab("Photo Editor", id="picgen"):
            with gr.Column(elem_id="col-container"):

                gr.HTML("""
                <div id="starter-grid" style="display:grid;grid-template-columns:repeat(10,1fr);gap:6px;margin-bottom:8px;width:100%;">
                </div>
                <style>
                .starter-card{display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;
                  border:1px solid var(--border-color-primary);border-radius:6px;padding:4px 2px;
                  background:var(--background-fill-secondary);transition:border-color .15s;}
                .starter-card:hover{border-color:var(--color-accent);}
                .starter-thumb{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;
                  background:var(--background-fill-primary);display:block;}
                .starter-thumb-placeholder{width:100%;aspect-ratio:1;border-radius:4px;
                  background:var(--background-fill-primary);display:flex;align-items:center;
                  justify-content:center;font-size:11px;color:var(--body-text-color-subdued);}
                .starter-label{font-size:11px;font-weight:600;text-align:center;
                  color:var(--body-text-color);line-height:1;}
                </style>
                """)
                starter_b64_output = gr.Textbox(value="", visible=False, elem_id="starter-b64-output")

                with gr.Row():
                    with gr.Column(scale=1):
                        hidden_images_b64 = gr.Textbox(
                            value="[]", elem_id="hidden-images-b64",
                            elem_classes="hidden-input", container=False, visible=False,
                        )
                        gr.HTML("""
                        <div class="uploader-toolbar">
                            <button id="tb-upload" class="tb-btn">Upload</button>
                            <button id="tb-remove" class="tb-btn">Remove Selected</button>
                            <button id="tb-clear" class="tb-btn">Clear All</button>
                        </div>
                        <div class="quality-toolbar">
                            <span class="quality-label">Quality:</span>
                            <button class="tb-btn tb-quality" data-size="256">256 ⚡ Fastest</button>
                            <button class="tb-btn tb-quality tb-quality-active" data-size="512">512 ✦ Default</button>
                            <button class="tb-btn tb-quality" data-size="1024">1024 ▲ Better</button>
                            <button class="tb-btn tb-quality" data-size="1920">1920 ▲▲ High</button>
                            <button class="tb-btn tb-quality" data-size="2560">2560 ★ Max</button>
                        </div>
                        <div id="gallery-drop-zone">
                            <div id="upload-prompt" class="upload-prompt-modern">
                                <div id="upload-click-area" class="upload-click-area">
                                    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <rect x="6" y="10" width="52" height="44" rx="5" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>
                                        <polygon points="10,50 24,32 34,42 44,28 54,50" fill="rgba(128,128,128,0.15)" stroke="currentColor" stroke-width="1.5"/>
                                        <circle cx="22" cy="24" r="5" fill="rgba(128,128,128,0.2)" stroke="currentColor" stroke-width="1.5"/>
                                    </svg>
                                    <span class="upload-main-text">Click or drag images here</span>
                                    <span class="upload-sub-text">Supports multiple images</span>
                                </div>
                            </div>
                            <input id="custom-file-input" type="file" accept="image/*" multiple style="display:none;" />
                            <div id="image-gallery-grid" class="image-gallery-grid" style="display:none;"></div>
                        </div>
                        """)
                        # Three tools, side by side, together spanning the full
                        # width of the Generate button below.
                        with gr.Row(elem_id="picgen-tool-row"):
                            pic_complete_body_btn = gr.Button(
                                "Multi-Tool", variant="secondary", scale=1,
                                elem_id="pic-complete-body-btn",
                            )
                            pic_add_l2r_btn = gr.Button(
                                "Add Left To Right", variant="secondary", scale=1,
                            )
                            pic_add_r2l_btn = gr.Button(
                                "Add Right To Left", variant="secondary", scale=1,
                            )
                        pic_run_button_top = gr.Button("Generate", variant="primary", size="lg")
                        pic_num_images = gr.Slider(
                            label="Number of images", minimum=1, maximum=4, step=1, value=1,
                        )
                        pic_prompt = gr.Textbox(
                            label="Prompt", show_label=True,
                            placeholder="describe the edit instruction",
                            lines=3, max_lines=20,
                        )
                        pic_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="censored, mosaic, blurred, clothed, soft, partial",
                            value="", lines=2, max_lines=10,
                        )

                    with gr.Column(scale=1):
                        pic_result = gr.Gallery(
                            label="Result",
                            show_label=False,
                            type="filepath",
                            interactive=False,
                            columns=2,
                            elem_id="picgen-result-gallery",
                            # Serve full-quality PNG instead of Gradio's default
                            # webp. Without this, right-click > "open image in
                            # new tab" hands the browser a .webp it downloads
                            # rather than displaying inline full size.
                            format="png",
                        )
                        use_output_btn = gr.Button("Use as input", variant="secondary", size="sm")
                        pic_download_btn = gr.Button("Download All currently outputted Images", variant="secondary", size="sm")
                        picgen_urls = gr.Textbox(visible=False, value="", elem_id="picgen-urls")

                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Solo", choices=list(solo_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown2 = gr.Dropdown(
                                label="Couple (Unseen)", choices=list(couple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown3 = gr.Dropdown(
                                label="Couple (Seen)", choices=list(couple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown4 = gr.Dropdown(
                                label="Multi Women", choices=list(multiple_women_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        with gr.Row():
                            preset_dropdown5 = gr.Dropdown(
                                label="Multi (Unseen)", choices=list(multiple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown6 = gr.Dropdown(
                                label="Multi (Seen)", choices=list(multiple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown7 = gr.Dropdown(
                                label="Multi-Step", choices=list(multistep_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        pic_run_button = gr.Button("Generate", variant="primary", size="lg")

                with gr.Accordion("Advanced Settings", open=False):
                    pic_seed = gr.Slider(label="Seed", minimum=0, maximum=PICGEN_MAX_SEED, step=1, value=0)
                    pic_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        pic_guidance = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                        pic_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=40, step=1, value=4)
                        pic_height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=None)
                        pic_width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=None)


                preset_dropdown.change(fn=update_solo_prompt, inputs=[preset_dropdown], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown2.change(fn=update_couple_man_unseen_prompt, inputs=[preset_dropdown2], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown3.change(fn=update_couple_man_seen_prompt, inputs=[preset_dropdown3], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown4.change(fn=update_multiple_women_prompt, inputs=[preset_dropdown4], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown5.change(fn=update_multiple_man_unseen_prompt, inputs=[preset_dropdown5], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown6.change(fn=update_multiple_man_seen_prompt, inputs=[preset_dropdown6], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown7.change(fn=update_multistep_prompt, inputs=[preset_dropdown7], outputs=[pic_prompt], scroll_to_output=False)

                _pic_infer_inputs = [
                    hidden_images_b64, pic_prompt, pic_negative_prompt,
                    pic_seed, pic_randomize_seed, pic_guidance, pic_steps,
                    pic_height, pic_width, pic_num_images,
                ]
                # No js= on the click handlers — we rely on hidden_images_b64
                # being kept in sync by syncToGradio() (called by the gallery JS
                # on every image add/remove).  The old js= approach that mutated
                # args[0] broke in Gradio 4.x: the JS pre-hook must return
                # exactly as many elements as there are inputs, but Gradio 4.x
                # passes component values differently from 3.x, so any mismatch
                # results in "needed: 10, got: 0".  The hidden textbox is the
                # correct channel — Gradio reads it as a normal input.

                def _wire_picgen_btn(trigger):
                    trigger(
                        fn=infer_with_preclear,
                        inputs=_pic_infer_inputs,
                        outputs=[pic_result, pic_seed, picgen_urls],
                    ).then(
                        fn=lambda: __import__('time').sleep(2),
                        inputs=[],
                        outputs=[],
                    ).then(
                        fn=protect_current_inputs,
                        inputs=[reference_image, end_image, merge_img_a, merge_img_b],
                        outputs=None,
                    ).then(
                        fn=clear_storage,
                        inputs=[],
                        outputs=[clear_storage_status],
                    )

                _wire_picgen_btn(pic_run_button.click)
                _wire_picgen_btn(pic_run_button_top.click)
                _wire_picgen_btn(pic_prompt.submit)

                # Tool buttons (Completed Body / Add L->R / Add R->L). These are
                # just PRESET PROMPTS: each button drops its preset text into the
                # prompt box, then runs the normal Qwen generate flow (same as
                # clicking Generate). No special pipeline.
                PICGEN_TOOL_PRESETS = {
                    "complete_body": (
                        "outpaint to only fill in what is missing in the white part of the photo so their full body is fully visible and also add tiny background space above their head and beneath their feet. everything else is unchanged."
                    ),
                    "add_l2r": (
                        "Perfectly crop the person alone from image 1 to image 2 to add them naturally into only the scene of the second image without editing or regenerating it at all. "
                        "Fit them to be next to them in only 2nd's background. "
                        "Except you outpaint to only fill in what is missing of the parts of the body of the person from the first photo so their full body is fully visible with visibility underneath and everywhere around their body also. "
                        "They're same size as the other person. Everything else is the same. Only environment of the 2nd photo!!! "
                        "The man's penis remains unchanged."
                    ),
                    "add_r2l": (
                        "Perfectly crop the person alone from image 2 to image 1 to add them naturally into only the scene of the first image without editing or regenerating it at all. "
                        "Fit them to be next to them in only 1st's background. "
                        "Except you outpaint to only fill in what is missing of the parts of the body of the person from the second photo so their full body is fully visible with visibility underneath and everywhere around their body also. "
                        "They're same size as the other person. Everything else is the same. Only environment of the 1st photo!!! "
                        "The man's penis remains unchanged."
                    ),
                }

                def _set_prompt(text):
                    """Return the preset text to load into the prompt box."""
                    return text

                def _wire_picgen_tool_preset(button, preset_key):
                    # 1) set the preset prompt into the prompt box, then
                    # 2) run the exact same generate flow as the Generate button.
                    button.click(
                        fn=lambda k=preset_key: PICGEN_TOOL_PRESETS[k],
                        inputs=[],
                        outputs=[pic_prompt],
                    ).then(
                        fn=infer_with_preclear,
                        inputs=_pic_infer_inputs,
                        outputs=[pic_result, pic_seed, picgen_urls],
                    ).then(
                        fn=lambda: __import__('time').sleep(2),
                        inputs=[],
                        outputs=[],
                    ).then(
                        fn=protect_current_inputs,
                        inputs=[reference_image, end_image, merge_img_a, merge_img_b],
                        outputs=None,
                    ).then(
                        fn=clear_storage,
                        inputs=[],
                        outputs=[clear_storage_status],
                    )

                _wire_picgen_tool_preset(pic_add_l2r_btn, "add_l2r")
                _wire_picgen_tool_preset(pic_add_r2l_btn, "add_r2l")

                # Completed Body button opens the outpaint popup via JS only.
                # The popup handles everything client-side: padding canvas,
                # replacing the gallery image, setting the prompt, and clicking
                # Generate. No server round-trip until Generate is pressed.
                pic_complete_body_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="() => { if (window.__openOutpaintPopup) window.__openOutpaintPopup(); }",
                )


                def output_to_b64(output_images):
                    """Convert gallery items to base64 JPEG list for the input gallery.
                    Gallery type="filepath": items are file path strings or dicts with 'name'/'path'."""
                    if not output_images:
                        return "[]"
                    b64_list = []
                    for item in output_images:
                        try:
                            if isinstance(item, dict):
                                fpath = item.get("name") or item.get("path") or item.get("url") or ""
                            elif isinstance(item, (list, tuple)):
                                fpath = item[0] if item else ""
                            else:
                                fpath = str(item)
                            if not fpath or not os.path.exists(fpath):
                                continue
                            img = Image.open(fpath).convert("RGB")
                            max_size = 512
                            if img.width > img.height:
                                img = img.resize((max_size, int(img.height * max_size / img.width)), Image.LANCZOS)
                            else:
                                img = img.resize((int(img.width * max_size / img.height), max_size), Image.LANCZOS)
                            buf = BytesIO()
                            img.save(buf, format="JPEG", quality=95)
                            b64_list.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
                        except Exception as e:
                            print(f"output_to_b64 skipped an item: {e}")
                            continue
                    return json.dumps(b64_list)

                use_output_btn.click(fn=output_to_b64, inputs=[pic_result], outputs=[hidden_images_b64])

                pic_download_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="""
                    async () => {
                        // Download every image currently shown in the result
                        // gallery. We read the actual <img> src values from the
                        // DOM (works for Gradio /file= URLs, blob: and data:
                        // URLs, and encrypted /media/ URLs alike) rather than a
                        // separate URL list that could be out of sync.
                        const gal = document.getElementById('picgen-result-gallery');
                        if (!gal) { alert('No images to download — generate images first.'); return; }
                        const imgEls = Array.from(gal.querySelectorAll('img'));
                        // De-duplicate by src (galleries can render thumb + preview).
                        const seen = new Set();
                        const srcs = [];
                        imgEls.forEach(im => {
                            const s = im.currentSrc || im.src;
                            if (s && !seen.has(s)) { seen.add(s); srcs.push(s); }
                        });
                        if (srcs.length === 0) { alert('No images to download — generate images first.'); return; }

                        const _f = window.__ngOrigFetch || window.fetch;
                        for (let i = 0; i < srcs.length; i++) {
                            const src = srcs[i];
                            let filename = 'picgen_' + (i + 1) + '.png';
                            try {
                                const clean = src.split('?')[0].split('#')[0];
                                const last = clean.split('/').pop();
                                if (last && last.indexOf('.') !== -1) filename = decodeURIComponent(last);
                            } catch (e) {}
                            try {
                                const headers = {};
                                const isMedia = src.indexOf('/media/') !== -1;
                                if (isMedia && window.__ngSecretHex) headers['X-NG-Secret'] = window.__ngSecretHex;
                                const resp = await _f(src, { headers });
                                if (!resp.ok) { console.warn('[PicDL] fetch failed', src, resp.status); continue; }
                                let plainBytes;
                                const encrypted = resp.headers.get('X-NG-Encrypted');
                                if (encrypted === '1' && window.__ngKey) {
                                    const buf = await resp.arrayBuffer();
                                    try {
                                        plainBytes = await crypto.subtle.decrypt(
                                            { name: 'AES-GCM', iv: buf.slice(0, 12) },
                                            window.__ngKey, buf.slice(12)
                                        );
                                    } catch (e) { console.warn('[PicDL] decrypt:', e); continue; }
                                } else {
                                    plainBytes = await resp.arrayBuffer();
                                }
                                const ext = (filename.split('.').pop() || 'png').toLowerCase();
                                const mime = ext === 'png' ? 'image/png'
                                           : (ext === 'webp' ? 'image/webp' : 'image/jpeg');
                                const blob = new Blob([plainBytes], { type: mime });
                                const blobUrl = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = blobUrl;
                                a.download = filename;
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
                                // Small stagger so the browser accepts multiple downloads.
                                await new Promise(r => setTimeout(r, 350));
                            } catch (e) { console.warn('[PicDL] failed:', src, e); }
                        }
                    }
                    """,
                )
                
                hidden_images_b64.change(
                    fn=None, inputs=[hidden_images_b64], outputs=None,
                    js="""(b64List) => {
                        if (!b64List || !window.__addImage) return;
                        // Guard: syncToGradio() dispatches input+change on this
                        // same textbox, which would re-trigger this handler and
                        // loop forever. Skip if the change originated from us.
                        if (window.__picgenSuppressSync) return;
                        try {
                            const incoming = JSON.parse(b64List);
                            if (!Array.isArray(incoming) || incoming.length === 0) return;
                            // Skip if gallery already contains exactly these images
                            const cur = (window.__picgenImages || []).map(i => i.b64);
                            if (incoming.length === cur.length && incoming.every((b,i) => b === cur[i])) return;
                            // Add the incoming images alongside any that are already there.
                            // Do NOT clear first — this handler is triggered by "Use as input"
                            // which should append, not replace.
                            window.__picgenSuppressSync = true;
                            try {
                                incoming.forEach((b64, idx) => {
                                    window.__addImage(b64, `output_${idx + 1}.png`);
                                });
                            } finally {
                                window.__picgenSuppressSync = false;
                            }
                        } catch (e) {
                            window.__picgenSuppressSync = false;
                            console.error('Failed to parse output images:', e);
                        }
                    }""",
                )

                starter_b64_output.change(
                    fn=None, inputs=[starter_b64_output], outputs=None,
                    js="(b64) => { if (b64 && window.__addImage) window.__addImage(b64, 'starter.jpg'); }",
                )



    demo.load(fn=None, js=gallery_js)
    _picgen_dl_intercept_js = """
() => {
    async function fetchAndDownload(mediaUrl) {
        if (!mediaUrl || !mediaUrl.startsWith('/media/')) return false;
        const filename = mediaUrl.split('/').pop().split('?')[0] || 'picgen.png';

        // Wait up to 5s for encryption key
        let waited = 0;
        while (!window.__ngSecretHex && waited < 5000) {
            await new Promise(r => setTimeout(r, 100));
            waited += 100;
        }

        try {
            const headers = {};
            if (window.__ngSecretHex) headers['X-NG-Secret'] = window.__ngSecretHex;
            const _f = window.__ngOrigFetch || window.fetch;
            const resp = await _f(mediaUrl, { headers });
            if (!resp.ok) return false;
            const encrypted = resp.headers.get('X-NG-Encrypted');
            let plainBytes;
            if (encrypted === '1' && window.__ngKey) {
                const buf = await resp.arrayBuffer();
                plainBytes = await crypto.subtle.decrypt(
                    { name: 'AES-GCM', iv: buf.slice(0, 12) },
                    window.__ngKey,
                    buf.slice(12)
                );
            } else {
                plainBytes = await resp.arrayBuffer();
            }
            const ext = filename.split('.').pop().toLowerCase();
            const mime = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
            const blob = new Blob([plainBytes], { type: mime });
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
            return true;
        } catch(e) {
            console.warn('[PicGalleryDL] fetch/decrypt failed:', e);
            return false;
        }
    }

    function getMediaUrls() {
        // Read the hidden picgen_urls textbox value
        const el = document.getElementById('picgen-urls');
        if (!el) return [];
        const ta = el.querySelector('textarea') || el.querySelector('input');
        if (!ta) return [];
        try { return JSON.parse(ta.value || '[]'); } catch(e) { return []; }
    }

    function patchGalleryDownloadButtons(gallery) {
        // Find all download anchor/button elements in the gallery.
        // Gradio renders them as <a download> or buttons with a download SVG.
        // We intercept the click event and prevent the default broken /file= request.
        const items = gallery.querySelectorAll('.thumbnail-item, [class*="thumbnail"]');
        items.forEach((item, idx) => {
            // Find the download button inside this item
            const dlBtn = item.querySelector('a[download], button[aria-label*="ownload"], a[aria-label*="ownload"]');
            if (!dlBtn || dlBtn.dataset.ngPatched) return;
            dlBtn.dataset.ngPatched = '1';
            dlBtn.addEventListener('click', async (e) => {
                const urls = getMediaUrls();
                if (urls.length === 0) return; // no urls, let default happen
                e.preventDefault();
                e.stopPropagation();
                const mediaUrl = urls[idx];
                if (mediaUrl) {
                    await fetchAndDownload(mediaUrl);
                }
            }, true);
        });
    }

    function watchPicgenGallery() {
        const gallery = document.getElementById('picgen-result-gallery');
        if (!gallery) { setTimeout(watchPicgenGallery, 500); return; }
        // Watch for gallery content changes (new images rendered after generation)
        const obs = new MutationObserver(() => patchGalleryDownloadButtons(gallery));
        obs.observe(gallery, { childList: true, subtree: true });
        patchGalleryDownloadButtons(gallery);
    }
    setTimeout(watchPicgenGallery, 1000);
}
"""
    demo.load(fn=None, js=_picgen_dl_intercept_js)
    _starter_grid_js = """
() => {
    function buildStarterGrid() {
        const grid = document.getElementById('starter-grid');
        if (!grid) { setTimeout(buildStarterGrid, 300); return; }
        // Clear and rebuild every time so tab re-renders don't leave a stale grid
        grid.innerHTML = '';

        for (let n = 1; n <= 10; n++) {
            const card = document.createElement('div');
            card.className = 'starter-card';
            card.dataset.num = n;

            // Thumbnail — use the raw original fetch to bypass the /media/ interceptor
            const img = document.createElement('img');
            img.className = 'starter-thumb';
            img.alt = String(n);
            img.loading = 'eager';

            const _rawFetch = window.__ngOrigFetch || window.fetch;
            _rawFetch('/starters/' + n)
                .then(r => {
                    if (!r.ok) throw new Error('not found');
                    return r.blob();
                })
                .then(blob => {
                    img.src = URL.createObjectURL(blob);
                })
                .catch(() => {
                    const ph = document.createElement('div');
                    ph.className = 'starter-thumb-placeholder';
                    ph.textContent = n;
                    if (img.parentNode) card.replaceChild(ph, img);
                    else card.insertBefore(ph, card.firstChild);
                });

            const label = document.createElement('div');
            label.className = 'starter-label';
            label.textContent = String(n);

            card.appendChild(img);
            card.appendChild(label);

            card.addEventListener('click', function() {
                const num = this.dataset.num;
                const _fetch = window.__ngOrigFetch || window.fetch;
                _fetch('/starters/' + num)
                    .then(r => {
                        if (!r.ok) throw new Error('not found');
                        const ct = r.headers.get('Content-Type') || 'image/jpeg';
                        return r.blob().then(blob => ({ blob, ct }));
                    })
                    .then(({ blob, ct }) => {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            const b64 = e.target.result;
                            if (window.__addImage) {
                                window.__addImage(b64, 'starter' + num + '.jpg');
                            }
                        };
                        reader.readAsDataURL(blob);
                    })
                    .catch(e => console.warn('[Starters] could not load starter', num, e));
            });

            grid.appendChild(card);
        }
    }

    // Build immediately and also re-build when the starter-grid appears after
    // a tab switch (Gradio renders tabs lazily so the element may not exist yet)
    buildStarterGrid();
    const _sgObs = new MutationObserver(() => {
        const g = document.getElementById('starter-grid');
        if (g && g.children.length === 0) buildStarterGrid();
    });
    _sgObs.observe(document.body, { childList: true, subtree: true });
}
"""
    demo.load(fn=None, js=_starter_grid_js)


    video_time_sync_js = """
() => {
    function startVideoSync() {
        const video = document.querySelector('#generated-video video');
        if (!video) { setTimeout(startVideoSync, 500); return; }
        // Update the number input every 200ms while video is playing
        setInterval(() => {
            const input = document.querySelector('#frame-time-input input');
            // Only update if input exists, video time is valid, and user is NOT actively editing the field
            if (input && !isNaN(video.currentTime) && document.activeElement !== input) {
                const nativeInput = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (nativeInput && nativeInput.set) {
                    nativeInput.set.call(input, video.currentTime.toFixed(2));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }, 200);
        // Re-run if a new video loads (MutationObserver on the video src)
        const observer = new MutationObserver(() => {
            const newVideo = document.querySelector('#generated-video video');
            if (newVideo && newVideo !== video) { observer.disconnect(); startVideoSync(); }
        });
        observer.observe(document.querySelector('#generated-video') || document.body, { childList: true, subtree: true });
    }
    setTimeout(startVideoSync, 1000);
}
"""
    demo.load(fn=None, js=video_time_sync_js)


    _encryption_init_js = """
() => {
    // ---- Key derivation (runs once per page load) -------------------------
    async function initEncryptionKey() {
        if (window.__ngKey) return;  // already initialised

        // Retrieve or generate the 32-byte device secret stored in localStorage.
        // This secret never leaves the browser — the server never sees the key,
        // only the secret from which it derives the matching key server-side.
        let secretHex = localStorage.getItem('__ngSecret__');
        if (!secretHex || secretHex.length !== 64) {
            const raw = new Uint8Array(32);
            crypto.getRandomValues(raw);
            secretHex = Array.from(raw).map(b => b.toString(16).padStart(2,'0')).join('');
            localStorage.setItem('__ngSecret__', secretHex);
        }
        window.__ngSecretHex = secretHex;

        // Import the raw secret as HKDF base key material
        const secretBytes = new Uint8Array(secretHex.match(/../g).map(h => parseInt(h,16)));
        const baseKey = await crypto.subtle.importKey(
            'raw', secretBytes, { name: 'HKDF' }, false, ['deriveKey']
        );

        // Derive AES-256-GCM media key (non-extractable, memory only)
        const enc = new TextEncoder();
        window.__ngKey = await crypto.subtle.deriveKey(
            { name: 'HKDF', hash: 'SHA-256', salt: enc.encode('newgen-media-v1'), info: enc.encode('aes-256-gcm-media') },
            baseKey,
            { name: 'AES-GCM', length: 256 },
            false,
            ['decrypt']
        );

        // Derive AES-256-GCM log key (non-extractable, memory only)
        window.__ngLogKey = await crypto.subtle.deriveKey(
            { name: 'HKDF', hash: 'SHA-256', salt: enc.encode('newgen-logs-v1'), info: enc.encode('aes-256-gcm-logs') },
            baseKey,
            { name: 'AES-GCM', length: 256 },
            false,
            ['decrypt']
        );
    }

    // ---- /media/ fetch interceptor ----------------------------------------
    // Wraps the global fetch so every request to /media/ automatically:
    //   1. Adds the X-NG-Secret header
    //   2. Decrypts the AES-256-GCM response before returning it
    // This is transparent to all other code (video player, gallery, downloads).
    const _origFetch = window.fetch;
    window.__ngOrigFetch = _origFetch;  // saved so video player can bypass interceptor
    window.fetch = async function(input, init) {
        const url = (typeof input === 'string') ? input : (input instanceof Request ? input.url : String(input));
        const isMedia = url.includes('/media/');

        if (!isMedia || !window.__ngSecretHex) {
            return _origFetch(input, init);
        }

        // Inject the secret header
        const headers = new Headers((init && init.headers) ? init.headers : {});
        headers.set('X-NG-Secret', window.__ngSecretHex);
        const newInit = Object.assign({}, init || {}, { headers });

        const response = await _origFetch(input, newInit);
        if (!response.ok) return response;

        const encrypted = response.headers.get('X-NG-Encrypted');
        if (encrypted !== '1' || !window.__ngKey) return response;

        // Decrypt: response body = 12-byte nonce + ciphertext
        const buf = await response.arrayBuffer();
        const nonce = buf.slice(0, 12);
        const ciphertext = buf.slice(12);
        let plaintext;
        try {
            plaintext = await crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: nonce },
                window.__ngKey,
                ciphertext
            );
        } catch(e) {
            console.warn('[NG-Enc] media decrypt failed:', e);
            return new Response(new Uint8Array(0), { status: 200 });
        }

        const origType = response.headers.get('X-NG-Original-Type') || 'application/octet-stream';
        const origName = response.headers.get('X-NG-Filename') || '';
        const respHeaders = new Headers();
        respHeaders.set('Content-Type', origType);
        respHeaders.set('Content-Length', String(plaintext.byteLength));
        if (origName) respHeaders.set('Content-Disposition', 'inline; filename="' + origName + '"');
        return new Response(plaintext, { status: 200, headers: respHeaders });
    };

    // ---- SSE log panel -------------------------------------------------------
    function connectLogStream() {
        if (!window.__ngLogKey || !window.__ngSecretHex) {
            setTimeout(connectLogStream, 1000);
            return;
        }
        const es = new EventSource('/logs/stream?_=' + Date.now());

        // We can't add custom headers to EventSource — use a modified URL approach:
        // close EventSource and switch to fetch-based SSE with the secret header.
        es.close();

        async function fetchSSE() {
            const dec = new TextDecoder();
            try {
                const resp = await _origFetch('/logs/stream', {
                    headers: { 'X-NG-Secret': window.__ngSecretHex },
                });
                if (!resp.body) return;
                const reader = resp.body.getReader();
                let buf = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    const lines = buf.split('\\n');
                    buf = lines.pop();
                    for (const raw of lines) {
                        const line = raw.trim();
                        if (!line || line.startsWith(':')) continue;  // keepalive
                        const b64 = line.replace(/^data:\\s*/, '');
                        if (!b64) continue;
                        try {
                            const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                            const nonce = bytes.slice(0, 12);
                            const ct    = bytes.slice(12);
                            const ptBuf = await crypto.subtle.decrypt(
                                { name: 'AES-GCM', iv: nonce },
                                window.__ngLogKey,
                                ct
                            );
                            const text = new TextDecoder().decode(ptBuf);
                            const panel = document.getElementById('ng-log-panel');
                            if (panel) {
                                const line_el = document.createElement('div');
                                line_el.textContent = text;
                                panel.appendChild(line_el);
                                // Keep last 500 lines
                                while (panel.children.length > 500) panel.removeChild(panel.firstChild);
                                panel.scrollTop = panel.scrollHeight;
                            }
                        } catch(e) { /* decrypt error on keepalive or bad frame */ }
                    }
                }
            } catch(e) {
                // Reconnect after 2s on any stream error
                setTimeout(fetchSSE, 2000);
            }
        }
        fetchSSE();
    }

    // Initialise key then start log stream
    initEncryptionKey().then(() => {
        connectLogStream();
    });
}
"""
    demo.load(fn=None, js=_encryption_init_js)

    # ── Multi-Tool popup ──────────────────────────────────────────────────────
    # Opened when the user clicks "Multi-Tool". Shows the current first gallery
    # image (or an upload zone if none). Edge drag-handles let the user extend
    # (outward = white space added) or crop (inward = image trimmed) on any
    # side. Buttons: Upload (import image), Update Input (push to gallery, no
    # generation), Reset, Generate (sets prompt + triggers normal Generate).
    _outpaint_popup_js = r"""
() => {
window.__openOutpaintPopup = function() {
    // ── constants ────────────────────────────────────────────────────────────
    const PROMPT = 'outpaint to only fill in what is missing in the white part of the photo so their full body is fully visible and also add tiny background space above their head and beneath their feet. everything else is unchanged.';
    const HANDLE_SIZE = 28;   // px — draggable edge strip width
    const MIN_PAD = -4000;    // negative = crop inward
    const MAX_PAD = 1200;     // positive = add white space outward

    // ── state ────────────────────────────────────────────────────────────────
    let srcB64 = null;        // original image base64
    let padTop = 0, padRight = 0, padBottom = 0, padLeft = 0;
    let dragging = null;
    // Index of the gallery image currently shown in the popup (cycles on swap)
    let _galleryIdx = 0;
    // Track whether the loaded image came from the popup's own upload
    // (true) or was pulled from the gallery (false). Determines whether
    // Update/Generate adds to the gallery or replaces.
    let _fromPopupUpload = false;

    // ── overlay ──────────────────────────────────────────────────────────────
    let overlay = document.getElementById('ng-outpaint-overlay');
    if (overlay) {
        const imgs = window.__picgenImages || [];
        // Restore last-viewed gallery index; clamp in case images were removed
        _galleryIdx = Math.min(overlay._galleryIdx || 0, Math.max(0, imgs.length - 1));
        const fresh = imgs.length > 0 ? imgs[_galleryIdx].b64 : null;
        if (fresh && overlay._loadSrc) {
            _fromPopupUpload = false;
            overlay._loadSrc(fresh);
        } else if (!fresh && overlay._render) {
            overlay._render();
        }
        // Show/hide swap button depending on current image count
        if (overlay._swapBtn) {
            overlay._swapBtn.style.display = imgs.length > 1 ? '' : 'none';
            overlay._swapBtn.textContent = '\u21c4 Swap (' + (imgs.length > 0 ? (_galleryIdx + 1) + '/' + imgs.length : '0') + ')';
        }
        overlay.style.display = 'flex';
        return;
    }

    overlay = document.createElement('div');
    overlay.id = 'ng-outpaint-overlay';
    overlay.style.cssText = [
        'position:fixed;top:0;left:0;width:100%;height:100%;',
        'background:rgba(0,0,0,0.85);z-index:20000;',
        'display:flex;flex-direction:column;align-items:center;justify-content:center;',
        'gap:12px;'
    ].join('');
    document.body.appendChild(overlay);

    // Title + close
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:16px;width:min(92vw,860px);';
    header.innerHTML = '<span style="color:#fff;font-size:15px;font-weight:600;flex:1;">'
        + 'Multi-Tool \u2014 drag edges outward to add space, inward to crop</span>'
        + '<button id="ng-op-close" style="width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.15);'
        + 'color:#fff;border:1px solid rgba(255,255,255,0.3);cursor:pointer;font-size:18px;'
        + 'display:flex;align-items:center;justify-content:center;">\u00d7</button>';
    overlay.appendChild(header);

    // Upload button row — always visible so user can import at any time
    const uploadRow = document.createElement('div');
    uploadRow.style.cssText = 'display:flex;align-items:center;gap:10px;width:min(92vw,860px);flex-wrap:wrap;';
    uploadRow.innerHTML = '<button id="ng-op-upload-btn" style="padding:6px 16px;border-radius:6px;'
        + 'background:rgba(255,255,255,0.12);color:#fff;border:1px solid rgba(255,255,255,0.35);'
        + 'cursor:pointer;font-size:12px;font-weight:500;">&#128193; Import image</button>'
        + '<button id="ng-op-swap" style="padding:6px 14px;border-radius:6px;display:none;'
        + 'background:rgba(255,200,80,0.18);color:#ffd966;border:1px solid rgba(255,200,80,0.5);'
        + 'cursor:pointer;font-size:12px;font-weight:600;">\u21c4 Swap (1/1)</button>'
        + '<span id="ng-op-upload-hint" style="font-size:11px;color:rgba(255,255,255,0.45);">'
        + 'Imports into the popup only \u2014 use Update Input or Generate to push to the gallery</span>'
        + '<input id="ng-op-file" type="file" accept="image/*" style="display:none;">';
    overlay.appendChild(uploadRow);

    // Upload drop zone — only shown when no image is loaded yet
    const uploadZone = document.createElement('div');
    uploadZone.id = 'ng-op-dropzone';
    uploadZone.style.cssText = 'width:min(92vw,860px);height:280px;border:2px dashed rgba(255,255,255,0.3);'
        + 'border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;'
        + 'gap:8px;cursor:pointer;color:rgba(255,255,255,0.6);font-size:13px;';
    uploadZone.innerHTML = '<svg width="48" height="48" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2">'
        + '<rect x="6" y="10" width="52" height="44" rx="5" stroke-dasharray="4 3"/>'
        + '<polygon points="10,50 24,32 34,42 44,28 54,50" fill="rgba(255,255,255,0.08)"/>'
        + '<circle cx="22" cy="24" r="5" fill="rgba(255,255,255,0.08)"/></svg>'
        + '<span>Click here or drag an image to load it</span>';
    overlay.appendChild(uploadZone);

    // Canvas wrapper — shown once an image is loaded
    const canvasWrap = document.createElement('div');
    canvasWrap.id = 'ng-op-wrap';
    canvasWrap.style.cssText = 'position:relative;display:none;touch-action:none;user-select:none;';
    overlay.appendChild(canvasWrap);

    const cvs = document.createElement('canvas');
    cvs.id = 'ng-op-canvas';
    cvs.style.cssText = 'display:block;border:1px solid rgba(255,255,255,0.2);border-radius:4px;';
    canvasWrap.appendChild(cvs);

    // Pad labels per edge
    const labels = {};
    ['top','right','bottom','left'].forEach(edge => {
        const lbl = document.createElement('div');
        lbl.id = 'ng-op-lbl-' + edge;
        lbl.style.cssText = 'position:absolute;background:rgba(0,0,0,0.7);color:#fff;'
            + 'font-size:11px;padding:2px 6px;border-radius:3px;pointer-events:none;white-space:nowrap;';
        canvasWrap.appendChild(lbl);
        labels[edge] = lbl;
    });

    // Drag handles — 4 edges
    const handles = {};
    ['top','right','bottom','left'].forEach(edge => {
        const h = document.createElement('div');
        h.dataset.edge = edge;
        h.style.cssText = 'position:absolute;cursor:'
            + (edge === 'top' || edge === 'bottom' ? 'ns-resize' : 'ew-resize')
            + ';z-index:10;';
        canvasWrap.appendChild(h);
        handles[edge] = h;
    });

    // Button row: Reset | Update Input | Generate
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:10px;flex-wrap:wrap;justify-content:center;';
    btnRow.innerHTML = ''
        + '<button id="ng-op-reset" style="padding:7px 18px;border-radius:6px;'
        + 'background:rgba(255,255,255,0.1);color:#fff;border:1px solid rgba(255,255,255,0.3);'
        + 'cursor:pointer;font-size:13px;">Reset</button>'
        + '<button id="ng-op-update" style="padding:7px 20px;border-radius:6px;'
        + 'background:rgba(80,160,255,0.25);color:#fff;border:1px solid rgba(80,160,255,0.6);'
        + 'cursor:pointer;font-size:13px;font-weight:600;">Update Input</button>'
        + '<button id="ng-op-generate" style="padding:7px 24px;border-radius:6px;'
        + 'background:#7c5cbf;color:#fff;border:none;cursor:pointer;font-size:14px;font-weight:600;">'
        + 'Generate (4 images)</button>';
    overlay.appendChild(btnRow);

    // ── helpers ───────────────────────────────────────────────────────────────
    function _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function _positionHandles(dw, dh) {
        const strip = HANDLE_SIZE;
        handles.top.style.cssText    += `;left:0;top:0;width:${dw}px;height:${strip}px;`;
        handles.bottom.style.cssText += `;left:0;bottom:0;width:${dw}px;height:${strip}px;`;
        handles.left.style.cssText   += `;top:0;left:0;width:${strip}px;height:${dh}px;`;
        handles.right.style.cssText  += `;top:0;right:0;width:${strip}px;height:${dh}px;`;
        labels.top.style.cssText     += `;left:50%;top:${strip+2}px;transform:translateX(-50%);`;
        labels.bottom.style.cssText  += `;left:50%;bottom:${strip+2}px;transform:translateX(-50%);`;
        labels.left.style.cssText    += `;top:50%;left:${strip+2}px;transform:translateY(-50%);`;
        labels.right.style.cssText   += `;top:50%;right:${strip+2}px;transform:translateY(-50%);`;
    }

    function _labelText(px) {
        if (px === 0) return '';
        return (px > 0 ? '+' : '') + px + 'px';
    }

    function _updateLabels() {
        labels.top.textContent    = _labelText(padTop);
        labels.right.textContent  = _labelText(padRight);
        labels.bottom.textContent = _labelText(padBottom);
        labels.left.textContent   = _labelText(padLeft);
    }

    function _render() {
        if (!srcB64) return;
        const img = new Image();
        img.onload = () => {
            const origW = img.naturalWidth;
            const origH = img.naturalHeight;

            // Effective canvas includes padding (positive) and excludes crop (negative)
            const totalW = Math.max(1, origW + padLeft + padRight);
            const totalH = Math.max(1, origH + padTop  + padBottom);

            const maxCssW = Math.floor(window.innerWidth  * 0.82);
            const maxCssH = Math.floor(window.innerHeight * 0.62);
            const scale   = Math.min(maxCssW / totalW, maxCssH / totalH, 1);
            const cssW    = Math.round(totalW * scale);
            const cssH    = Math.round(totalH * scale);

            cvs.width  = totalW;
            cvs.height = totalH;
            cvs.style.width  = cssW + 'px';
            cvs.style.height = cssH + 'px';

            const ctx = cvs.getContext('2d');
            // White background for padded areas
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, totalW, totalH);
            // Draw image offset by left/top padding (negative = image shifted left/up = crop)
            ctx.drawImage(img, padLeft, padTop, origW, origH);

            // Guide lines for padding (blue) and crop borders (red)
            // Top
            if (padTop !== 0) {
                ctx.strokeStyle = padTop > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padTop < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(0, padTop); ctx.lineTo(totalW, padTop); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Bottom
            if (padBottom !== 0) {
                ctx.strokeStyle = padBottom > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padBottom < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(0, origH+padTop); ctx.lineTo(totalW, origH+padTop); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Left
            if (padLeft !== 0) {
                ctx.strokeStyle = padLeft > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padLeft < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(padLeft, 0); ctx.lineTo(padLeft, totalH); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Right
            if (padRight !== 0) {
                ctx.strokeStyle = padRight > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padRight < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(origW+padLeft, 0); ctx.lineTo(origW+padLeft, totalH); ctx.stroke();
                ctx.setLineDash([]);
            }

            canvasWrap.style.width  = cssW + 'px';
            canvasWrap.style.height = cssH + 'px';
            uploadZone.style.display  = 'none';
            canvasWrap.style.display  = 'block';

            _positionHandles(cssW, cssH);
            _updateLabels();
        };
        img.src = srcB64;
    }

    function _loadSrc(b64) {
        srcB64 = b64;
        padTop = padRight = padBottom = padLeft = 0;
        // Persist the current gallery index on the overlay element so the
        // re-open branch can restore it next time the popup is opened.
        if (overlay) overlay._galleryIdx = _galleryIdx;
        _render();
    }
    overlay._loadSrc = _loadSrc;
    overlay._render  = _render;

    // ── swap button wiring ────────────────────────────────────────────────────
    const swapBtn = document.getElementById('ng-op-swap');
    overlay._swapBtn = swapBtn;
    // Show button only when more than one gallery image exists
    const _imgs0 = window.__picgenImages || [];
    if (_imgs0.length > 1) {
        swapBtn.style.display = '';
        swapBtn.textContent = '\u21c4 Swap (1/' + _imgs0.length + ')';
    }
    swapBtn.addEventListener('click', () => {
        const imgs = window.__picgenImages || [];
        if (imgs.length < 2) return;
        _galleryIdx = (_galleryIdx + 1) % imgs.length;
        overlay._galleryIdx = _galleryIdx;
        _fromPopupUpload = false;
        _loadSrc(imgs[_galleryIdx].b64);
        swapBtn.textContent = '\u21c4 Swap (' + (_galleryIdx + 1) + '/' + imgs.length + ')';
    });

    // ── drag logic ────────────────────────────────────────────────────────────
    function _onPointerDown(e) {
        const edge = e.currentTarget.dataset.edge;
        if (!edge || !srcB64) return;
        e.preventDefault();
        dragging = { edge, startX: e.clientX, startY: e.clientY,
                     startPad: { top:padTop, right:padRight, bottom:padBottom, left:padLeft } };
        window.addEventListener('pointermove', _onPointerMove);
        window.addEventListener('pointerup',   _onPointerUp);
    }

    function _onPointerMove(e) {
        if (!dragging) return;
        const dx = e.clientX - dragging.startX;
        const dy = e.clientY - dragging.startY;
        const img = new Image();
        img.src = srcB64;
        const origW = cvs.width  - dragging.startPad.left - dragging.startPad.right;
        const origH = cvs.height - dragging.startPad.top  - dragging.startPad.bottom;
        const totalW = origW + dragging.startPad.left + dragging.startPad.right;
        const totalH = origH + dragging.startPad.top  + dragging.startPad.bottom;
        const maxCssW = Math.floor(window.innerWidth  * 0.82);
        const maxCssH = Math.floor(window.innerHeight * 0.62);
        const scale   = Math.min(maxCssW / totalW, maxCssH / totalH, 1);
        switch (dragging.edge) {
            case 'top':    padTop    = _clamp(dragging.startPad.top    - Math.round(dy / scale), MIN_PAD, MAX_PAD); break;
            case 'bottom': padBottom = _clamp(dragging.startPad.bottom + Math.round(dy / scale), MIN_PAD, MAX_PAD); break;
            case 'left':   padLeft   = _clamp(dragging.startPad.left   - Math.round(dx / scale), MIN_PAD, MAX_PAD); break;
            case 'right':  padRight  = _clamp(dragging.startPad.right  + Math.round(dx / scale), MIN_PAD, MAX_PAD); break;
        }
        _render();
    }

    function _onPointerUp() {
        if (dragging) _recentlyDragged = true;
        dragging = null;
        window.removeEventListener('pointermove', _onPointerMove);
        window.removeEventListener('pointerup',   _onPointerUp);
        setTimeout(() => { _recentlyDragged = false; }, 400);
    }

    Object.values(handles).forEach(h => h.addEventListener('pointerdown', _onPointerDown));

    // ── file import ───────────────────────────────────────────────────────────
    function _handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            _fromPopupUpload = true;
            _loadSrc(ev.target.result);
        };
        reader.readAsDataURL(file);
    }

    const opFile = document.getElementById('ng-op-file');
    document.getElementById('ng-op-upload-btn').addEventListener('click', () => opFile.click());
    opFile.addEventListener('change', (e) => { _handleFile(e.target.files[0]); e.target.value = ''; });
    uploadZone.addEventListener('click', () => opFile.click());
    uploadZone.addEventListener('dragover',  (e) => { e.preventDefault(); uploadZone.style.borderColor='#7c5cbf'; });
    uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor='rgba(255,255,255,0.3)'; });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor='rgba(255,255,255,0.3)';
        _handleFile(e.dataTransfer.files[0]);
    });

    // ── export canvas to b64 ─────────────────────────────────────────────────
    function _exportCanvas() {
        return cvs.toDataURL('image/jpeg', 0.95);
    }

    // ── push to gallery (shared by Update Input and Generate) ─────────────────
    // If the image came from the popup's own upload, ADD it alongside existing images.
    // If it came from the gallery, REPLACE only that specific slot (_galleryIdx)
    // so all other images are preserved.
    function _pushToGallery(exportB64, name) {
        window.__picgenSuppressSync = true;
        try {
            if (_fromPopupUpload) {
                // Fresh import — add without disturbing existing images
                if (window.__addImage) window.__addImage(exportB64, name);
            } else {
                // Editing an existing gallery slot — swap only that index
                const imgs = window.__picgenImages;
                if (imgs && imgs.length > 0) {
                    const idx = Math.min(_galleryIdx, imgs.length - 1);
                    // Take a snapshot, replace the target slot, then rebuild gallery
                    const snapshot = imgs.map((img, i) =>
                        i === idx ? { id: Date.now() + Math.random(), b64: exportB64, name: name } : img
                    );
                    // Rebuild: clear then re-add in order (triggers proper render + sync)
                    imgs.length = 0;
                    if (window.__clearAll) window.__clearAll();
                    snapshot.forEach(img => { if (window.__addImage) window.__addImage(img.b64, img.name); });
                } else {
                    // Gallery is empty — just add
                    if (window.__addImage) window.__addImage(exportB64, name);
                }
            }
        } finally {
            window.__picgenSuppressSync = false;
            const _imgs = window.__picgenImages || [];
            const _container = document.getElementById('hidden-images-b64');
            if (_container) {
                _container.querySelectorAll('input,textarea').forEach(el => {
                    const _proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                    const _ns = Object.getOwnPropertyDescriptor(_proto, 'value');
                    if (_ns && _ns.set) {
                        _ns.set.call(el, JSON.stringify(_imgs.map(i => i.b64)));
                        el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    }
                });
            }
        }
    }

    // ── close ─────────────────────────────────────────────────────────────────
    let _recentlyDragged = false;
    function _close() { overlay.style.display = 'none'; _recentlyDragged = false; }
    document.getElementById('ng-op-close').addEventListener('click', _close);
    overlay.addEventListener('click', (e) => {
        if (_recentlyDragged) { e.stopImmediatePropagation(); _recentlyDragged = false; }
    });

    // ── reset ─────────────────────────────────────────────────────────────────
    document.getElementById('ng-op-reset').addEventListener('click', () => {
        padTop = padRight = padBottom = padLeft = 0;
        _render();
    });

    // ── update input (no generation) ─────────────────────────────────────────
    document.getElementById('ng-op-update').addEventListener('click', () => {
        if (!srcB64) { alert('Load an image first.'); return; }
        _pushToGallery(_exportCanvas(), 'edited_input.jpg');
        _close();
    });

    // ── generate ─────────────────────────────────────────────────────────────
    document.getElementById('ng-op-generate').addEventListener('click', () => {
        if (!srcB64) { alert('Load an image first.'); return; }
        _pushToGallery(_exportCanvas(), 'outpaint_input.jpg');

        // Write prompt into the visible Prompt textarea
        const picgenCol = document.getElementById('col-container');
        let promptTA = null;
        if (picgenCol) {
            for (const ta of Array.from(picgenCol.querySelectorAll('textarea'))) {
                const block = ta.closest('.block') || ta.parentElement;
                const lbl = block ? block.querySelector('label span, .label-wrap span') : null;
                if (lbl && lbl.textContent.trim().toLowerCase() === 'prompt') { promptTA = ta; break; }
            }
            if (!promptTA) {
                const all = Array.from(picgenCol.querySelectorAll('textarea'));
                promptTA = all.find(ta => {
                    const b = ta.closest('.block') || ta.parentElement;
                    const l = b ? (b.querySelector('label span, .label-wrap span') || {}).textContent || '' : '';
                    return !l.toLowerCase().includes('negative');
                }) || all[0];
            }
        }
        if (promptTA) {
            const ns = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            if (ns && ns.set) {
                ns.set.call(promptTA, PROMPT);
                promptTA.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                promptTA.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        }

        // Set Number of images slider to 4
        if (picgenCol) {
            picgenCol.querySelectorAll('input[type="range"]').forEach(sl => {
                const b = sl.closest('.block');
                const l = b ? b.querySelector('label span') : null;
                if (l && l.textContent.includes('Number of images')) {
                    const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                    if (ns && ns.set) {
                        ns.set.call(sl, '4');
                        sl.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                        sl.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                    }
                }
            });
        }

        _close();
        setTimeout(() => {
            const genBtn = Array.from(document.querySelectorAll('button')).find(b =>
                b.textContent.trim() === 'Generate' &&
                b.id !== 'ng-op-generate' &&
                b.closest('#col-container')
            );
            if (genBtn) genBtn.click();
        }, 150);
    });

    // ── populate from gallery on first open ───────────────────────────────────
    const existing = window.__picgenImages;
    if (existing && existing.length > 0) {
        _galleryIdx = 0;
        overlay._galleryIdx = 0;
        _fromPopupUpload = false;
        _loadSrc(existing[0].b64);
    }
};
}
"""
    demo.load(fn=None, js=_outpaint_popup_js)

from fastapi.responses import Response as _FastAPIResponse
from fastapi import Request as _FastAPIRequest

@demo.app.get("/api.md")
async def _export_api_md():
    """Download a Markdown API reference with Node.js examples."""
    import json as _json
    import urllib.request as _urllib

    try:
        with _urllib.urlopen("http://127.0.0.1:7860/info", timeout=10) as resp:
            info = _json.loads(resp.read())
    except Exception as e:
        return _FastAPIResponse(
            content=("# API Export Error\n\nCould not fetch /info: " + str(e) + "\n").encode(),
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="api.md"'},
        )

    named = info.get("named_endpoints", {})
    unnamed = info.get("unnamed_endpoints", {})

    def _type_str(t):
        if isinstance(t, dict):
            return t.get("type", t.get("description", str(t)))
        return str(t)

    def _build_params_table(params):
        if not params:
            return "_No parameters._\n"
        rows = ["| Name | Type | Default | Description |",
                "|------|------|---------|-------------|"]
        for p in params:
            name = p.get("label") or p.get("parameter_name") or p.get("name") or "?"
            typ  = _type_str(p.get("type", p.get("python_type", {})))
            default = str(p.get("default", "_required_"))
            desc = (p.get("description") or "").replace("|", "\\|").replace("\n", " ")
            rows.append("| `" + name + "` | `" + typ + "` | `" + default + "` | " + desc + " |")
        return "\n".join(rows) + "\n"

    def _build_returns_table(returns):
        if not returns:
            return "_No return values._\n"
        rows = ["| Name | Type | Description |",
                "|------|------|-------------|"]
        for r in returns:
            name = r.get("label") or r.get("name") or "?"
            typ  = _type_str(r.get("type", r.get("python_type", {})))
            desc = (r.get("description") or "").replace("|", "\\|").replace("\n", " ")
            rows.append("| `" + name + "` | `" + typ + "` | " + desc + " |")
        return "\n".join(rows) + "\n"

    def _js_example(api_name, params):
        param_names = [(p.get("parameter_name") or p.get("label") or p.get("name") or "param") for p in (params or [])]
        args_obj = "{ " + ", ".join(n + ": <value>" for n in param_names) + " }" if param_names else "{}"
        return (
            "```javascript\n"
            "// Node.js — requires: npm install @gradio/client\n"
            "import { Client } from \"@gradio/client\";\n\n"
            "const client = await Client.connect(\"http://0.0.0.0:7860\");\n"
            "const result = await client.predict(\"" + api_name + "\", " + args_obj + ");\n"
            "console.log(result.data);\n"
            "```"
        )

    lines = [
        "# Newgen API Reference",
        "",
        "Generated from the running Gradio instance at `http://0.0.0.0:7860`.",
        "All examples use the [`@gradio/client`](https://www.npmjs.com/package/@gradio/client) Node.js package.",
        "",
        "```bash",
        "npm install @gradio/client",
        "```",
        "",
        "---",
        "",
    ]

    if named:
        lines.append("## Named Endpoints\n")
        for api_name, ep in named.items():
            params  = ep.get("parameters", [])
            returns = ep.get("returns", [])
            lines.append("### `" + api_name + "`\n")
            desc = ep.get("description") or ""
            if desc:
                lines.append(desc + "\n")
            lines.append("**Parameters**\n")
            lines.append(_build_params_table(params))
            lines.append("**Returns**\n")
            lines.append(_build_returns_table(returns))
            lines.append("**Node.js Example**\n")
            lines.append(_js_example(api_name, params))
            lines.append("")
            lines.append("---\n")

    if unnamed:
        lines.append("## Unnamed Endpoints (by index)\n")
        for idx, ep in unnamed.items():
            params  = ep.get("parameters", [])
            returns = ep.get("returns", [])
            lines.append("### Endpoint `" + str(idx) + "`\n")
            lines.append("**Parameters**\n")
            lines.append(_build_params_table(params))
            lines.append("**Returns**\n")
            lines.append(_build_returns_table(returns))
            lines.append("**Node.js Example**\n")
            lines.append(_js_example(str(idx), params))
            lines.append("")
            lines.append("---\n")

    md = "\n".join(lines)
    return _FastAPIResponse(
        content=md.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="api.md"'},
    )

@demo.app.get("/starters/{num}")
async def _serve_starter(num: int):
    """Serve a starter image by number (tries .jpg, .png, .webp).
    Used by the thumbnail <img> elements in the picgen starter grid."""
    path, ext = _find_starter_path(num)
    if path is None:
        return _FastAPIResponse(status_code=404, content=b"not found")
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    with open(path, "rb") as f:
        data = f.read()
    return _FastAPIResponse(
        content=data,
        media_type=mime,
        headers={"Cache-Control": "public, max-age=86400"},
    )

@demo.app.get("/media/{key}/{filename}")
async def _stream_media(key: str, filename: str, request: _FastAPIRequest):
    entry = _media_store_get(key)
    if entry is None:
        return _FastAPIResponse(status_code=404, content=b"not found")
    data, stored_filename = entry
    ext = stored_filename.rsplit(".", 1)[-1].lower() if "." in stored_filename else ""
    media_type_map = {
        "mp4": "video/mp4",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webm": "video/webm",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    secret_hex = request.headers.get("X-NG-Secret", "").strip()
    if secret_hex and len(secret_hex) == 64:
        key_bytes = _derive_media_key(secret_hex)
        encrypted = _encrypt_bytes(data, key_bytes)
        return _FastAPIResponse(
            content=encrypted,
            media_type="application/octet-stream",
            headers={
                "X-NG-Encrypted": "1",
                "X-NG-Original-Type": media_type,
                "X-NG-Filename": stored_filename,
                "Cache-Control": "no-store",
                "Content-Length": str(len(encrypted)),
            },
        )

    return _FastAPIResponse(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{stored_filename}"',
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
        },
    )


@demo.app.get("/logs/stream")
async def _logs_stream(request: _FastAPIRequest):
    """SSE endpoint that streams encrypted log lines to the browser.

    Each event is: data: <base64(nonce+ciphertext)>\\n\\n
    The browser decrypts with its HKDF-derived log key.
    Requires X-NG-Secret header (same localStorage secret as media).
    """
    from fastapi.responses import StreamingResponse as _StreamingResponse

    secret_hex = request.headers.get("X-NG-Secret", "").strip()
    if not secret_hex or len(secret_hex) != 64:
        return _FastAPIResponse(status_code=403, content=b"missing secret")

    log_key = _derive_log_key(secret_hex)

    async def _event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                line = _log_queue.get(timeout=0.5)
                enc = _encrypt_log_line(line, log_key)
                yield f"data: {enc}\n\n"
            except Exception:
                yield ": keepalive\n\n"

    return _StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    _start_push_api()

    if DUAL_GPU:
        print(f" GRADIO LAUNCHING  Wan on {WAN_DEVICE}, Qwen on {PIC_DEVICE}. Both tabs ready.")
        demo.queue()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, os.path.join(SCRIPT_DIR, "tmp", "gradio")],
        )
    else:
        if STARTUP_MODE == "vidgen":
            print(" GRADIO LAUNCHING  Wan on GPU, vidgen ready immediately.")
        else:
            print(" GRADIO LAUNCHING  Qwen on GPU, picgen ready immediately.")
        demo.queue()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, os.path.join(SCRIPT_DIR, "tmp", "gradio")],
        )

        def _bg_load():
            try:
                time.sleep(2.0)
                if STARTUP_MODE == "vidgen":
                    print("Background: Loading Qwen to CPU...")
                    global pic_pipe
                    start = time.time()
                    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
                    if not os.path.exists(model_index_path):
                        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
                        pipe = QwenImageEditPlusPipeline.from_pretrained(
                            "Qwen/Qwen-Image-Edit-2511",
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                            cache_dir=BASE_MODEL_LOCAL_PATH, use_safetensors=True
                        )
                    else:
                        pipe = QwenImageEditPlusPipeline.from_pretrained(
                            BASE_MODEL_LOCAL_PATH,
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                            local_files_only=True, use_safetensors=True
                        )
                    v23 = NSFW_WEIGHTS_LOCAL_PATH
                    if not os.path.exists(v23):
                        v23 = hf_hub_download(
                            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
                            cache_dir=PICGEN_MODELS_DIR,
                            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
                        )
                    sd = load_file(v23)
                    tw, vw, ew = {}, {}, {}
                    for k, v in sd.items():
                        if k.startswith("model.diffusion_model."):   tw[k.replace("model.diffusion_model.", "")] = v
                        elif k.startswith("transformer."):           tw[k.replace("transformer.", "")] = v
                        elif k.startswith("first_stage_model."):     vw[k.replace("first_stage_model.", "")] = v
                        elif k.startswith("vae."):                   vw[k.replace("vae.", "")] = v
                        elif "text_encoder" in k or "conditioner" in k:
                            if "conditioner.embedders.0." in k:      ew[k.replace("conditioner.embedders.0.", "")] = v
                            elif "text_encoder." in k:               ew[k.replace("text_encoder.", "")] = v
                    if tw: pipe.transformer.load_state_dict(tw, strict=False)
                    if vw: pipe.vae.load_state_dict(vw, strict=False)
                    if ew: pipe.text_encoder.load_state_dict(ew, strict=False)
                    del sd, tw, vw, ew
                    torch.cuda.empty_cache()
                    pipe.vae.enable_tiling()
                    pipe.vae.enable_slicing()
                    pipe.transformer.to("cpu")
                    pipe.text_encoder.to("cpu")
                    pipe.vae.to("cpu")
                    pic_pipe = pipe
                    print(f" Qwen on CPU in {time.time()-start:.1f}s  tab switching ready!")
                else:
                    print("Background: Loading Wan to CPU...")
                    _load_wan("cpu")
                    print(" Wan on CPU  tab switching ready!")
            except Exception as e:
                print(f" Background load failed: {e}")
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg_load, daemon=True).start()

        def _bg_download_loras():
            """After startup, auto-download all LoRAs that have download URLs."""
            try:
                time.sleep(5.0)
                config = load_lora_config()
                if not config:
                    return
                downloaded_any = False
                for lora_id, lora_info in config.items():
                    for side in ('high', 'low'):
                        url_key = f'{side}_url'
                        file_key = f'{side}_filename'
                        url = lora_info.get(url_key)
                        filename = lora_info.get(file_key)
                        if not url or not filename:
                            continue
                        dest = LORA_DIR / filename
                        if dest.exists():
                            continue
                        print(f"[AutoDownload] Downloading LoRA {lora_id} ({side}): {filename}")
                        success, msg = download_lora_file(url, filename)
                        if success:
                            downloaded_any = True
                            print(f"[AutoDownload] OK: {filename}")
                        else:
                            print(f"[AutoDownload] FAILED: {filename} — {msg}")
                if downloaded_any:
                    global AVAILABLE_LORAS, LORA_STATUS
                    AVAILABLE_LORAS = discover_loras()
                    LORA_STATUS = check_lora_status(config)
                    print("[AutoDownload] LoRA catalog refreshed — new files ready without restart.")
            except Exception as e:
                print(f"[AutoDownload] Error during LoRA auto-download: {e}")
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg_download_loras, daemon=True).start()

        def _bg_predownload_assets():
            """Pre-download rembg/BiRefNet + audio models after primary model swap finishes."""
            try:
                # Wait for _bg_load to finish (~100s) before competing for bandwidth
                _t0 = time.time()
                while time.time() - _t0 < 300:
                    time.sleep(10)
                    if _AUDIO_ENGINE_AVAILABLE:
                        print("[Predownload] Audio engine already ready — skipping.")
                        return
                    if time.time() - _t0 > 100:
                        break
            except Exception:
                pass

            print("[Predownload] Starting background asset pre-download...")

            # 1. rembg BiRefNet (Merge Photos background removal)
            try:
                import os
                os.environ['ORT_FORCE_CPU'] = '1'
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                from rembg import new_session as _rs
                _rs("birefnet-general")
                print("[Predownload] rembg BiRefNet ready.")
            except Exception as _e:
                print(f"[Predownload] rembg BiRefNet failed (non-fatal): {_e}")

            # 2+3. F5-TTS + HunyuanVideo-Foley weights
            try:
                if not _AUDIO_ENGINE_AVAILABLE:
                    print("[Predownload] Downloading audio engine assets...")
                    _ensure_audio_engines()
                    print("[Predownload] Audio engine assets ready.")
                else:
                    print("[Predownload] Audio engine already ready — OK.")
            except Exception as _e:
                print(f"[Predownload] Audio engine failed (non-fatal): {_e}")
                import traceback; traceback.print_exc()

            print("[Predownload] All background pre-downloads complete.")

        threading.Thread(target=_bg_predownload_assets, daemon=True).start()





