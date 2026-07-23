import os
import re
import subprocess
import sys
import copy
import random
import tempfile
import warnings
import time
import gc
import uuid
import re
import threading
from pathlib import Path
from tqdm import tqdm

# Set temp directory before torch imports
os.makedirs("/root/vidgen/tmp", exist_ok=True)
os.environ["TMPDIR"] = "/root/vidgen/tmp"
os.environ["TEMP"] = "/root/vidgen/tmp"
os.environ["TMP"] = "/root/vidgen/tmp"

import cv2
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
from huggingface_hub import list_models, list_repo_files, hf_hub_download
from torch.nn import functional as F
from PIL import Image

import gradio as gr
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.utils.export_utils import export_to_video

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "8"

warnings.filterwarnings("ignore")

# --- FRAME EXTRACTION JS & LOGIC ---

get_timestamp_js = """
function() {
    const video = document.querySelector('#generated-video video');
    if (video) {
        console.log("Video found! Time: " + video.currentTime);
        return video.currentTime;
    } else {
        console.log("No video element found.");
        return 0;
    }
}
"""


def extract_frame(video_path, timestamp):
    if not video_path:
        return None
    print(f"Extracting frame at timestamp: {timestamp}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_num = int(float(timestamp) * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame_num >= total_frames:
        target_frame_num = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# RIFE
if not os.path.exists("train_log/RIFE_HDv3.py"):
    print("Downloading RIFE Model...")
    if not os.path.exists("RIFEv4.26_0921.zip"):
        subprocess.run([
            "wget", "-q",
            "https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip",
            "-O", "RIFEv4.26_0921.zip"
        ], check=True)
    subprocess.run(["unzip", "-n", "RIFEv4.26_0921.zip"], check=True)

sys.path.append(os.path.join(os.getcwd(), "train_log"))

from train_log.RIFE_HDv3 import Model
device = torch.device("cuda")

_thread_local = threading.local()
_pipeline_lock = threading.Lock()
_pipeline_counter = 0
_scheduler_locks = []

def get_assigned_pipeline():
    global _pipeline_counter
    if not hasattr(_thread_local, 'pipe_id'):
        with _pipeline_lock:
            _thread_local.pipe_id = 0
            _pipeline_counter += 1
            print(f"Thread {threading.current_thread().name} assigned to pipeline {_thread_local.pipe_id}")
    return _thread_local.pipe_id

rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()
rife_model.device()
rife_model.flownet = rife_model.flownet.half()


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
        t = t.permute(2, 0, 1).unsqueeze(0)
        return F.pad(t, padding).half()

    def from_tensor(tensor):
        t = tensor[0, :, :H, :W]
        t = t.permute(1, 2, 0)
        return t.float().cpu().numpy()

    def make_inference(I0, I1, n):
        if rife_model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(rife_model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
            return res
        else:
            middle = rife_model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n//2)
            second_half = make_inference(middle, I1, n=n//2)
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
            I1 = to_tensor(frames_np[i+1])
            mid_tensors = make_inference(I0, I1, n_interp)
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))
            if (i + 1) % 50 == 0:
                pbar.update(50)
        pbar.update(total_steps % 50)
        output_frames.append(from_tensor(I1))

    del I0, I1, mid_tensors
    return output_frames


# --- LORA REPO SCANNING ---

LORA_REPO = "tianbugao/wan_i2v"

# High/low noise indicator patterns (order matters: more specific first)
_HIGH_PATTERNS = [
    r'[_\-\s]high[_\-\s]noise',
    r'[_\-\s]highnoise',
    r'\d+high[_\-\s]noise',
    r'[_\-\s]HIGH[_\-\s]',
    r'[_\-\s]HN[_\-\s]',
    r'[_\-\s]high[_\-\s]',
    r'[_\-\s]high\.',
    r'_high$',
]
_LOW_PATTERNS = [
    r'[_\-\s]low[_\-\s]noise',
    r'[_\-\s]lownoise',
    r'\d+low[_\-\s]noise',
    r'[_\-\s]LOW[_\-\s]',
    r'[_\-\s]LN[_\-\s]',
    r'[_\-\s]low[_\-\s]',
    r'[_\-\s]low\.',
    r'_low$',
]


def _is_high(name: str) -> bool:
    nl = name.lower()
    return bool(re.search(r'[_\-\s]high([_\-\s.]|$)|[_\-\s]hn([_\-\s.]|$)|\dhigh[_\-\s]', nl))


def _is_low(name: str) -> bool:
    nl = name.lower()
    return bool(re.search(r'[_\-\s]low([_\-\s.]|$)|[_\-\s]ln([_\-\s.]|$)|\dlow[_\-\s]', nl))


def _base_name(stem: str) -> str:
    """Strip high/low noise indicators to get a pairing key."""
    s = stem
    for pat in [
        r'[_\-\s]?high[_\-\s]?noise', r'[_\-\s]?low[_\-\s]?noise',
        r'[_\-\s]?highnoise', r'[_\-\s]?lownoise',
        r'\d+high[_\-\s]?noise', r'\d+low[_\-\s]?noise',
        r'[_\-\s]HIGH', r'[_\-\s]LOW',
        r'[_\-\s]HN([_\-\s]|$)', r'[_\-\s]LN([_\-\s]|$)',
        r'[_\-\s]high([_\-\s]|$)', r'[_\-\s]low([_\-\s]|$)',
        r'_high$', r'_low$',
    ]:
        s = re.sub(pat, '', s, flags=re.IGNORECASE).strip('_- ')
    return s


def build_lora_choices() -> dict:
    """Scan LORA_REPO loras/ folder and build display_name -> {high_tr, low_tr} dict."""
    try:
        all_files = list(list_repo_files(LORA_REPO))
    except Exception as e:
        print(f"Warning: could not list LoRA repo files: {e}")
        return {}

    lora_files = [
        f for f in all_files
        if f.startswith("loras/") and f.endswith(".safetensors")
        and not (re.search(r't2v', f, re.IGNORECASE) and not re.search(r'i2v', f, re.IGNORECASE))
    ]

    high_map = {}  # base_name -> filename (with loras/ prefix)
    low_map = {}
    singles = []

    for f in lora_files:
        stem = Path(f).stem
        if _is_high(stem):
            key = _base_name(stem)
            high_map[key] = Path(f).name
        elif _is_low(stem):
            key = _base_name(stem)
            low_map[key] = Path(f).name
        else:
            singles.append(Path(f).name)

    choices = {}
    paired_keys = set(high_map.keys()) & set(low_map.keys())
    for key in sorted(paired_keys):
        display = key
        choices[display] = {"high_tr": f"loras/{high_map[key]}", "low_tr": f"loras/{low_map[key]}"}

    # Unpaired highs
    for key in sorted(set(high_map.keys()) - paired_keys):
        choices[key] = {"high_tr": f"loras/{high_map[key]}", "low_tr": None}

    # Singles
    for fname in sorted(singles):
        display = Path(fname).stem
        choices[display] = {"high_tr": f"loras/{fname}", "low_tr": None}

    print(f"Found {len(choices)} LoRA entries ({len(paired_keys)} pairs, {len(singles)} singles)")
    return choices


LORA_CHOICES = build_lora_choices()
LORA_NAMES = ["None"] + sorted(LORA_CHOICES.keys())


# --- MMAUDIO SETUP ---

MMAUDIO_REPO = "cloud19/NSFW_MMaudio"
MMAUDIO_DIR = Path("/root/vidgen/mmaudio")
MMAUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Files to download from cloud19/NSFW_MMaudio
_MMAUDIO_FILES = [
    "weights/mmaudio_large_44k_v2.pth",
    "ext_weights/synchformer_state_dict.pth",
    "ext_weights/v1-44.pth",
    "nsfw_gold_8.5k_final.pth",
]

def _ensure_mmaudio_files():
    for repo_path in _MMAUDIO_FILES:
        local_path = MMAUDIO_DIR / repo_path
        if not local_path.exists():
            print(f"Downloading mmaudio/{repo_path}...")
            hf_hub_download(
                repo_id=MMAUDIO_REPO,
                filename=repo_path,
                local_dir=str(MMAUDIO_DIR),
                local_dir_use_symlinks=False,
            )

_ensure_mmaudio_files()

try:
    import mmaudio
    from mmaudio.eval_utils import generate, load_video, make_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils
    from mmaudio.model.sequence_config import CONFIG_44K

    _mm_dtype = torch.bfloat16
    _mm_model_path = MMAUDIO_DIR / "weights/mmaudio_large_44k_v2.pth"
    _mm_nsfw_path  = MMAUDIO_DIR / "nsfw_gold_8.5k_final.pth"
    _mm_vae_path   = MMAUDIO_DIR / "ext_weights/v1-44.pth"
    _mm_sync_path  = MMAUDIO_DIR / "ext_weights/synchformer_state_dict.pth"

    def _load_mmaudio():
        seq_cfg = CONFIG_44K
        net: MMAudio = get_my_mmaudio("large_44k").to(device, _mm_dtype).eval()
        net.load_weights(torch.load(str(_mm_nsfw_path), map_location=device, weights_only=True))
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=str(_mm_vae_path),
            synchformer_ckpt=str(_mm_sync_path),
            enable_conditions=True,
            mode="44k",
            bigvgan_vocoder_ckpt=None,
            need_vae_encoder=False,
        ).to(device, _mm_dtype).eval()
        return net, feature_utils, seq_cfg

    print("Loading MMAudio...")
    _mm_net, _mm_feature_utils, _mm_seq_cfg = _load_mmaudio()
    print("MMAudio loaded.")
    _MMAUDIO_AVAILABLE = True

except Exception as e:
    print(f"MMAudio load failed: {e}")
    _MMAUDIO_AVAILABLE = False
    _mm_net = _mm_feature_utils = _mm_seq_cfg = None


@torch.inference_mode()
def add_audio_to_video(video_path: str, audio_prompt: str, duration_sec: float) -> str:
    if not _MMAUDIO_AVAILABLE:
        return video_path
    try:
        rng = torch.Generator(device=device)
        rng.seed()
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
        video_info = load_video(Path(video_path), duration_sec)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        _mm_seq_cfg.duration = video_info.duration_sec
        _mm_net.update_seq_lengths(
            _mm_seq_cfg.latent_seq_len,
            _mm_seq_cfg.clip_seq_len,
            _mm_seq_cfg.sync_seq_len,
        )
        audios = generate(
            clip_frames,
            sync_frames,
            [audio_prompt],
            negative_text=["music"],
            feature_utils=_mm_feature_utils,
            net=_mm_net,
            fm=fm,
            rng=rng,
            cfg_strength=4.5,
        )
        audio = audios.float().cpu()[0]
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        make_video(video_info, out_path, audio, sampling_rate=_mm_seq_cfg.sampling_rate)
        return out_path
    except Exception as e:
        print(f"MMAudio generation failed: {e}")
        return video_path


# WAN
ORG_NAME = "TestOrganizationPleaseIgnore"
MODEL_ID = "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/")
os.makedirs(CACHE_DIR, exist_ok=True)

LORA_MODELS = []

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 160

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = 60.0
SEGMENT_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)  # max per segment (~10s)

SCHEDULER_MAP = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

try:
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    print("Loaded model from local cache.")
except Exception:
    print("Downloading model...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

pipes = []
original_schedulers = []

# Full precision, full GPU
print("Loading full precision model directly to GPU...")
pipe.text_encoder = pipe.text_encoder.to('cuda')
pipe.transformer = pipe.transformer.to('cuda')
pipe.transformer_2 = pipe.transformer_2.to('cuda')
pipe.vae = pipe.vae.to('cuda')
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

try:
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Detected GPU VRAM: {gpu_vram_gb:.1f} GB")
except:
    print("Could not detect GPU VRAM, continuing anyway...")

print("All model components loaded to GPU.")

pipes.append(pipe)
original_schedulers.append(copy.deepcopy(pipe.scheduler))
_scheduler_locks.append(threading.Lock())

print(f"Total pipeline instances: {len(pipes)}")

for i, lora in enumerate(LORA_MODELS):
    name_high_tr = lora["high_tr"].split(".")[0].split("/")[-1] + "Hh"
    name_low_tr = lora["low_tr"].split(".")[0].split("/")[-1] + "Ll"
    try:
        for pipe_idx, current_pipe in enumerate(pipes):
            current_pipe.load_lora_weights(lora["repo_id"], weight_name=lora["high_tr"], adapter_name=name_high_tr)
            current_pipe.load_lora_weights(lora["repo_id"], weight_name=lora["low_tr"], adapter_name=name_low_tr, load_into_transformer_2=True)
            current_pipe.set_adapters([name_high_tr, name_low_tr], adapter_weights=[1.0, 1.0])
            current_pipe.fuse_lora(adapter_names=[name_high_tr], lora_scale=lora["high_scale"], components=["transformer"])
            current_pipe.fuse_lora(adapter_names=[name_low_tr], lora_scale=lora["low_scale"], components=["transformer_2"])
            current_pipe.unload_lora_weights()
        print(f"Applied LoRA: {lora['high_tr']}, {i+1}/{len(LORA_MODELS)}")
    except Exception as e:
        print("LoRA error:", str(e))
        for current_pipe in pipes:
            current_pipe.unload_lora_weights()

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def model_title():
    repo_name = MODEL_ID.split('/')[-1].replace("_", " ")
    url = f"https://huggingface.co/{MODEL_ID}"
    return f"## This space is currently running [{repo_name}]({url}) 🐢"


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image
    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF
    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
    ))


def run_inference(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    frame_multiplier,
    quality,
    duration_seconds,
    lora_name,
    lora_high_scale,
    lora_low_scale,
    progress=gr.Progress(track_tqdm=True),
):
    pipe_id = get_assigned_pipeline()
    current_pipe = pipes[pipe_id]

    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    needs_scheduler_change = (
        scheduler_class.__name__ != current_pipe.scheduler.config._class_name or
        flow_shift != current_pipe.scheduler.config.get("flow_shift", 6.0)
    )

    if needs_scheduler_change:
        config = copy.deepcopy(original_schedulers[pipe_id].config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config['shift'] = flow_shift
        else:
            config['flow_shift'] = flow_shift
        current_pipe.scheduler = scheduler_class.from_config(config)

    # Dynamic LoRA: load, use, unload
    lora_applied = False
    if lora_name and lora_name != "None" and lora_name in LORA_CHOICES:
        lora_info = LORA_CHOICES[lora_name]
        high_tr = lora_info["high_tr"]
        low_tr = lora_info["low_tr"]
        try:
            import logging
            _hf_logger = logging.getLogger("diffusers.loaders.lora_base")
            _prev_level = _hf_logger.level
            _hf_logger.setLevel(logging.ERROR)

            name_h = re.sub(r'[^a-zA-Z0-9_]', '_', Path(high_tr).stem) + "_H"
            current_pipe.load_lora_weights(LORA_REPO, weight_name=high_tr, adapter_name=name_h)
            if low_tr:
                name_l = re.sub(r'[^a-zA-Z0-9_]', '_', Path(low_tr).stem) + "_L"
                current_pipe.load_lora_weights(LORA_REPO, weight_name=low_tr, adapter_name=name_l, load_into_transformer_2=True)
                current_pipe.set_adapters([name_h, name_l], adapter_weights=[float(lora_high_scale), float(lora_low_scale)])
            else:
                current_pipe.set_adapters([name_h], adapter_weights=[float(lora_high_scale)])

            _hf_logger.setLevel(_prev_level)
            lora_applied = True
            print(f"LoRA applied: {lora_name} (high={lora_high_scale}" + (f", low={lora_low_scale}" if low_tr else "") + ")")
        except Exception as e:
            print(f"LoRA error: {e}")
            try:
                current_pipe.unload_lora_weights()
            except Exception:
                pass

    task_name = str(uuid.uuid4())[:8]
    print(f"Generating {num_frames} frames, task: {task_name}, {duration_seconds}s, {resized_image.size}")
    start = time.time()

    generator = torch.Generator(device='cuda').manual_seed(current_seed)

    result = current_pipe(
        image=resized_image,
        last_image=processed_last_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=generator,
        output_type="np"
    )
    print("gen time passed:", time.time() - start)

    if lora_applied:
        try:
            current_pipe.unload_lora_weights()
        except Exception as e:
            print(f"LoRA unload error: {e}")

    raw_frames_np = result.frames[0]
    current_pipe.scheduler = original_schedulers[pipe_id]

    frame_factor = frame_multiplier // FIXED_FPS
    if frame_factor > 1:
        start = time.time()
        print(f"Processing frames (RIFE Multiplier: {frame_factor}x)...")
        rife_model.device()
        rife_model.flownet = rife_model.flownet.half()
        final_frames = interpolate_bits(raw_frames_np, multiplier=int(frame_factor))
        print("Interpolation time passed:", time.time() - start)
    else:
        final_frames = list(raw_frames_np)

    final_fps = FIXED_FPS * int(frame_factor)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    start = time.time()
    with tqdm(total=3, desc="Rendering Media", unit="clip") as pbar:
        pbar.update(2)
        export_to_video(final_frames, video_path, fps=final_fps, quality=quality)
        pbar.update(1)
    print(f"Export time passed, {final_fps} FPS:", time.time() - start)

    return video_path, task_name


def concatenate_videos(video_paths: list, output_path: str, quality: int):
    """Concatenate multiple mp4 clips into one using ffmpeg concat."""
    if len(video_paths) == 1:
        import shutil
        shutil.copy(video_paths[0], output_path)
        return
    list_file = output_path + ".txt"
    with open(list_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ], check=True, capture_output=True)
    os.unlink(list_file)


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=3,
    negative_prompt=default_negative_prompt,
    duration_seconds=3.5,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=True,
    quality=7,
    scheduler="UniPCMultistep",
    flow_shift=6.9,
    frame_multiplier=16,
    video_component=True,
    lora_name="None",
    lora_high_scale=1.0,
    lora_low_scale=1.0,
    add_audio=False,
    audio_prompt="natural ambient sound",
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    try:
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        resized_image = resize_image(input_image)

        # Split into segments of at most SEGMENT_DURATION seconds
        remaining = float(duration_seconds)
        segment_clips = []
        current_input = resized_image
        seg_seed = current_seed

        while remaining > 0:
            seg_dur = min(remaining, SEGMENT_DURATION)
            num_frames = get_num_frames(seg_dur)

            # For the first segment use user's last_image; subsequent segments have no last_image
            seg_last = None
            if not segment_clips and last_image:
                seg_last = resize_and_crop_to_match(last_image, resized_image)

            video_path, task_n = run_inference(
                current_input,
                seg_last,
                prompt,
                steps,
                negative_prompt,
                num_frames,
                guidance_scale,
                guidance_scale_2,
                seg_seed,
                scheduler,
                flow_shift,
                frame_multiplier,
                quality,
                seg_dur,
                lora_name,
                lora_high_scale,
                lora_low_scale,
                progress,
            )
            segment_clips.append(video_path)
            print(f"Segment complete: {task_n} ({seg_dur:.1f}s)")

            # Extract last frame of this clip to use as input for next segment
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ret, frame = cap.read()
            cap.release()
            if ret:
                current_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break  # can't extract frame, stop chaining

            remaining -= seg_dur
            seg_seed = random.randint(0, MAX_SEED)  # vary seed per segment

        # Concatenate all segments
        if len(segment_clips) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(segment_clips, final_path, quality)
            for p in segment_clips:
                try:
                    os.unlink(p)
                except Exception:
                    pass
        else:
            final_path = segment_clips[0]

        if add_audio and _MMAUDIO_AVAILABLE:
            print("Adding audio with MMAudio...")
            try:
                final_path = add_audio_to_video(final_path, audio_prompt, duration_seconds)
            except Exception as e:
                print(f"MMAudio error (returning video without audio): {e}")

        print(f"All segments complete, total duration: {duration_seconds}s")
        return (final_path if video_component else None), final_path, current_seed

    except Exception as e:
        print(f"Generation error (process kept alive): {e}")
        raise gr.Error(f"Generation failed: {e}")


# --- GRADIO UI ---

with gr.Blocks() as demo:
    gr.Markdown(model_title())

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            last_image = gr.Image(label="Last Frame (optional)", type="pil")
            prompt = gr.Textbox(label="Prompt", value=default_prompt_i2v, lines=3)
            duration_seconds = gr.Slider(
                MIN_DURATION, MAX_DURATION, value=3.5, step=0.1, label="Duration (s)"
            )
            generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Row():
                add_audio_cb = gr.Checkbox(label="Add Audio (MMAudio)", value=False)
                audio_prompt_tb = gr.Textbox(label="Audio Prompt", value="natural ambient sound")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", elem_id="generated-video", autoplay=True)
            use_as_first_btn = gr.Button("Use as First Image")
            video_file = gr.File(label="Download Video")
            seed_output = gr.Number(label="Seed Used")

    gr.Markdown("### LoRA")
    with gr.Row():
        lora_dropdown = gr.Dropdown(
            choices=LORA_NAMES,
            value="None",
            label="LoRA",
        )
    with gr.Row():
        lora_high_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA High Scale")
        lora_low_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Low Scale")

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            steps = gr.Slider(1, 50, value=3, step=1, label="Steps")
        with gr.Row():
            guidance_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Guidance Scale")
            guidance_scale_2 = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Guidance Scale 2")
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row():
            quality = gr.Slider(1, 10, value=7, step=1, label="Export Quality")
            frame_multiplier = gr.Slider(16, 64, value=16, step=16, label="Frame Multiplier (output FPS)")
        with gr.Row():
            scheduler = gr.Dropdown(
                choices=list(SCHEDULER_MAP.keys()),
                value="UniPCMultistep",
                label="Scheduler",
            )
            flow_shift = gr.Slider(1.0, 20.0, value=6.9, step=0.1, label="Flow Shift")
        negative_prompt = gr.Textbox(
            label="Negative Prompt", value=default_negative_prompt, lines=3
        )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            input_image,
            last_image,
            prompt,
            steps,
            negative_prompt,
            duration_seconds,
            guidance_scale,
            guidance_scale_2,
            seed,
            randomize_seed,
            quality,
            scheduler,
            flow_shift,
            frame_multiplier,
            gr.State(True),
            lora_dropdown,
            lora_high_scale,
            lora_low_scale,
            add_audio_cb,
            audio_prompt_tb,
        ],
        outputs=[video_output, video_file, seed_output],
    )

    timestamp_state = gr.State(0)
    use_as_first_btn.click(
        fn=None,
        inputs=[],
        outputs=[timestamp_state],
        js=get_timestamp_js,
    ).then(
        fn=extract_frame,
        inputs=[video_file, timestamp_state],
        outputs=[input_image],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
