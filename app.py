import os
import sys
import copy
import random
import io
import warnings
import time
import gc
import uuid
import threading
from tqdm import tqdm

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

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig

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


def extract_frame(video_bytes, timestamp):
    """Extract a frame from in-memory video bytes at the given timestamp."""
    if not video_bytes:
        return None

    print(f"Extracting frame at timestamp: {timestamp}")

    # Write to a tiny temp buffer for OpenCV (OpenCV needs seekable input)
    buf = np.frombuffer(video_bytes, dtype=np.uint8)
    cap = cv2.VideoCapture()
    # Use imdecode-style approach via imencode buffer trick
    # OpenCV VideoCapture can read from memory via FFMPEG pipe trick, but simplest
    # cross-platform way is a BytesIO-backed approach using cv2.VideoCapture with
    # a memory-mapped approach. Since OpenCV doesn't support BytesIO directly,
    # we use a named pipe or just decode via imageio. Use imageio for in-memory.
    import imageio.v3 as iio
    try:
        frames = iio.imread(io.BytesIO(video_bytes), plugin="pyav", index=None)
        # frames shape: (T, H, W, C)
        fps_meta = iio.improps(io.BytesIO(video_bytes), plugin="pyav").fps
        if fps_meta is None:
            fps_meta = 16
        target_frame = int(float(timestamp) * fps_meta)
        target_frame = max(0, min(target_frame, len(frames) - 1))
        return frames[target_frame]  # RGB numpy array
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return None

# --- END FRAME EXTRACTION LOGIC ---


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# In-memory video store: holds only the current generated video bytes
_current_video_bytes = None
_video_lock = threading.Lock()


def clear_current_video():
    global _current_video_bytes
    with _video_lock:
        _current_video_bytes = None
    gc.collect()


# Clear any existing CUDA memory from previous runs
print("Clearing GPU memory from previous runs...")
clear_vram()
torch.cuda.reset_peak_memory_stats()

# RIFE - download only if missing, never re-download
import subprocess
# Need both train_log/flownet.pkl (weights) and model/warplayer.py (architecture)
if not os.path.exists("train_log/flownet.pkl") or not os.path.exists("model/warplayer.py"):
    print("Setting up RIFE...")
    subprocess.run(["rm", "-rf", "train_log", "__MACOSX", "RIFEv4.26_0921.zip"], check=False)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hzwer/Practical-RIFE.git", "/tmp/rife"], check=True)
    subprocess.run(["cp", "-r", "/tmp/rife/model", "."], check=True)
    subprocess.run(["rm", "-rf", "/tmp/rife"], check=False)
    subprocess.run([
        "wget", "-q",
        "https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip",
        "-O", "RIFEv4.26_0921.zip"
    ], check=True)
    subprocess.run(["unzip", "-o", "RIFEv4.26_0921.zip"], check=True)
    subprocess.run(["rm", "-f", "RIFEv4.26_0921.zip"], check=False)

sys.path.append(os.path.join(os.getcwd(), "train_log"))

from train_log.RIFE_HDv3 import Model
device = torch.device("cuda")

_pipeline_lock = threading.Lock()
_scheduler_locks = []

rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()


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
    torch.cuda.empty_cache()
    return output_frames


# WAN — always use local cache, never re-download if files exist
MODEL_ID = os.getenv("REPO_ID") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
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
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

SCHEDULER_MAP = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

# Load model — try local cache first, fall back to download on first run
print("Loading pipeline from cache...")
try:
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
except Exception:
    print("Not in cache, downloading model (first run only)...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )

pipes = []
original_schedulers = []

print("Creating 1 pipeline instance")

gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Detected GPU VRAM: {gpu_vram_gb:.1f} GB")

# Always quantize (RTX 5090 benefits from fp8, and --quantize flag is now irrelevant)
print("Applying quantization on CPU...")
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

print("Enabling VAE slicing and tiling...")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

if gpu_vram_gb <= 32:
    print("Enabling model CPU offloading...")
    pipe.enable_model_cpu_offload()
else:
    print("Moving quantized model to GPU...")
    for name in ["text_encoder", "transformer", "transformer_2", "vae"]:
        print(f"  - Moving {name}...")
        setattr(pipe, name, getattr(pipe, name).to('cuda'))
        clear_vram()

pipes.append(pipe)
original_schedulers.append(copy.deepcopy(pipe.scheduler))
_scheduler_locks.append(threading.Lock())

print(f"Total pipeline instances: {len(pipes)}")

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def model_title():
    repo_name = MODEL_ID.split('/')[-1].replace("_", " ")
    url = f"https://huggingface.co/{MODEL_ID}"
    return f"## [{repo_name}]({url})"


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
    progress=gr.Progress(track_tqdm=True),
):
    current_pipe = pipes[0]
    target_device = 'cuda'

    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    needs_scheduler_change = (
        scheduler_class.__name__ != current_pipe.scheduler.config._class_name or
        flow_shift != current_pipe.scheduler.config.get("flow_shift", 6.0)
    )

    if needs_scheduler_change:
        config = copy.deepcopy(original_schedulers[0].config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config['shift'] = flow_shift
        else:
            config['flow_shift'] = flow_shift
        current_pipe.scheduler = scheduler_class.from_config(config)

    clear_vram()

    task_name = str(uuid.uuid4())[:8]
    print(f"Generating {num_frames} frames, task: {task_name}, dur={duration_seconds}, size={resized_image.size}")
    start = time.time()

    generator = torch.Generator(device=target_device).manual_seed(current_seed)

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
    print("gen time:", time.time() - start)

    raw_frames_np = result.frames[0]  # (T, H, W, C) float32
    current_pipe.scheduler = original_schedulers[0]

    frame_factor = frame_multiplier // FIXED_FPS
    if frame_factor > 1:
        start = time.time()
        print(f"RIFE interpolation {frame_factor}x...")
        rife_model.device()
        rife_model.flownet = rife_model.flownet.half()
        final_frames = interpolate_bits(raw_frames_np, multiplier=int(frame_factor))
        print("Interpolation time:", time.time() - start)
    else:
        final_frames = list(raw_frames_np)

    final_fps = FIXED_FPS * int(frame_factor)

    # Render video to in-memory bytes — no disk writes
    start = time.time()
    buf = io.BytesIO()
    import tempfile, os as _os
    # export_to_video requires a file path; use a true temp file but delete immediately after read
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        export_to_video(final_frames, tmp_path, fps=final_fps, quality=quality)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        _os.unlink(tmp_path)

    print(f"Export time ({final_fps} FPS):", time.time() - start)

    # Free raw frames immediately
    del raw_frames_np, final_frames
    gc.collect()

    return video_bytes, task_name


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=3,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    quality=5,
    scheduler="UniPCMultistep",
    flow_shift=10.0,
    frame_multiplier=16,
    video_component=True,
    progress=gr.Progress(track_tqdm=True),
):
    global _current_video_bytes

    if input_image is None:
        raise gr.Error("Please upload an input image.")

    # Clear previous video from memory before generating new one
    clear_current_video()

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    processed_last_image = None
    if last_image:
        processed_last_image = resize_and_crop_to_match(last_image, resized_image)

    video_bytes, task_n = run_inference(
        resized_image,
        processed_last_image,
        prompt,
        steps,
        negative_prompt,
        num_frames,
        guidance_scale,
        guidance_scale_2,
        current_seed,
        scheduler,
        flow_shift,
        frame_multiplier,
        quality,
        duration_seconds,
        progress,
    )

    # Free input images from memory
    del resized_image, processed_last_image
    gc.collect()

    # Store only current video in memory
    with _video_lock:
        _current_video_bytes = video_bytes

    print(f"GPU complete: {task_n}")

    # Return bytes directly to Gradio — no file path needed
    return (video_bytes if video_component else None), video_bytes, current_seed


CSS = """
#hidden-timestamp {
    opacity: 0;
    height: 0px;
    width: 0px;
    margin: 0px;
    padding: 0px;
    overflow: hidden;
    position: absolute;
    pointer-events: none;
}
"""


with gr.Blocks() as demo:
    gr.Markdown(model_title())

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard"])
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            frame_multi = gr.Dropdown(
                choices=[FIXED_FPS, FIXED_FPS*2, FIXED_FPS*4, FIXED_FPS*8],
                value=FIXED_FPS,
                label="Video Fluidity (Frames per Second)",
                info="Extra frames generated using RIFE flow estimation."
            )
            with gr.Accordion("Advanced Settings", open=False):
                last_image_component = gr.Image(type="pil", label="Last Image (Optional)", sources=["upload", "clipboard"])
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=7, label="Video Quality")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=3, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=list(SCHEDULER_MAP.keys()),
                    value="UniPCMultistep",
                )
                flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=10.0, label="Flow Shift")
                play_result_video = gr.Checkbox(label="Display result", value=True, interactive=True)

            generate_button = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, sources=["upload"], buttons=["download", "share"], interactive=True, elem_id="generated-video")

            with gr.Row():
                grab_frame_btn = gr.Button("📸 Use Current Frame as Input", variant="secondary")
                timestamp_box = gr.Number(value=0, label="Timestamp", visible=True, elem_id="hidden-timestamp")

            file_output = gr.File(label="Download Video")

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider, frame_multi,
        play_result_video
    ]

    generate_button.click(
        fn=generate_video,
        inputs=ui_inputs,
        outputs=[video_output, file_output, seed_input],
    )

    grab_frame_btn.click(
        fn=None,
        inputs=None,
        outputs=[timestamp_box],
        js=get_timestamp_js
    )

    timestamp_box.change(
        fn=extract_frame,
        inputs=[video_output, timestamp_box],
        outputs=[input_image_component]
    )

if __name__ == "__main__":
    demo.queue(max_size=30, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        show_error=True,
        max_threads=20,
    )
