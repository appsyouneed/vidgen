import os
import subprocess
import sys
import copy
import random
import tempfile
import warnings
import time
import gc
import uuid
import threading
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
from huggingface_hub import list_models
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


# RIFE — only download if model files are not already present
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


# WAN

ORG_NAME = "TestOrganizationPleaseIgnore"
MODEL_ID = os.getenv("REPO_ID") or random.choice(
    list(list_models(author=ORG_NAME, filter='diffusers:WanImageToVideoPipeline'))
).modelId
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

try:
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    print("Loaded model from local cache.")
except Exception:
    print("Local cache miss, downloading model...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )

pipes = []
original_schedulers = []

gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Detected GPU VRAM: {gpu_vram_gb:.1f} GB")

# Full precision, full GPU — no quantization, no offloading
print("Loading full precision model directly to GPU...")
pipe.text_encoder = pipe.text_encoder.to('cuda')
pipe.transformer = pipe.transformer.to('cuda')
pipe.transformer_2 = pipe.transformer_2.to('cuda')
pipe.vae = pipe.vae.to('cuda')
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


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    quality=5,
    scheduler="UniPCMultistep",
    flow_shift=6.0,
    frame_multiplier=16,
    video_component=True,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    try:
        num_frames = get_num_frames(duration_seconds)
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        resized_image = resize_image(input_image)

        processed_last_image = None
        if last_image:
            processed_last_image = resize_and_crop_to_match(last_image, resized_image)

        video_path, task_n = run_inference(
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
        print(f"GPU complete: {task_n}")
        return (video_path if video_component else None), video_path, current_seed

    except Exception as e:
        print(f"Generation error (process kept alive): {e}")
        raise gr.Error(f"Generation failed: {e}")


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
    gr.Markdown("#### RTX Pro 6000 Blackwell — Full precision, full GPU, no quantization")

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
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, info="Used if any Guidance Scale > 1.", lines=3)
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=7, label="Video Quality")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=4, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=list(SCHEDULER_MAP.keys()),
                    value="UniPCMultistep",
                )
                flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=3.0, label="Flow Shift")
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

    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, file_output, seed_input])

    grab_frame_btn.click(fn=None, inputs=None, outputs=[timestamp_box], js=get_timestamp_js)
    timestamp_box.change(fn=extract_frame, inputs=[video_output, timestamp_box], outputs=[input_image_component])

if __name__ == "__main__":
    demo.queue(max_size=30, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        show_error=True,
        max_threads=20,
    )
