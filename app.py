import argparse
import gc
import os
import logging
import warnings
import cv2
import torch
import torch._dynamo
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import numpy as np
from PIL import Image
import random

# Suppress all library loggers to prevent prompts/inputs leaking into logs
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("gradio").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Performance optimizations
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

_last_video_path = None
_last_video_bytes = None

parser = argparse.ArgumentParser()
parser.add_argument("--no-lora", action="store_true", help="Disable Lightning LoRA (uses base model, better prompt following, slower — recommend 20-30 steps)")
parser.add_argument("--no-quantize", action="store_true", help="Disable fp8/int8 quantization (uses full bf16, best quality, needs ~28GB+ VRAM)")
parser.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.cache/vidgen_models"), help="Directory to cache downloaded models (default: ~/.cache/vidgen_models)")
parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio on (default: 7860)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind Gradio to (default: 0.0.0.0)")
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
args = parser.parse_args()

os.makedirs(args.cache_dir, exist_ok=True)
OUTPUT_DIR = "/root/vidgen/tmp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TRANSFORMER_ID = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 240

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

print(f"Loading model... (cache dir: {args.cache_dir})")
print(f"Quantization: {'DISABLED (full bf16)' if args.no_quantize else 'ENABLED (fp8/int8)'}")

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        TRANSFORMER_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=args.cache_dir,
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        TRANSFORMER_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=args.cache_dir,
    ),
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
).to("cuda")

if not args.no_lora:
    pipe.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v",
        cache_dir=args.cache_dir,
    )
    kwargs_lora = {"load_into_transformer_2": True}
    pipe.load_lora_weights(
        "Kijai/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v_2",
        cache_dir=args.cache_dir,
        **kwargs_lora,
    )
    pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
    pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3.0, components=["transformer"])
    pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"])
    pipe.unload_lora_weights()
    print("Lightning LoRA loaded.")
else:
    print("Lightning LoRA DISABLED — using base model.")

if not args.no_quantize:
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
    print("Applying quantization...")
    quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

print("Model ready.")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()


get_timestamp_js = """
function() {
    const video = document.querySelector('#generated-video video');
    const t = video ? video.currentTime : 0;
    // Append a random suffix so the value always changes, guaranteeing .change() fires
    return t + ":" + Math.random();
}
"""


def extract_frame(video_path, timestamp_str):
    if not video_path or not timestamp_str:
        return None
    try:
        timestamp = float(str(timestamp_str).split(":")[0])
    except (ValueError, IndexError):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(int(float(timestamp) * fps), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


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


def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
    ))


def generate_video(
    input_image,
    last_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    flow_shift,
    quality,
    seed,
    randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    global _last_video_path, _last_video_bytes

    # Clear previous video from disk and memory before starting
    if _last_video_path and os.path.exists(_last_video_path):
        os.remove(_last_video_path)
        _last_video_path = None
    _last_video_bytes = None
    gc.collect()
    torch.cuda.empty_cache()

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    resized_last_image = None
    if last_image is not None:
        resized_last_image = resize_image(last_image)

    # Free PIL image references (Gradio temp files are managed by Gradio itself)
    input_image = None
    last_image = None

    pipe.scheduler.flow_shift = float(flow_shift)

    call_kwargs = dict(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    )
    if resized_last_image is not None:
        call_kwargs["last_image"] = resized_last_image

    output_frames_list = pipe(**call_kwargs).frames[0]

    video_path = os.path.join(OUTPUT_DIR, "vidgen_output.mp4")
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS, quality=int(quality))
    output_frames_list = None

    # Read into memory so the file can be deleted after Gradio serves it
    with open(video_path, "rb") as f:
        _last_video_bytes = f.read()
    _last_video_path = video_path

    return video_path, current_seed


with gr.Blocks() as demo:
    gr.Markdown("# Wan 2.2 I2V (14B) with Lightning LoRA")
    gr.Markdown("Fast 4-8 step image-to-video generation with Lightning LoRA and optional fp8 quantization.")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="First Frame (required)")
            last_image_component = gr.Image(type="pil", label="Last Frame (optional — guides the ending of the video)")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v, lines=3)
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=5.0,
                label="Duration (seconds)",
                info=f"Model supports {MIN_FRAMES_MODEL}–{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps."
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=3, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.5, value=1.0, label="Guidance Scale (high noise stage)")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.5, value=1.0, label="Guidance Scale 2 (low noise stage)")
                flow_shift_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.5, value=10.0, label="Flow Shift", info="Controls motion dynamics. Higher = more motion.")
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=7, label="Video Quality", info="Higher = better quality but larger file size.")

            generate_button = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=True, sources=["upload"], elem_id="generated-video")
            with gr.Row():
                grab_frame_btn = gr.Button("📸 Use Current Frame as First Image", variant="secondary")
                timestamp_box = gr.Textbox(value="", visible=False)

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input,
        flow_shift_input, quality_slider, seed_input, randomize_seed_checkbox,
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    grab_frame_btn.click(fn=None, inputs=None, outputs=[timestamp_box], js=get_timestamp_js)
    timestamp_box.change(fn=extract_frame, inputs=[video_output, timestamp_box], outputs=[input_image_component])

css = """
.gradio-container .contain{max-width: 1000px !important; margin: 0 auto !important}
"""

if __name__ == "__main__":
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[OUTPUT_DIR],
        css=css,
    )
