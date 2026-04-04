import gradio as gr
import numpy as np
import random
import torch
# spaces import REMOVED — ZeroGPU is HuggingFace-only infrastructure, not needed locally


import gc

from safetensors.torch import load_file
# hf_hub_download import REMOVED — replaced with local path constants below


from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
# from optimization import optimize_pipeline_
# from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
# from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
# from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

# InferenceClient import REMOVED — was calling Nebius/HuggingFace cloud API
# Replaced below with a local Qwen2.5-VL model running on the same GPU
import math

import os
import base64
from io import BytesIO
import json

# ---------------------------------------------------------------------------
# LOCAL MODEL PATHS — relative to this script for portability
# Models will auto-download to ./models/ folder on first run
# ---------------------------------------------------------------------------
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

BASE_MODEL_LOCAL_PATH = os.path.join(MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

# Detect operating system
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
print(f"Operating System: {platform.system()}")


def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Clear GPU memory from previous runs ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Clearing GPU memory from previous runs...")
clear_vram()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model pipeline
from safetensors.torch import load_file
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig


# ---------------------------------------------------------------------------
# INTELLIGENT MEMORY MANAGEMENT - Works on any GPU
# ---------------------------------------------------------------------------
def setup_intelligent_memory_pipeline(pipe, apply_quantization=True):
    """Intelligent memory management based on actual VRAM availability"""
    if not torch.cuda.is_available():
        print("No GPU detected - using CPU only")
        return pipe
    
    clear_vram()
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.1f}GB")
    print(f"Base model size: ~55GB unquantized, ~20GB with quantization")
    
    # Apply quantization to reduce memory footprint (only if requested)
    if apply_quantization:
        print("Applying quantization on CPU...")
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    
    # Enable VAE optimizations for all GPUs
    print("Enabling VAE slicing and tiling...")
    if hasattr(pipe, 'vae'):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # Memory strategy based on VRAM
    if total_vram >= 40:
        print("High VRAM GPU: Loading fully on GPU")
        print("  - Moving text_encoder...")
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to('cuda')
        clear_vram()
        
        print("  - Moving transformer...")
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            pipe.transformer = pipe.transformer.to('cuda')
        clear_vram()
        
        print("  - Moving VAE...")
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            pipe.vae = pipe.vae.to('cuda')
        clear_vram()
        
        return pipe
    elif total_vram >= 16:
        print("Mid-range GPU: Using model CPU offloading")
        pipe.enable_model_cpu_offload()
        return pipe
    else:
        print("Low VRAM GPU: Using sequential CPU offloading")
        pipe.enable_sequential_cpu_offload()
        return pipe

print("loading base pipeline architecture...")

# Auto-download if not present - check for model_index.json
model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
if not os.path.exists(model_index_path):
    print(f"Base model not found. Downloading to {BASE_MODEL_LOCAL_PATH}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Download directly to local path
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        cache_dir=BASE_MODEL_LOCAL_PATH,
    )
    # Don't apply memory management yet - need to load NSFW weights first
else:
    print(f"Loading from local path: {BASE_MODEL_LOCAL_PATH}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL_LOCAL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # Don't apply memory management yet - need to load NSFW weights first

# force euler ancestral scheduler
#pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 2. LOAD RAW WEIGHTS FROM LOCAL FILE
# ------------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CHANGE 3 of 4: NSFW WEIGHTS — auto-download to local models folder
# ---------------------------------------------------------------------------
print("accessing v23 checkpoint...")

# Auto-download if not present
if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
    print(f"NSFW weights not found. Downloading to {NSFW_WEIGHTS_LOCAL_PATH}...")
    os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
    from huggingface_hub import hf_hub_download
    v23_path = hf_hub_download(
        repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
        filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
        cache_dir=MODELS_DIR,
        local_dir=os.path.join(MODELS_DIR, "rapid-aio"),
        local_dir_use_symlinks=False,
    )
else:
    print(f"Loading from local path: {NSFW_WEIGHTS_LOCAL_PATH}")
    v23_path = NSFW_WEIGHTS_LOCAL_PATH

print(f"loading 28GB state dict into cpu memory...")
state_dict = load_file(v23_path)

# 3. DYNAMIC COMPONENT MAPPING (NO ASSUMPTIONS)
# ------------------------------------------------------------------------------
print("sorting weights into components...")

# containers for the sorted weights
transformer_weights = {}
vae_weights = {}
text_encoder_weights = {}

# analyze the first key to determine the format
first_key = next(iter(state_dict.keys()))
print(f"format detection - first key detected: {first_key}")

# iterate and sort
for k, v in state_dict.items():
    # MAPPING: TRANSFORMER
    # ComfyUI usually prefixes with 'model.diffusion_model.'
    if k.startswith("model.diffusion_model."):
        new_key = k.replace("model.diffusion_model.", "")
        transformer_weights[new_key] = v
    # Or sometimes just 'transformer.' or 'model.'
    elif k.startswith("transformer."):
        new_key = k.replace("transformer.", "")
        transformer_weights[new_key] = v
    
    # MAPPING: VAE
    # ComfyUI prefix: 'first_stage_model.'
    elif k.startswith("first_stage_model."):
        new_key = k.replace("first_stage_model.", "")
        vae_weights[new_key] = v
    # Diffusers prefix: 'vae.'
    elif k.startswith("vae."):
        new_key = k.replace("vae.", "")
        vae_weights[new_key] = v

    # MAPPING: TEXT ENCODER
    # ComfyUI prefix: 'conditioner.embedders.' or 'text_encoder.'
    elif "text_encoder" in k or "conditioner" in k:
        # this is tricky, we try to keep the suffix
        if "conditioner.embedders.0." in k:
            new_key = k.replace("conditioner.embedders.0.", "")
            text_encoder_weights[new_key] = v
        elif "text_encoder." in k:
            new_key = k.replace("text_encoder.", "")
            text_encoder_weights[new_key] = v

# 4. INJECT WEIGHTS (COMPONENT LEVEL)
# ------------------------------------------------------------------------------
print(f"injection statistics:")
print(f" - transformer keys found: {len(transformer_weights)}")
print(f" - vae keys found: {len(vae_weights)}")
print(f" - text encoder keys found: {len(text_encoder_weights)}")

if len(transformer_weights) > 0:
    print("injecting transformer weights...")
    msg = pipe.transformer.load_state_dict(transformer_weights, strict=False)
    print(f"transformer missing keys: {len(msg.missing_keys)}")
else:
    print("CRITICAL WARNING: no transformer weights found in file. check mapping logic.")

if len(vae_weights) > 0:
    print("injecting vae weights...")
    pipe.vae.load_state_dict(vae_weights, strict=False)

if len(text_encoder_weights) > 0:
    print("injecting text encoder weights...")
    # text encoder structure can vary wildly, strict=False is mandatory here
    pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

# 5. CLEANUP & APPLY MEMORY MANAGEMENT
# ------------------------------------------------------------------------------
del state_dict
del transformer_weights
del vae_weights
del text_encoder_weights
gc.collect()
torch.cuda.empty_cache()

# NOW apply memory management WITHOUT quantization (causes CUDA errors)
print("Applying memory optimizations after weight injection...")
pipe = setup_intelligent_memory_pipeline(pipe, apply_quantization=False)


#################################


# # --- 1. setup pipeline with lightning (this works fine) ---
# pipe = QwenImageEditPlusPipeline.from_single_file(
#     "path/to/Qwen-Rapid-AIO-NSFW-v21.safetensors",
#     original_config="Qwen/Qwen-Image-Edit-2511", # pulls the config from the base repo
#     scheduler=scheduler,
#     torch_dtype=torch.bfloat16 # use bf16 for speed on zerogpu
# ).to("cuda")

# print("loading lightning lora...")
# pipe.load_lora_weights(
#     "lightx2v/Qwen-Image-Edit-2511-Lightning", 
#     weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# )
# pipe.fuse_lora()
# print("lightning lora fused.")


# # Apply the same optimizations from the first version
# pipe.transformer.__class__ = QwenImageTransformer2DModel
# pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# # --- Ahead-of-time compilation ---
# optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(output_images):
    """Convert output images to input format for the gallery"""
    if output_images is None or len(output_images) == 0:
        return []
    return output_images

def add_starter_image(starter_num, current_images):
    """Add a starter image to the current gallery"""
    from PIL import Image
    import os
    
    starter_path = f"starters/start{starter_num}.jpg"
    if not os.path.exists(starter_path):
        return current_images
    
    img = Image.open(starter_path)
    if current_images is None:
        return [img]
    return list(current_images) + [img]

# --- Main Inference Function (with hardcoded negative prompt) ---
# ---------------------------------------------------------------------------
# CHANGE 4 of 4: @spaces.GPU() decorator REMOVED
# That decorator allocates a GPU from HuggingFace's shared ZeroGPU pool.
# On a local VPS the GPU is always available — the decorator is not needed
# and would crash without the `spaces` library installed.
# The entire function body below is 100% identical to the original.
# ---------------------------------------------------------------------------
def infer(
    images,
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
    Run image-editing inference using the Qwen-Image-Edit pipeline.

    Parameters:
        images (list): Input images from the Gradio gallery (PIL or path-based).
        prompt (str): Editing instruction.
        seed (int): Random seed for reproducibility.
        randomize_seed (bool): If True, overrides seed with a random value.
        true_guidance_scale (float): CFG scale used by Qwen-Image.
        num_inference_steps (int): Number of diffusion steps.
        height (int | None): Optional output height override.
        width (int | None): Optional output width override.
        num_images_per_prompt (int): Number of images to generate.
        progress: Gradio progress callback.

    Returns:
        tuple: (generated_images, seed_used, UI_visibility_update)
    """
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load input images into PIL Images
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item[0], str):
                    pil_images.append(Image.open(item[0]).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception:
                continue

    if height==256 and width==256:
        height, width = None, None
    
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    

    # Generate the image
    image = pipe(
        image=pil_images if len(pil_images) > 0 else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    # Return images, seed, and make button visible
    return image, seed, gr.update(visible=True)

# --- Examples and UI Layout ---
examples = []

css = """
body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}
#col-container {
    margin: 0 !important;
    max-width: 100% !important;
    padding: 0 !important;
}
.contain {
    padding: 0 !important;
}
#preset-row {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
#preset-row > * {
    flex: 1 !important;
}
#preset-row button {
    flex: 0 0 auto !important;
    min-width: 80px !important;
}
#preset-row input[type="text"] {
    pointer-events: none !important;
    user-select: none !important;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Row(elem_id="preset-row"):
            preset_dropdown = gr.Dropdown(
                label="Quick Prompts",
                choices=[
                    "Lift Up Vag Flash",
                    "Same Place Sitting Tease",
                    "Same Place Shh Tease",
                    "Same Place Shh Tease 2",
                    "1 Leg Up Inviting Tease",
                    "Same Place Squatting Frontal",
                    "Dick Holding Pleasure Tease",
                    "Dick Holding Almost BJ",
                    "Same Place Partial BJ",
                    "Side View Oral",
                    "Remove Her Clothes",
                    "Full Body Nudity",
                    "Secret Silent Sex",
                    "Rear View Anal",
                    "Spread On Bed Pose",
                    "Come Get It Pose",
                    "Secret Silent Room Sex",
                    "Front View Couch Sex",
                    "Rear View Couch Sex"
                ],
                value=None,
                interactive=True,
                show_label=False,
                allow_custom_value=False
            )
            run_button_top = gr.Button("Edit!", variant="primary", size="sm")
        
        with gr.Row(elem_id="preset-row"):
            preset_dropdown2 = gr.Dropdown(
                label="More Prompts",
                choices=[
                    "Balcony Risky Sex",
                    "Bathroom Sink Sex",
                    "Bed Arching Back Orgasm Pose",
                    "Bed Doggy Style Front Turned",
                    "Bent Over Table Anal-Vaginal Tease",
                    "Car Backseat Ride",
                    "Chair Ride Facing Camera",
                    "Close Up Face With Vaginal Spread And Penis Tease",
                    "Couch Lap Dance Penetration",
                    "Intense Missionary Leg Lift",
                    "Kitchen Counter Penetration",
                    "Kneeling Frontal Invitation With Fingers",
                    "Lotus Position Deep Intimate",
                    "Mirror Reflection Tease",
                    "Overhead Squat Ride Ecstasy",
                    "Prone Bone Close Up",
                    "Reverse Cowgirl Deep Squat",
                    "Shower Wet Sex",
                    "Sideways Spooning Vaginal Sex",
                    "Standing Bent Over Window Sex",
                    "Standing Carry Position Tease",
                    "Wall Leaning Deep Penetration Tease",
                    "Wall Pressed Standing Sex",
                    "Yoga Mat Deep Stretch Sex",
                    "Full Penetration Attempt 1",
                    "Full Penetration Attempt 2",
                    "Full Penetration Attempt 3",
                    "Full Penetration Attempt 4",
                    "Full Penetration Attempt 5",
                    "Full Penetration Attempt 6",
                    "Full Penetration Attempt 7",
                    "Full Penetration Attempt 8",
                    "Full Penetration Attempt 9",
                    "Full Penetration Attempt 10"
                ],
                value=None,
                interactive=True,
                show_label=False,
                allow_custom_value=False
            )
            run_button_top2 = gr.Button("Edit!", variant="primary", size="sm")
        
        with gr.Row(elem_id="preset-row"):
            preset_dropdown3 = gr.Dropdown(
                label="Couple Prompts",
                choices=[
                    "Merge Together",
                    "Kissing Handjob",
                    "Shower Sex Together",
                    "Lift Up Vag Flash (Couple)",
                    "Same Place Sitting Tease (Couple)",
                    "Same Place Shh Tease (Couple)",
                    "Same Place Shh Tease 2 (Couple)",
                    "1 Leg Up Inviting Tease (Couple)",
                    "Same Place Squatting Frontal (Couple)",
                    "Dick Holding Pleasure Tease (Couple)",
                    "Dick Holding Almost BJ (Couple)",
                    "Same Place Partial BJ (Couple)",
                    "Side View Oral (Couple)",
                    "Remove Her Clothes (Couple)",
                    "Full Body Nudity (Couple)",
                    "Secret Silent Sex (Couple)",
                    "Rear View Anal (Couple)",
                    "Spread On Bed Pose (Couple)",
                    "Come Get It Pose (Couple)",
                    "Secret Silent Room Sex (Couple)",
                    "Front View Couch Sex (Couple)",
                    "Rear View Couch Sex (Couple)",
                    "Balcony Risky Sex (Couple)",
                    "Bathroom Sink Sex (Couple)",
                    "Bed Arching Back Orgasm Pose (Couple)",
                    "Bed Doggy Style Front Turned (Couple)",
                    "Bent Over Table Anal-Vaginal Tease (Couple)",
                    "Car Backseat Ride (Couple)",
                    "Chair Ride Facing Camera (Couple)",
                    "Close Up Face With Vaginal Spread And Penis Tease (Couple)",
                    "Couch Lap Dance Penetration (Couple)",
                    "Intense Missionary Leg Lift (Couple)",
                    "Kitchen Counter Penetration (Couple)",
                    "Kneeling Frontal Invitation With Fingers (Couple)",
                    "Lotus Position Deep Intimate (Couple)",
                    "Mirror Reflection Tease (Couple)",
                    "Overhead Squat Ride Ecstasy (Couple)",
                    "Prone Bone Close Up (Couple)",
                    "Reverse Cowgirl Deep Squat (Couple)",
                    "Shower Wet Sex (Couple)",
                    "Sideways Spooning Vaginal Sex (Couple)",
                    "Standing Bent Over Window Sex (Couple)",
                    "Standing Carry Position Tease (Couple)",
                    "Wall Leaning Deep Penetration Tease (Couple)",
                    "Wall Pressed Standing Sex (Couple)",
                    "Yoga Mat Deep Stretch Sex (Couple)",
                    "Full Penetration Attempt 1 (Couple)",
                    "Full Penetration Attempt 2 (Couple)",
                    "Full Penetration Attempt 3 (Couple)",
                    "Full Penetration Attempt 4 (Couple)",
                    "Full Penetration Attempt 5 (Couple)",
                    "Full Penetration Attempt 6 (Couple)",
                    "Full Penetration Attempt 7 (Couple)",
                    "Full Penetration Attempt 8 (Couple)",
                    "Full Penetration Attempt 9 (Couple)",
                    "Full Penetration Attempt 10 (Couple)"
                ],
                value=None,
                interactive=True,
                show_label=False,
                allow_custom_value=False
            )
            run_button_top3 = gr.Button("Edit!", variant="primary", size="sm")
        
        with gr.Row(elem_id="preset-row"):
            preset_dropdown4 = gr.Dropdown(
                label="Group Prompts (1 Man + 2-3 Women)",
                choices=[
                    "Merge Together 1",
                    "Merge Together 2",
                    "Naked Together",
                    "Near Dick 1",
                    "Near Dick 2",
                    "Near Dick 3",
                    "Near Dick 4",
                    "Near Dick 5",
                    "Shh Holding Dick",
                    "Both Near Dick",
                    "Both Holding Dick 1",
                    "Both Holding Dick 2",
                    "Single BJ",
                    "Single Anal",
                    "Left 1 BJ",
                    "Left 1 Anal",
                    "Left 1 Vag On Knee",
                    "Fucked after 1",
                    "Fucked after 2",
                    "Penetration 1",
                    "Penetration 2",
                    "Penetration 3",
                    "Shower Sex Together",
                    "Kissing Handjob",
                    "BJ",
                    "BJ 2",
                    "Bed BJ 1",
                    "Bed BJ 2",
                    "Bed Sex 1",
                    "Bed Sex2",
                    "Bed Sex 3",
                    "Bed Sex 4",
                    "Side Fuck",
                    "Kissing Sex"
                ],
                value=None,
                interactive=True,
                show_label=False,
                allow_custom_value=False
            )
            run_button_top4 = gr.Button("Edit!", variant="primary", size="sm")
        
        with gr.Row():
            start1_btn = gr.Button("Start 1", size="sm")
            start2_btn = gr.Button("Start 2", size="sm")
            start3_btn = gr.Button("Start 3", size="sm")
            start4_btn = gr.Button("Start 4", size="sm")
        
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil", interactive=False)
                use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

        with gr.Row():
            prompt = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
                    lines=3,
                    max_lines=10,
            )
            run_button = gr.Button("Edit!", variant="primary")

        num_images_per_prompt = gr.Slider(
            label="Number of images",
            minimum=1,
            maximum=4,
            step=1,
            value=1,
        )
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt (what you DON'T want)",
            placeholder="censored, mosaic, blurred, clothed, soft, partial",
            value="",
            lines=2,
            max_lines=5,
        )

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=4,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
            
            fullscreen_toggle = gr.Checkbox(
                label="Full Screen Mode",
                value=False,
                info="Expand image boxes to full page width"
            )
            
            keyboard_toggle = gr.Checkbox(
                label="Disable On-Screen Keyboard",
                value=False,
                info="Prevent keyboard from appearing on mobile devices"
            )

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    # Preset prompt dictionary (in exact order from picgen Prompts.txt)
    prompts_dict = {
        "Lift Up Vag Flash": "She lifts the entire bottom portion of her clothing upward to just above her vagina area. All of the bottom fabric is now fully scrunched and bunched tightly in her hands and not visible anywhere else, completely exposing her shaved, beautiful vagina. Any other objects that were in her hands must be removed. Tiny gap of space between her legs. Everything else in the scene remains exactly the same as before.",
        "Same Place Sitting Tease": "She sits on the floor in the same room with the same clothes on but the top of them are pulled down to be under her exposed breasts which must be firm and perky and standing up straight with nipples hard and pointing right at the camera without any  sagging. and the bottom of her clothes are pulled up in a way to not cover her vagina. but with her knees facing straight towards me but her vagina is facing at me, same exact identical face with all of it's exact features and same exact original expression with no changes except her eyes are looking directly at the camera, she uses her hands to grip both sides of her vulva and fully part/spread both sides so that her vagina insides are visible and as far as they be parted  as those same hands also push against her inner thighs visibly, everything else the same",
        "Same Place Shh Tease": "Her left hand's index finger is hovering away from her vagina and she is pointing it directly at her vagina hole while she also holds her right hand's index finger on her mouth saying shh but still smiling extremely sexually suggestively and invitingly, everything else stays the same!",
        "Same Place Shh Tease 2": "Her left hand is being used to touch her vaginal clitoris without covering the whole vagina while she also holds her right hand's index finger on her mouth saying shh but still smiling extremely sexually suggestively and invitingly, vagina fully visible between her now slightly more spread legs. tonge is pressed between her teeth in a sexually suggestive way. Exact same identical face with all the exact same identical features. everything else stays the same!",
        "1 Leg Up Inviting Tease": "She holds 1 leg far up in the air, to fully show vagina and, vagina is facing at me, everything else the same",
        "Same Place Squatting Frontal": "Same exact identical face with all the same exact identical features. \nhead is turned almost fully around naturally to look directly into my eyes. She is being fully penetrated by a small but wide penis coming from the bottom of the picture attached to a man who is no visible who has his legs stretched under her. her mouth wide open with her lips making an o shape. She looks like she is imdeep in love and dreaming in a fantasy. She is very close to the camera sitting on the penis that is now fully inside her vagina hole, showing how it parts her vulva from its penetration.\nremove the objects in the way. She touches her clitoris with just one of her hand's fingers and her other hand stays is grabbing the penis in her vagina.\neverything else is the exact same!",
        "Dick Holding Pleasure Tease": "She is sitting up close to the camera with her legs spread. With her right hand, she is eagerly, firmly, and tightly gripping a very short 5 inch small penis with girth coming from the bottom of the image; the penis being held far off to the left of her vagina so you can see her whole vaginaon the right of it. The penis is coming from an otherwise completely unseen man whose legs are stretched out flat beneath her.  she sits directly behind the penis, holding it far off to the left of her vagina, in her hand only. You can see her whole vagina because she also uses her left hand to play with it while she holds the penis to the far left Outside of it. YOU MUST BE ABLE TO SEE HER ENTIRE VAGINA ON THE RIGHT OF THE PENIS! She stares erotically directly into the camera.  Both of her breasts are fully exposed outside of her clothing, with visibly hardened nipples. . Exact same woman, same face and features. Everything else must remain exactly the same.",
        "Dick Holding Almost BJ": "She leans forward with her mouth wide open and lips in an O shape so that her mouth is closer to the penis but not touching it. her tongue is visible only behind the penis pressed against it. She is still firmly grasping the penis the same exact way. Close up of her face. Same exact identical face with all the same features. Everything else is the same",
        "Same Place Partial BJ": "Same environment.\nShe is now on her stomach as far back enough as needed and bending forward enough so that the penis that is coming up from the bottom of the photo that is attached to an unseen man is now all the way in her mouth. put the penis much deeper in her mouth.  You can see her butt visibly behind her head as well which has smooth skin and arches for the cheeks. same exact identical face with all the same exact identical features. Everything else is the exact same.",
        "Side View Oral": "She is wearing the same clothing the same exact way. Her left hand is sexually touching her vagina between her fully spread legs. Her vagina is fully visible and you can see she put fingers inside of it's hole. \nThe same sized but smaller penis from the original photo is now  all the way deep inside of her mouth and you can see that it is in her throat because there is a bulge from it on her throat. The very short penis is coming from the right edge of the photo. Her lips are tightly closed surrounding the bottom part of the penis. She is squatting at a side angle in a way that you can see what I said she is doing with her vagina. The camera angle makes that possible. Her right hand holds the penis and pulls it more into her mouth. Same exact identical face with exact same facial features. her eyes are open and rolled back in pleasure. Remove the man from the photo and keep only the shaft of the penis coming from the right edge of the photo.",
        "Remove Her Clothes": "no clothing, naked, perfectly accurate, everything else is the exact same",
        "Full Body Nudity": "no clothing, naked, full body, perfectly accurate, same exact identical face with all the same exact idential features, everything else is the exact same",        
        "Secret Silent Sex": "close up, same exact face, same clothes but breasts fully out with hard nipples and vagina fully showing as the clothes just barely don't cover it. she is leaning very far forwards very close to the camera, full breasts visible and perfect, sitting on a very small penis fully penetrating all the way to the farthest depths of inside her vagina, coming up from an unseen man with his legs stretched underneath her, everything else the same, vulva fully  parted from the penis, she also firmly presses her right hand's index finger up to the center of her lips as if to say shh  but with her mouth wide open, tongue partially showing inside of it with her lips making a small partially closed O shape.\neverything else stays the same!",
        "Rear View Anal": "She is turned around  facing away from the camera, squatting and bent over a chair she holds that is far away from her but in front of her. He clothing is still the same and each part is the same but her breasts are visible outside and the bottom of her clothing is bunched up above her waist. She uses the mans penis that is in her ass to support her. Her feet are touching the ground. her butt is higher in the air so that her ass hole is penetrated by the penis. Same exact identical face as the original woman in the original photo. Her head is turned almost completely around naturally and her eyes are seen looking directly into the camera with her mouth wide open and her lips making an O shape. She is pushing her body backwards into man's penis from the bottom of the photo to get the penis further than it already is inside her ass hole. the ass is higher than the vagina and the vagina cannot be seen. her butt is angled upwards so her vagina is close up facing right at me. Make the ass hole itself higher than normal where her tailbone is instead",
        "Spread On Bed Pose": "no clothes, laying on her girly bed in a just slightly dark room, same exact identical face with all the same exact identical features. her legs are much thinner skinnier and shorter and spread apart to fully show her tiny tight beautiful vagina, identical face and expression, vagina pointed at me. everything else the same!",
        "Come Get It Pose": "no clothes, laying on her girly bed in a slightly dimly lit room, her legs are much thinner skinnier and shorter and spread apart to fully show her tiny tight beautiful vagina, exact same identical face and all the exact same facial features, vagina pointed at me, her left hand's index finger is hovering away from her vagina and she is pointing it directly at her vagina hole while she also holds her right hand's index finger on her mouth saying shh behind a partial kissy face. but still looking right into my eyes  extremely sexually suggestively and excitingly and invitingly, slightly smiling looking excited to finally have me all to herself. close up of vagina. full breasts are visible with hardened nipples. everything else stays the same!",
        "Secret Silent Room Sex": "close up, same exact identical face with the exact same identical features, on a couch, no clothes, she is leaning very far forwards very close to the camera, full breasts visible and perfect, sitting on a very small penis fully penetrating all the way to the farthest depths of inside her vagina, coming up from an unseen man with his legs stretched underneath her, everything else the same, vulva fully  parted from the penis, she also holds her right hand's index finger on her mouth saying shh but with a sexy secretive sneaky look   extremely sexually but distracted by the pleasure, mouth slightly open with lips making an o shape from the immense pleasure, dark bedroom background, everything else stays the same!",
        "Front View Couch Sex": "front view, close up of her face, she is on a very small  penis that cannot be fully seen because it is already fully all the way inside her vagina, coming up from a man that cannot be seen on the bottom of the photo underneath her, same exact face, fully naked, on a couch, full perky breasts visible and perfect,  same exact identical face and person in general but her mouth is open with lips in a closed small surprised o shape with her eyes looking at me but slightly to the top right, dark bedroom background, top part of her sexy cocktail dress is pulled down to be underneath her visible breasts and the bottom part of her dress is pulled upwards still around her legs but not covering her vagina, Exact same identical face with all the exact same identical features. everything else the same.",
        "Rear View Couch Sex": "she is squatting all the way down at the bottom center of the photo with her vagina fully penetrated by a short little penis also coming from all the way at the bottom center of the photo. \nback view, fully naked on a couch with a slight dark bedroom background. the man's legs are seen completely closed together but out straight between her legs. the entire penis is hidden inside of her vagina. she wears a sexy cocktail dress that doesn't cover her vagina. her breasts are also outside of the dress with hardened nipple showing. Her legs are much thinner and petite. both sides of her fat  vulva are noticeably spread apart from the penis and her head is turned almost completely around naturally as far as she can to look directly into my eyes, her mouth is open with lips in a closed small surprised o shape. her butt is pointing up at the camera so you can see the intimate details of her full penetrated vagina. Exact same identical face with all the exact same identical features. she is also reaching behind her back and using only  her little hand to grab and spreading more open her right butt cheek so the penis is seen even deeper inside of her. only the penis of the man is visible. everything else is the exact same!"
    }
    
    # More prompts dictionary
    more_prompts_dict = {
        "Balcony Risky Sex": "On a balcony at night, leaning over the railing, small penis inside her vagina from behind coming from an unseen man beyond the edge of the photo, dress bunched at waist, wind blowing, one leg lifted, vagina fully visible and dripping. Same exact identical face with all the same exact identical features turned back with risky excited smile and open mouth. Everything else the exact same!",
        "Bathroom Sink Sex": "Sitting on the bathroom sink facing the camera, legs spread wide on the edges, small penis fully penetrating her vagina coming from an unseen man beyond the edge of the photo, water running in background, both hands pulling her breasts up and together while vagina is  slightly naturally stretched visibly. Same exact identical face with all the same exact identical features moaning with pure pleasure looking straight at viewer. Everything else stays the exact same!",
        "Bed Arching Back Orgasm Pose": "No clothes, lying on her girly bed with back arched high, legs spread maximally and pulled back, small penis fully embedded in her vagina coming from an unseen man beyond the edge of the photo from below with the man's legs under her, her hands pinching her hard nipples. Same exact identical face with all the same exact identical features showing maximum ecstasy with eyes rolled back and tongue hanging out. Vagina is fully visible and glistening. Everything else stays the same!",
        "Bed Doggy Style Front Turned": "On all fours on the bed facing away but head fully turned back to camera naturally, small penis penetrating her vagina deeply from behind coming from an unseen man beyond the edge of the photo, one hand reaching under to spread her vulva lips around the shaft showing insertion. Same exact identical face with all the same exact identical features with eyes locked on viewer in dreamy loving pleasure. Her butt cheeks are spread naturally. Everything else the exact same!",
        "Bent Over Table Anal-Vaginal Tease": "Bent over a table in the same room, clothing bunched at waist, small penis deep in her ass coming from an unseen man beyond the edge of the photo but legs spread so her vagina is prominently visible and dripping below the penetration, she reaches back with both hands to spread her ass cheeks and vulva simultaneously. Same exact identical face with all the same exact identical features turned around to the camera with wide O mouth in overwhelming pleasure. Everything else the exact same!",
        "Car Backseat Ride": "In the backseat of a car, straddling facing camera, small penis buried inside her vagina coming from an unseen man beyond the edge of the photo, windows slightly fogged, dress pulled down and up, hands on the headrest behind her pushing her breasts forward. Same exact identical face with all the same exact identical features in maximum arousal looking at viewer. Everything else the exact same!",
        "Chair Ride Facing Camera": "Sitting on a chair facing the camera, small penis fully inside her vagina coming from an unseen man beyond the edge of the photo, legs draped over the chair arms spreading her extremely wide, both hands pulling her vulva lips apart showing the penetration and her insides. Breasts fully exposed with hard nipples. Same exact identical face with all the same exact identical features staring directly at the viewer with raw sexual hunger. Everything else stays the exact same!",
        "Close Up Face With Vaginal Spread And Penis Tease": "Extreme close up on her face and upper body but vagina visible in lower frame, she holds the small penis coming from an unseen man beyond the edge of the photo with one hand pressing it against her clitoris while the other hand spreads her vulva, mouth open tongue out licking her lips suggestively. Same exact identical face with all the same exact identical features but with maximum seductive arousal. Everything else stays the same!",
        "Couch Lap Dance Penetration": "On the couch, she straddles facing the camera in lap position with small penis fully inside her vagina coming from an unseen man beyond the edge of the photo, dress pulled up and down exposing breasts and vagina, hands on her own breasts squeezing while bouncing slightly implied. Same exact identical face with all the same exact identical features, mouth open in surprised pleasure O shape looking slightly up at camera. Everything else the exact same!",
        "Intense Missionary Leg Lift": "No clothes, lying on her back on the girly bed with both legs pulled up high and spread extremely wide, small penis buried to the hilt inside her vagina coming from an unseen man beyond the edge of the photo from below with the unseen man's legs stretched underneath. Her hands grip her own thighs pulling them apart even more so the slightly naturally stretched vulva and deep penetration are perfectly visible. Same exact identical face with all the same exact identical features showing pure overwhelming ecstasy, eyes half-closed in bliss, mouth wide open in a perfect moaning O shape. Everything else stays the exact same!",
        "Kitchen Counter Penetration": "Bent over the kitchen counter, legs spread wide, small penis deep inside her vagina from behind coming from an unseen man beyond the edge of the photo, dress pulled up, breasts hanging and pressed against the counter, one hand reaching back spreading herself. Same exact identical face with all the same exact identical features turned back with an expression of being fucked senseless in ecstasy. Everything else the exact same!",
        "Kneeling Frontal Invitation With Fingers": "Fully naked, kneeling on the floor close to camera with legs spread, using both index fingers to hook and slightly naturally pull her vagina lips apart extremely wide showing the deepest insides while a small penis tip coming from an unseen man beyond the edge of the photo presses against the entrance from below. Same exact identical face with all the same exact identical features smiling extremely sexually suggestively with pure invitation in her eyes. Everything else stays the same!",
        "Lotus Position Deep Intimate": "Fully naked in lotus position on the bed facing the camera, small penis completely inside her vagina coming from an unseen man beyond the edge of the photo, legs wrapped around the unseen man but spread enough for full view, her hands on his shoulders (unseen) while she grinds. Same exact identical face with all the same exact identical features showing intense loving pleasure, mouth open, eyes dreamy. Everything else the exact same!",
        "Mirror Reflection Tease": "Standing in front of a large mirror, bent slightly forward, small penis entering her vagina from behind coming from an unseen man beyond the edge of the photo while she watches herself in the mirror, one hand spreading her vulva, the other pinching a nipple. Same exact identical face with all the same exact identical features reflected in the mirror staring back at the viewer with maximum seductive arousal. Everything else stays the exact same!",
        "Overhead Squat Ride Ecstasy": "From an overhead camera angle, she squats deeply onto the small penis fully inside her vagina coming from an unseen man beyond the edge of the photo, vulva slightly naturally stretched wide and parted, her hands reaching down to rub her clitoris around the shaft. Same exact identical face with all the same exact identical features turned upwards to the camera with an expression of being lost in intense sexual fantasy and love. Everything else the exact same!",
        "Prone Bone Close Up": "Lying flat on her stomach on the bed, ass slightly raised, small penis deep inside her vagina from behind coming from an unseen man beyond the edge of the photo, one hand reaching back to slightly naturally spread her ass and vulva so the slightly naturally stretched hole and penetration are clearly visible. Same exact identical face with all the same exact identical features turned to the side looking back at the camera with blissful orgasm face. Everything else the exact same!",
        "Reverse Cowgirl Deep Squat": "Fully naked, squatting with her back to the camera but head turned almost 180 degrees naturally to stare straight into the viewer's eyes, small penis completely swallowed by her vagina coming from an unseen man beyond the edge of the photo, vulva lips gripping the base tightly. One hand reaches back spreading her ass cheek while the other rubs her clitoris around the shaft. Same exact identical face with all the same exact identical features lost in maximum sexual pleasure, tongue slightly out. Everything else the exact same!",
        "Shower Wet Sex": "Standing in the shower under running water, one leg wrapped around the unseen man, small penis deep inside her vagina coming from an unseen man beyond the edge of the photo, water cascading over her body, one hand spreading her vulva around the shaft. Same exact identical face with all the same exact identical features showing wet, slippery, maximum sexual ecstasy. Everything else stays the exact same!",
        "Sideways Spooning Vaginal Sex": "She lies on her side in the same room, one leg lifted high allowing full view of her vagina being penetrated by the small penis coming from the side edge of the photo attached to unseen man, her top hand slightly naturally spreading her vulva further while bottom hand grips the penis base. Same exact identical face with all the same exact identical features, mouth in perfect O shape staring invitingly at camera. Everything else the exact same!",
        "Standing Bent Over Window Sex": "She bends forward pressing her hands against the window, ass pushed out, small penis thrusting deep into her vagina from behind coming from an unseen man beyond the edge of the photo, legs spread wide so her dripping vagina is fully visible between her thighs. Clothing bunched at waist exposing breasts and vagina. Same exact identical face with all the same exact identical features turned back to the camera with desperate arousal and open moaning mouth. Everything else stays the exact same!",
        "Standing Carry Position Tease": "She is held up by unseen man (his arms and legs partially visible), legs wrapped around but spread to show vagina penetrated by small penis standing coming from an unseen man beyond the edge of the photo, her hands around his neck but since unseen focus on her, vagina fully visible slightly naturally stretched. Same exact identical face with all the same exact identical features smiling invitingly with tongue between teeth. Everything else the exact same!",
        "Wall Leaning Deep Penetration Tease": "She leans back against the wall with her legs spread wide and one knee raised high, the small penis  coming from an unseen man beyond the edge of the photo below is thrusting deep into her vagina causing visible stretching slightly naturally and her juices flowing freely. Same exact identical face with all the same exact identical features looking straight at the camera with eyes full of desperate arousal and lips parted in a silent scream of pleasure. Everything else the exact same!",
        "Wall Pressed Standing Sex": "Pressed face-first against the wall, ass pushed out, small penis buried deep in her vagina coming from an unseen man beyond the edge of the photo, one leg lifted high to the side for maximum visibility of the penetration and her dripping vulva. Same exact identical face with all the same exact identical features pressed against the wall but turned enough to lock eyes with the viewer in pure lust. Everything else stays the exact same!",
        "Yoga Mat Deep Stretch Sex": "On a yoga mat in downward dog position but modified, small penis penetrating her vagina deeply from behind coming from an unseen man beyond the edge of the photo, one leg lifted high to the side for extreme spread and visibility of penetration. Same exact identical face with all the same exact identical features looking back between her legs at the camera in overwhelming pleasure. Everything else stays the exact same!",
        "Full Penetration Attempt 1": "She is sitting up close to the camera with her legs spread. With her right hand, she is eagerly, firmly, and tightly gripping a very short 5 inch small penis with girth coming from the bottom of the image; the penis being held far off to the left of her vagina so you can see her whole vaginaon the right of it. The penis is coming from an otherwise completely unseen man whose legs are stretched out flat beneath her.  she sits directly behind the penis, holding it far off to the left of her vagina, in her hand only. You can see her whole vagina because she also uses her left hand to play with it while she holds the penis to the far left Outside of it. YOU MUST BE ABLE TO SEE HER ENTIRE VAGINA ON THE RIGHT OF THE PENIS! She stares erotically directly into the camera.  Both of her breasts are fully exposed outside of her clothing, with visibly hardened nipples. . Exact same woman, same face and features. Everything else must remain exactly the same.",
        "Full Penetration Attempt 2": "She kneels with mouth wide open, the entire short penis is completely buried deep inside her mouth with absolutely zero shaft visible, her lips pressed firmly against the base where the penis meets the body, throat bulging from the full depth penetration. No part of the penis shaft can be seen. Her hand grips only the very base. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 3": "She lies on her back, legs spread maximally, the complete short penis is fully swallowed inside her vagina with zero shaft showing, only the base visible where it enters her body, her vulva lips stretched tightly around the very base with no penis shaft exposed at all. Her hands spread her vulva to show the complete insertion. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 4": "She bends forward on all fours, the entire short penis is completely buried inside her ass with absolutely no shaft visible, only the base where it meets the body can be seen, her ass cheeks pressed against the base. She reaches back spreading herself to show the full insertion with zero shaft exposed. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 5": "Close up of her face, the short penis is fully embedded all the way down her throat, her nose pressed against the base, lips sealed around where the penis meets the body with absolutely zero shaft visible, eyes watering from the complete depth. No part of the shaft can be seen. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 6": "She squats directly down, the short penis is completely hidden inside her vagina, not a single bit of shaft showing, her vulva grips only the very base where penis meets body, she is fully impaled with the entire length buried inside. Her hands pull her lips apart showing zero shaft. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 7": "She is positioned with her ass high in the air, the complete short penis is fully inserted into her ass with zero shaft visible, only the base can be seen, her hole stretched around the very base with the entire length swallowed inside. She spreads her cheeks showing no shaft at all. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 8": "Side view, her mouth completely engulfs the short penis with the entire shaft buried inside, her lips touching the base where penis meets body, cheeks hollowed, throat visibly stretched, absolutely no shaft showing. Her hand holds only the very base. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 9": "She rides facing camera, the short penis is fully buried inside her vagina with the complete shaft hidden, zero shaft visible, her body pressed down completely, vulva stretched around only the base where it meets the body. She grinds with the entire length inside. Same exact identical face with all the same exact identical features. Everything else stays the exact same!",
        "Full Penetration Attempt 10": "She leans forward, the entire short penis is completely inserted into her ass, the full shaft buried with absolutely nothing showing except the base where it meets the body, her hole gripping the very base. She looks back while fully impaled with zero shaft exposed. Same exact identical face with all the same exact identical features. Everything else stays the exact same!"
    }
    
    # Dropdown handler
    def update_prompt_from_dropdown(choice):
        if choice:
            return prompts_dict.get(choice, "")
        return ""
    
    # More prompts dropdown handler
    def update_prompt_from_dropdown2(choice):
        if choice:
            return more_prompts_dict.get(choice, "")
        return ""
    
    # Couple prompts dictionary
    couple_prompts_dict = {
        "Merge Together": "They are both standing next to each other facing the camera. She wears a cute sundress that is loose at the bottom and he wears a stylish open button down shirt with a white t-shirt under it. Both maintain their exact same identical original faces with all features preserved.",
        "Kissing Handjob": "He is sitting on the ground facing the bottom right corner while she squats higher in the air beside him, almost hip to hip. His penis is out of his pants and she is fully gripping it tightly around the full circumference of its shaft in one hand while her other hand is behind his head as they passionately kiss on the lips. Both maintain their exact same identical original faces.",
        "Shower Sex Together": "They are both standing in the shower, she has one leg lifted near him, his penis deep inside her vagina, her hand spreading her vulva around the shaft. Zoomed out for full view. Both maintain their exact same identical original faces with all features preserved.",
        "Lift Up Vag Flash (Couple)": "She lifts the entire bottom portion of her clothing upward to just above her vagina area. All of the bottom fabric is now fully scrunched and bunched tightly in her hands, completely exposing her shaved vagina. He stands beside her. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same.",
        "Same Place Sitting Tease (Couple)": "She sits on the floor with her clothes pulled down to expose her breasts and pulled up to expose her vagina. She uses her hands to grip both sides of her vulva and fully part/spread them so her vagina insides are visible. He is positioned nearby. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same.",
        "Same Place Shh Tease (Couple)": "Her left hand's index finger hovers away from and points directly at her vagina hole while her right hand's index finger is on her mouth saying shh, smiling sexually suggestively. He is positioned nearby. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Same Place Shh Tease 2 (Couple)": "Her left hand touches her clitoris without covering her vagina while her right hand's index finger is on her mouth saying shh, tongue pressed between her teeth sexually. He is positioned nearby. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same!",
        "1 Leg Up Inviting Tease (Couple)": "She holds one leg far up in the air to fully show her vagina facing the camera. He is positioned nearby. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Same Place Squatting Frontal (Couple)": "She is being fully penetrated by his small but wide penis with his legs stretched under her. Her mouth is wide open with lips making an O shape, looking deeply in love. She is very close to the camera sitting on his penis that is fully inside her vagina, showing how it parts her vulva. She touches her clitoris with one finger while her other hand grabs his penis. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same!",
        "Dick Holding Pleasure Tease (Couple)": "She is sitting up close to the camera with legs spread. With her right hand, she eagerly, firmly grips his very short 5 inch penis with girth, holding it far off to the left of her vagina so you can see her whole vagina on the right of it. His legs are stretched out flat beneath her. She uses her left hand to play with her vagina while holding his penis to the far left. YOU MUST BE ABLE TO SEE HER ENTIRE VAGINA ON THE RIGHT OF THE PENIS! She stares erotically at the camera. Both her breasts are fully exposed with hardened nipples. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Dick Holding Almost BJ (Couple)": "She leans forward with mouth wide open and lips in an O shape, her mouth closer to his penis but not touching it. Her tongue is visible behind the penis pressed against it. She is still firmly grasping his penis. Close up of her face. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same.",
        "Same Place Partial BJ (Couple)": "She is on her stomach bending forward so his penis is all the way in her mouth, deep in her mouth. Her butt is visibly behind her head with smooth skin and arched cheeks. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same.",
        "Side View Oral (Couple)": "She wears the same clothing. Her left hand sexually touches her vagina between her fully spread legs with fingers inside. His penis is all the way deep inside her mouth in her throat with a visible bulge. Her lips are tightly closed surrounding the bottom of his penis. She squats at a side angle. Her right hand holds his penis and pulls it more into her mouth. Both maintain their exact same identical original faces with all features preserved, her eyes open and rolled back in pleasure.",
        "Remove Her Clothes (Couple)": "She has no clothing, naked, perfectly accurate. He remains as is. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Full Body Nudity (Couple)": "Both are naked, full body, perfectly accurate. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same.",
        "Secret Silent Sex (Couple)": "Close up, she wears the same clothes but breasts are fully out with hard nipples and vagina fully showing. She leans very far forward very close to the camera, sitting on his very small penis fully penetrating to the farthest depths inside her vagina, with his legs stretched underneath her. Vulva fully parted from his penis. She firmly presses her right index finger to her lips saying shh with mouth wide open, tongue partially showing, lips making a small O shape. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Rear View Anal (Couple)": "She is turned around facing away from the camera, squatting and bent over a chair in front of her. Her clothing is bunched up above her waist with breasts visible. She uses his penis in her ass for support. Her feet touch the ground, butt higher in the air with her ass hole penetrated by his penis. Her head is turned almost completely around naturally, eyes looking at camera with mouth wide open, lips making an O shape. She pushes her body backwards into his penis. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Spread On Bed Pose (Couple)": "She has no clothes, laying on a girly bed in a slightly dark room. Her legs are spread apart to fully show her vagina pointed at the camera. He is positioned nearby. Both maintain their exact same identical original faces with all features preserved. Everything else stays the same!",
        "Come Get It Pose (Couple)": "She has no clothes, laying on a girly bed in a dimly lit room. Her legs are spread apart to fully show her vagina pointed at the camera. Her left index finger hovers and points at her vagina hole while her right index finger is on her mouth saying shh behind a kissy face, looking extremely sexually suggestive and inviting. Her breasts are visible with hardened nipples. He is positioned nearby. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Secret Silent Room Sex (Couple)": "Close up, she is on a couch with no clothes, leaning very far forward very close to the camera with full breasts visible, sitting on his very small penis fully penetrating to the farthest depths inside her vagina, with his legs stretched underneath her. Vulva fully parted from his penis. She holds her right index finger on her mouth saying shh with a sexy secretive sneaky look, mouth slightly open with lips making an O shape from pleasure. Dark bedroom background. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Front View Couch Sex (Couple)": "Front view, close up of her face. She is on his very small penis that is fully all the way inside her vagina, with him underneath her on the couch. Fully naked with full perky breasts visible. Her mouth is open with lips in a closed small surprised O shape, eyes looking slightly to the top right. Dark bedroom background. Her dress is pulled down underneath her breasts and pulled up around her legs. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Rear View Couch Sex (Couple)": "She is squatting all the way down with her vagina fully penetrated by his short little penis. Back view, fully naked on a couch with slight dark bedroom background. His legs are seen completely closed together but out straight between her legs. The entire penis is hidden inside her vagina. She wears a sexy cocktail dress that doesn't cover her vagina, breasts outside with hardened nipples. Both sides of her vulva are noticeably spread apart from his penis. Her head is turned almost completely around naturally, mouth open with lips in a closed small surprised O shape. Her butt points up at the camera showing intimate details of her full penetrated vagina. She reaches behind her back using her hand to grab and spread her right butt cheek. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Balcony Risky Sex (Couple)": "On a balcony at night, she leans over the railing with his small penis inside her vagina from behind, dress bunched at waist, wind blowing, one leg lifted, vagina fully visible and dripping. Her head is turned back with risky excited smile and open mouth. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Bathroom Sink Sex (Couple)": "She sits on the bathroom sink facing the camera, legs spread wide on the edges, his small penis fully penetrating her vagina, water running in background. Both her hands pull her breasts up and together while her vagina is slightly naturally stretched visibly. She moans with pure pleasure looking straight at viewer. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Bed Arching Back Orgasm Pose (Couple)": "No clothes, she lies on a girly bed with back arched high, legs spread maximally and pulled back, his small penis fully embedded in her vagina from below with his legs under her. Her hands pinch her hard nipples showing maximum ecstasy with eyes rolled back and tongue hanging out. Vagina is fully visible and glistening. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Bed Doggy Style Front Turned (Couple)": "She is on all fours on the bed facing away but head fully turned back to camera naturally, his small penis penetrating her vagina deeply from behind. One hand reaches under to spread her vulva lips around the shaft showing insertion. Her eyes are locked on viewer in dreamy loving pleasure. Her butt cheeks are spread naturally. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Bent Over Table Anal-Vaginal Tease (Couple)": "She is bent over a table, clothing bunched at waist, his small penis deep in her ass but legs spread so her vagina is prominently visible and dripping below the penetration. She reaches back with both hands to spread her ass cheeks and vulva simultaneously. Her head is turned around to the camera with wide O mouth in overwhelming pleasure. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Car Backseat Ride (Couple)": "In the backseat of a car, she straddles facing camera with his small penis buried inside her vagina, windows slightly fogged, dress pulled down and up. Her hands are on the headrest behind her pushing her breasts forward in maximum arousal looking at viewer. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Chair Ride Facing Camera (Couple)": "She sits on a chair facing the camera with his small penis fully inside her vagina, legs draped over the chair arms spreading her extremely wide. Both her hands pull her vulva lips apart showing the penetration and her insides. Breasts fully exposed with hard nipples. She stares directly at the viewer with raw sexual hunger. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Close Up Face With Vaginal Spread And Penis Tease (Couple)": "Extreme close up on her face and upper body but vagina visible in lower frame. She holds his small penis with one hand pressing it against her clitoris while the other hand spreads her vulva, mouth open tongue out licking her lips suggestively with maximum seductive arousal. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Couch Lap Dance Penetration (Couple)": "On the couch, she straddles facing the camera in lap position with his small penis fully inside her vagina, dress pulled up and down exposing breasts and vagina. Her hands are on her own breasts squeezing while bouncing slightly implied. Her mouth is open in surprised pleasure O shape looking slightly up at camera. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Intense Missionary Leg Lift (Couple)": "No clothes, she lies on her back on a girly bed with both legs pulled up high and spread extremely wide, his small penis buried to the hilt inside her vagina from below with his legs stretched underneath. Her hands grip her own thighs pulling them apart even more so the slightly naturally stretched vulva and deep penetration are perfectly visible. She shows pure overwhelming ecstasy, eyes half-closed in bliss, mouth wide open in a perfect moaning O shape. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Kitchen Counter Penetration (Couple)": "She is bent over the kitchen counter, legs spread wide, his small penis deep inside her vagina from behind, dress pulled up, breasts hanging and pressed against the counter. One hand reaches back spreading herself. Her head is turned back with an expression of being fucked senseless in ecstasy. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Kneeling Frontal Invitation With Fingers (Couple)": "Fully naked, she kneels on the floor close to camera with legs spread, using both index fingers to hook and slightly naturally pull her vagina lips apart extremely wide showing the deepest insides while his small penis tip presses against the entrance from below. She smiles extremely sexually suggestively with pure invitation in her eyes. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Lotus Position Deep Intimate (Couple)": "Fully naked in lotus position on the bed facing the camera, his small penis completely inside her vagina, her legs wrapped around him but spread enough for full view. Her hands are on his shoulders while she grinds, showing intense loving pleasure, mouth open, eyes dreamy. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Mirror Reflection Tease (Couple)": "She stands in front of a large mirror, bent slightly forward, his small penis entering her vagina from behind while she watches herself in the mirror. One hand spreads her vulva, the other pinches a nipple. Her face is reflected in the mirror staring back at the viewer with maximum seductive arousal. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Overhead Squat Ride Ecstasy (Couple)": "From an overhead camera angle, she squats deeply onto his small penis fully inside her vagina, vulva slightly naturally stretched wide and parted. Her hands reach down to rub her clitoris around the shaft. Her face is turned upwards to the camera with an expression of being lost in intense sexual fantasy and love. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Prone Bone Close Up (Couple)": "She lies flat on her stomach on the bed, ass slightly raised, his small penis deep inside her vagina from behind. One hand reaches back to slightly naturally spread her ass and vulva so the slightly naturally stretched hole and penetration are clearly visible. Her head is turned to the side looking back at the camera with blissful orgasm face. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Reverse Cowgirl Deep Squat (Couple)": "Fully naked, she squats with her back to the camera but head turned almost 180 degrees naturally to stare straight into the viewer's eyes, his small penis completely swallowed by her vagina, vulva lips gripping the base tightly. One hand reaches back spreading her ass cheek while the other rubs her clitoris around the shaft. She is lost in maximum sexual pleasure, tongue slightly out. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Shower Wet Sex (Couple)": "They stand in the shower under running water, her one leg wrapped around him, his small penis deep inside her vagina, water cascading over their bodies. Her hand spreads her vulva around the shaft showing wet, slippery, maximum sexual ecstasy. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Sideways Spooning Vaginal Sex (Couple)": "She lies on her side, one leg lifted high allowing full view of her vagina being penetrated by his small penis. Her top hand slightly naturally spreads her vulva further while bottom hand grips his penis base. Her mouth is in perfect O shape staring invitingly at camera. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Standing Bent Over Window Sex (Couple)": "She bends forward pressing her hands against the window, ass pushed out, his small penis thrusting deep into her vagina from behind, legs spread wide so her dripping vagina is fully visible between her thighs. Clothing bunched at waist exposing breasts and vagina. Her head is turned back to the camera with desperate arousal and open moaning mouth. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Standing Carry Position Tease (Couple)": "She is held up by him (his arms and legs partially visible), her legs wrapped around but spread to show her vagina penetrated by his small penis standing. Her hands are around his neck, vagina fully visible slightly naturally stretched. She smiles invitingly with tongue between teeth. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Wall Leaning Deep Penetration Tease (Couple)": "She leans back against the wall with her legs spread wide and one knee raised high, his small penis thrusting deep into her vagina causing visible stretching slightly naturally and her juices flowing freely. She looks straight at the camera with eyes full of desperate arousal and lips parted in a silent scream of pleasure. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Wall Pressed Standing Sex (Couple)": "She is pressed face-first against the wall, ass pushed out, his small penis buried deep in her vagina, one leg lifted high to the side for maximum visibility of the penetration and her dripping vulva. Her face is pressed against the wall but turned enough to lock eyes with the viewer in pure lust. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Yoga Mat Deep Stretch Sex (Couple)": "On a yoga mat in downward dog position but modified, his small penis penetrating her vagina deeply from behind, one leg lifted high to the side for extreme spread and visibility of penetration. She looks back between her legs at the camera in overwhelming pleasure. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 1 (Couple)": "She is sitting up close to the camera with legs spread. With her right hand, she eagerly, firmly grips his very short 5 inch penis with girth, holding it far off to the left of her vagina so you can see her whole vagina on the right of it. His legs are stretched out flat beneath her. She uses her left hand to play with her vagina while holding his penis to the far left. YOU MUST BE ABLE TO SEE HER ENTIRE VAGINA ON THE RIGHT OF THE PENIS! She stares erotically at the camera. Both her breasts are fully exposed with hardened nipples. Both maintain their exact same identical original faces. Everything else stays the same.",
        "Full Penetration Attempt 2 (Couple)": "She kneels with mouth wide open, his entire short penis completely buried deep inside her mouth with absolutely zero shaft visible, her lips pressed firmly against the base where his penis meets the body, throat bulging from the full depth penetration. No part of the penis shaft can be seen. Her hand grips only the very base. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 3 (Couple)": "She lies on her back, legs spread maximally, his complete short penis fully swallowed inside her vagina with zero shaft showing, only the base visible where it enters her body. Her vulva lips are stretched tightly around the very base with no penis shaft exposed at all. Her hands spread her vulva to show the complete insertion. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 4 (Couple)": "She bends forward on all fours, his entire short penis completely buried inside her ass with absolutely no shaft visible, only the base where it meets the body can be seen, her ass cheeks pressed against the base. She reaches back spreading herself to show the full insertion with zero shaft exposed. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 5 (Couple)": "Close up of her face, his short penis fully embedded all the way down her throat, her nose pressed against the base, lips sealed around where his penis meets the body with absolutely zero shaft visible, eyes watering from the complete depth. No part of the shaft can be seen. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 6 (Couple)": "She squats directly down, his short penis completely hidden inside her vagina, not a single bit of shaft showing. Her vulva grips only the very base where penis meets body, she is fully impaled with the entire length buried inside. Her hands pull her lips apart showing zero shaft. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 7 (Couple)": "She is positioned with her ass high in the air, his complete short penis fully inserted into her ass with zero shaft visible, only the base can be seen, her hole stretched around the very base with the entire length swallowed inside. She spreads her cheeks showing no shaft at all. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 8 (Couple)": "Side view, her mouth completely engulfs his short penis with the entire shaft buried inside, her lips touching the base where penis meets body, cheeks hollowed, throat visibly stretched, absolutely no shaft showing. Her hand holds only the very base. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 9 (Couple)": "She rides facing camera, his short penis fully buried inside her vagina with the complete shaft hidden, zero shaft visible, her body pressed down completely, vulva stretched around only the base where it meets the body. She grinds with the entire length inside. Both maintain their exact same identical original faces. Everything else stays the same!",
        "Full Penetration Attempt 10 (Couple)": "She leans forward, his entire short penis completely inserted into her ass, the full shaft buried with absolutely nothing showing except the base where it meets the body, her hole gripping the very base. She looks back while fully impaled with zero shaft exposed. Both maintain their exact same identical original faces. Everything else stays the same!"
    }
    
    # Couple prompts dropdown handler
    def update_prompt_from_dropdown3(choice):
        if choice:
            return couple_prompts_dict.get(choice, "")
        return ""
    
    # Group prompts dictionary (1 man + 2-3 women)
    group_prompts_dict = {
        "Merge Together 1": "make them both standing next to eachother facing the camera. She is to the right of him. Her face and body are perfectly exactly the same. she wears a cute sundress that is loose at the bottom  and he wears a stylish open button down shirt with a white t shift under it. his penis is exactly the same.",
        "Merge Together 2": "They are both standing next to each other facing the camera. She is on his right. She wears the same exact clothing (or none if she isn't wearing any) that is loose at the bottom and he wears a stylish open button down shirt with a white t-shirt under it. Both maintain their exact same identical original faces with all features preserved. his penis is every way the same, unchanged.",
        "Naked Together": "Add the woman from the first photo to be standing on the right of the man in the second photo, ensuring the woman is fully naked wearing no clothes at all. She has the exact same face, facial expression, etc as her original photo. Keep his penis exactly the way it is without altering it. Both maintain their exact same identical original faces with all features preserved. Her face and facial expression and angle of it must be exactly identical.es preserved.",
        "Near Dick 1": "fully zoomed out so she is fully in view. the woman from the first picture looking exactly 100% the same with same exact face and facial features and expression are exactly the same. She is completely moved from the 1st photo to be in the 2nd photo so that she is sitting behind that man's separate penis. she has no pants on and no underwear on and nothing covering her vagina. she has no clothing around her waste or under it. Her legs are spread and she is sitting in a way so that her entire vagina is fully visible. The  man's penis is exactly the same. the man is still wearing the same clothing. Remove all other things from the first picture and only use the background/environment of the second picture.",
        "Near Dick 2": "She has no clothing around her waste or under it and she is squatting near the penis from photo 2. we can see her full vagina. you can still see her face. she is looking at the camera the exact same way as her original photo. The penis from photo 2 is in her hand. she has the exact very young looking same face and everything else about her face and head and hair is the exact same. everything else the same from photo 2.",
        "Near Dick 3": "zoomed out to show her full body. She has no clothing on and she is now only in photo 2 near the man's penis. you can still see her face. she is looking at the camera the exact same way as her original photo. she holds the man's penis from photo 2 is in her hand as it is attached to the man in the same exact way. she has the exact very young looking same face and everything else about her face and head and hair is the exact same. everything else the same from photo 2.",
        "Near Dick 4": "zoomed out to show her full body. She has no clothing on from her waist down and she is now perfectly embedded only in photo 2 near the man's penis. His penis looks exactly the same from the original photo. her legs are open showing her vagina. you can still see her face. she is looking at the camera the exact same way as her original photo. she holds the man's penis from photo 2 is in her hand as it is attached to the man in the same exact way. she has the exact very young looking same face and everything else about her face and head and hair is the exact same. everything else the same from photo 2.",
        "Near Dick 5": "She is sitting on the man in the second photo that has everything the exact same. she has her legs spread behind that exact same penis and she is pushing her vagina against the back of it which is wedging the penis in between her now visibly spread vulva. everything else from the second photo is exactly the same. the environment of the first photo is removed and replaced with the exact environment of the second photo. the penis is exactly the same identical penis in the same exact position etc.",
        "Shh Holding Dick": "She is now holding his penis with only one of her hands as she looks sexually horny at the camera and her other hand has it's index finder on her lips as she says shh. she is demanding sex. everything else is the exact same.",
        "Both Near Dick": "both women from the first picture looking exactly 100% the same are now only in the second photo, near the legs and penis of the man in the second photo.  the women's faces are fully preserved/unchanged as well as their clothing etc. The man's penis is exactly the same. the man is still wearing the same clothing. Remove all other things from the first photo and only use the background/environment of the second photo. they are in the second photo only. do not include  anything from the first photo except the 2 women.",
        "Both Holding Dick 1": "both women are how holding his penis in their hands and looking at the camera. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Both Holding Dick 2": "both women from the first picture looking exactly 100% the same are now only in the second photo, near the legs and penis of the man in the second photo.  the women's faces are fully preserved/unchanged as well as their clothing etc. The man's penis is exactly the same. the man is still wearing the same clothing. Remove all other things from the first photo and only use the background/environment of the second photo. they are in the second photo only. do not include  anything from the first photo except the 2 women.",
        "Single BJ": "the woman is now bent forward down fully sucking his entire shaft of his penis now hidden inside of her mouth with her lips sealed around the circumference of the lowest part of the shaft. She does all of that while sexually touching her own vagina. She maintains her exact same identical original face with all the exact same identical features. Everything else stays the exact same!",
        "Single Anal": "the woman is now turned around and on his penis that is now fully penetrated deep inside of her ass. The entire shaft of his penis now hidden inside of her ass which has the entire penis all the way to the lowest part of the shaft. She is turned so you can see her exact same original face. She does this while sexually touching her own vagina. The woman maintains her exact same identical original face with all the exact same identical features. Everything else stays the exact same!",
        "Left 1 BJ": "the woman on the left   is bent forward down fully sucking his entire shaft of his penis now hidden inside of her mouth with her lips sealed around the circumference of the lowest part of the shaft. . the other woman  is watching while sexually touching her own vagina. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Left 1 Anal": "the woman on the left is now turned around and on his penis that is now fully penetrated deep inside of her ass.  the entire shaft of his penis now hidden inside of her ass which has the entire penis all the way to the lowest part of the shaft. she is turned so you can see her exact same original face. the other woman  is watching while sexually touching her own vagina. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Left 1 Vag On Knee": "the woman on the left is now sitting on his right knee with her vagina showing. the other woman is on his left knee with her vagina showing. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Fucked after 1": "the woman on the left   is now turned around and on his penis that is now fully penetrated deep inside of her ass.  the entire shaft of his penis now hidden inside of her ass which has the entire penis all the way to the lowest part of the shaft. she is turned so you can see her exact same original face. the other woman  is watching while sexually touching her own vagina. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Fucked after 2": "the woman on the right   is now turned around and on his penis that is now fully penetrated deep inside of her ass.  the entire shaft of his penis now hidden inside of her ass which has the entire penis all the way to the lowest part of the shaft. she is turned so you can see her exact same original face. the other woman  is watching while sexually touching her own vagina. Both women maintain their exact same identical original faces with all the exact same identical features. Everything else stays the exact same!",
        "Penetration 1": "the penis is now only fully penetrated inside her vagina. she is still looking at the camera but with her mouth open and her lips in the shape of an o. she no longer holds a penis because it is now sticking into her vagina. the penis is the same in every way.",
        "Penetration 2": "the same exact original  penis is now only fully penetrated inside her vagina. she is still looking at the camera but with her mouth open and her lips in the shape of an o. she no longer holds a penis because the same one is now sticking into her vagina. the penis is the same in every way. the mans legs fully outstretched and together under her. the penis is exactly the same.",
        "Penetration 3": "the penis is now only fully penetrated all the way inside her vagina as she has 1 leg in the air and looks dreamily into his eyes. the full shaft of the penis now inserted and hidden inside of her. with her mouth open and her lips in the shape of an o. her face is exactly the same",
        "Shower Sex Together": "They are both standing in the shower, she has one leg lifted near the man in the photo, his penis deep inside her vagina, one hand spreading her vulva around the shaft.  zoomed out for full view",
        "Kissing Handjob": "the man is now sitting down on the ground  facing the bottom right corner of the photo  and the woman is squatting higher in the air beside him, almost hip to hip. the man now has his penis out of his pants and the woman's is fully gripping it tightly around the full circumference of it's shaft in one of her hands while her other hand is behind his head while she passionately kisses him on the lips.",
        "BJ": "The woman from the second photo eagerly has her hand that is the closest to him holding his penis with her fingers wrapped fully around the shaft. The position of the rest of her body remains unchanged. The man from the first photo remains completely unchanged. She remains in the exact same position next to him. You can fully see her hand's fingers wrapped completely around it. the penis stays exactly the way it is in the original photo. Keep his penis exactly the way it is without altering it. Both the man and woman preserve their exact same faces and facial features.",
        "BJ 2": "the woman from the first photo squats down next to the man from the second photo so that her face is at the man's penis which is now all the way deep inside her mouth with the full shaft of it hidden inside of her mouth which is closed around it.  Her right hand holds his penis and pulls it more into her mouth. Her lips are tightly closed surrounding the bottom of his penis.  Her left hand also sexually touches her vagina between her fully spread legs with fingers inside. the man's head is hidden. is moved first so that all of these things are possible.  her legs are spread and you can fully see her vagina in the photo and also see her fingering her vagina. Keep her exact same face and facial features preserved, her eyes open and rolled back in pleasure.",
        "Bed BJ 1": "the man from the second photo and the woman from the first photo are on a bed in a girly room. the man is has his eyes closed. her exact face is perfectly preserved. she has his full penis inside of her mouth. she has a sexually aroused smiling and sneaky look on her face because she is doing this without his permission while he is sleeping. The full shaft of the penis is buried deep inside of her mouth so it is hidden from the view of the camera. zoomed out view. his penis is exactly where it should be on his body.",
        "Bed BJ 2": "the man from the second photo and the woman from the first photo are on a bed in a girly room. her exact face is perfectly preserved. she has his full penis inside of her mouth as she has a freaky sexual deviant look on her face. she looks at the penis as it is all the way inside so that the full shaft of it is buried deep inside and hidden from view of the camera.",
        "Bed Sex 1": "the man from the second photo and the woman from the first photo are on a bed in a girly room. they are having passionate sex. You can see his penis all the way inside of her vagina and her face with wide open mouth and lips making an o shape. both their faces perfectly preserved. they both look down at the penis going into her vagina all the way inside so that the full shaft of it is buried deep inside and hidden from view of the camera.",
        "Bed Sex2": "the man from the second photo and the woman from the first photo are on a bed in a girly room.  You can see the man's penis is all the way inside of the woman's vagina. Her face is perfectly preserved. Her eyes look down at the penis going into her vagina. The man's penis is all the way inside of her vagina so that the full shaft of it is buried deep inside and hidden from view of the camera.",
        "Bed Sex 3": "the male in the second photo has his penis fully penetrating the female from the first photo in her vagina, on a bed. the female's face is fully preserved as the exact same with all the same exact features.",
        "Bed Sex 4": "the woman from the first photo is in the exact same pose/position. the man in the second photo is laying on her same bed with his legs outwads between hers and his penis penetrating her vagina. everything about them is the same",
        "Side Fuck": "She lies on her side, one leg lifted high allowing full view of her vagina being penetrated by his entire penis. Her mouth is in perfect O shape.  Both maintain their exact same identical original faces. they both are looking down at the penetration. Everything else stays the same!",
        "Kissing Sex": "she drops her butt and vagina more onto his penis so that his penis is fully inserted deeply into her vagina as she rides his penis (they are having sex) as they open-mouth French kiss passionately"
    }
    
    # Group prompts dropdown handler
    def update_prompt_from_dropdown4(choice):
        if choice:
            return group_prompts_dict.get(choice, "")
        return ""
    
    preset_dropdown.change(
        fn=update_prompt_from_dropdown,
        inputs=[preset_dropdown],
        outputs=[prompt],
        scroll_to_output=False
    )
    
    preset_dropdown2.change(
        fn=update_prompt_from_dropdown2,
        inputs=[preset_dropdown2],
        outputs=[prompt],
        scroll_to_output=False
    )
    
    preset_dropdown3.change(
        fn=update_prompt_from_dropdown3,
        inputs=[preset_dropdown3],
        outputs=[prompt],
        scroll_to_output=False
    )
    
    preset_dropdown4.change(
        fn=update_prompt_from_dropdown4,
        inputs=[preset_dropdown4],
        outputs=[prompt],
        scroll_to_output=False
    )

    gr.on(
        triggers=[run_button.click, run_button_top.click, run_button_top2.click, run_button_top3.click, run_button_top4.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images,
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
            num_images_per_prompt,
        ],
        outputs=[result, seed, use_output_btn],
    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )
    
    # Starter image button handlers
    start1_btn.click(fn=lambda imgs: add_starter_image(1, imgs), inputs=[input_images], outputs=[input_images])
    start2_btn.click(fn=lambda imgs: add_starter_image(2, imgs), inputs=[input_images], outputs=[input_images])
    start3_btn.click(fn=lambda imgs: add_starter_image(3, imgs), inputs=[input_images], outputs=[input_images])
    start4_btn.click(fn=lambda imgs: add_starter_image(4, imgs), inputs=[input_images], outputs=[input_images])

    # Preset handler for negative prompts
    def update_negative_prompt(preset):
        presets = {
            "None": "",
            "Full Penetration (Vaginal)": "partial insertion, visible shaft, shallow, tip only, halfway, partially inserted, pulling out, not fully inside",
            "Full Penetration (Oral)": "shallow, tip only, not deepthroat, partially in mouth, lips at tip, halfway, visible shaft outside mouth",
            "Full Penetration (Anal)": "partial insertion, visible shaft, shallow, tip only, halfway, partially inserted, not fully inside, pulling out",
        }
        return presets.get(preset, "")
    
    # REMOVED - negative_preset dropdown no longer exists
    
    # Fullscreen toggle handler
    fullscreen_toggle.change(
        fn=None,
        inputs=[fullscreen_toggle],
        outputs=None,
        js="""
        (fullscreen) => {
            const style = document.getElementById('fullscreen-style') || document.createElement('style');
            style.id = 'fullscreen-style';
            if (fullscreen) {
                style.textContent = `
                    .gradio-container { max-width: 100vw !important; padding: 0 !important; }
                    .contain { max-width: 100% !important; }
                    .wrap { max-width: 100% !important; }
                    #component-0 { max-width: 100% !important; }
                `;
            } else {
                style.textContent = '';
            }
            if (!document.getElementById('fullscreen-style')) {
                document.head.appendChild(style);
            }
            return fullscreen;
        }
        """
    )
    
    # Keyboard toggle handler
    keyboard_toggle.change(
        fn=None,
        inputs=[keyboard_toggle],
        outputs=None,
        js="""
        (disable) => {
            setTimeout(() => {
                const style = document.getElementById('keyboard-style') || document.createElement('style');
                style.id = 'keyboard-style';
                if (disable) {
                    style.textContent = `
                        input[type="text"]:not([role="combobox"]), 
                        input[type="number"], 
                        textarea:not([role="combobox"]) { 
                            -webkit-user-select: none !important;
                            user-select: none !important;
                            pointer-events: none !important;
                        }
                    `;
                    document.querySelectorAll('input[type="text"]:not([role="combobox"]), input[type="number"], textarea:not([role="combobox"])').forEach(el => {
                        el.setAttribute('readonly', 'readonly');
                        el.setAttribute('inputmode', 'none');
                    });
                } else {
                    style.textContent = '';
                    document.querySelectorAll('input[type="text"], input[type="number"], textarea').forEach(el => {
                        el.removeAttribute('readonly');
                        el.removeAttribute('inputmode');
                    });
                }
                if (!document.getElementById('keyboard-style')) {
                    document.head.appendChild(style);
                }
            }, 100);
            return disable;
        }
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
