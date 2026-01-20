"""
Gradio Web UI for Z-Image Turbo on Kaggle.

Usage on Kaggle (notebook):

    !pip install -q "git+https://github.com/huggingface/diffusers"
    !pip install -q transformers accelerate safetensors gradio pillow

    # Optional: if the model repo uses Xet storage, this helps performance:
    # !pip install -q "huggingface_hub[hf_xet]"

    from z_image_turbo_app import load_pipeline, create_demo

    pipe = load_pipeline()
    demo = create_demo(pipe)
    demo.queue().launch()

Z-Image Turbo uses the `ZImagePipeline` class in diffusers
and currently supports text-to-image generation.
"""

import os
from typing import Optional

import torch
from diffusers import ZImagePipeline
import gradio as gr


MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model_id: str = MODEL_ID, use_fp16: bool = True) -> ZImagePipeline:
    """
    Load the Z-Image Turbo diffusion pipeline using `ZImagePipeline`.

    On Kaggle T4 (16GB), the full model in GPU memory can trigger CUDA OOM,
    so we default to CPU offload when CUDA is available.
    """
    if use_fp16 and DEVICE == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_fp16 and DEVICE == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=False)

    if DEVICE == "cuda":
        # Use accelerate-based CPU offload to fit into 16GB VRAM.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    return pipe


def text_to_image(
    pipe: ZImagePipeline,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: Optional[int],
):
    if not prompt:
        return None

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]


def _normalize_seed(raw_seed) -> Optional[int]:
    try:
        s_int = int(raw_seed)
    except (TypeError, ValueError):
        return None
    return None if s_int < 0 else s_int


def create_demo(pipe: ZImagePipeline):
    with gr.Blocks() as demo:
        gr.Markdown("# Z-Image Turbo Web UI (Kaggle)")

        with gr.Tab("Text to Image"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=3)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=3)
            with gr.Row():
                steps = gr.Slider(4, 16, value=9, step=1, label="Steps (Turbo ≈ 8 NFEs)")
                guidance = gr.Slider(0.0, 4.0, value=0.0, step=0.1, label="Guidance scale (Turbo: 0.0)")
            with gr.Row():
                height = gr.Slider(512, 1152, value=1024, step=16, label="Height")
                width = gr.Slider(512, 1152, value=1024, step=16, label="Width")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

            btn = gr.Button("Generate")
            output = gr.Image(label="Output")

            btn.click(
                fn=lambda p, np, st, g, h, w, s: text_to_image(
                    pipe, p, np, st, g, int(h), int(w), _normalize_seed(s)
                ),
                inputs=[prompt, negative_prompt, steps, guidance, height, width, seed],
                outputs=output,
            )

    return demo


if __name__ == "__main__":
    z_pipe = load_pipeline()
    ui = create_demo(z_pipe)
    ui.queue().launch(server_name="0.0.0.0", server_port=7860)
