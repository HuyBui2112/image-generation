"""
Gradio Web UI for Z-Image Turbo on Kaggle.

Usage on Kaggle (notebook):

    !pip install -q diffusers transformers accelerate safetensors gradio pillow

    # Optional: if the model repo uses Xet storage, this helps performance:
    # !pip install -q "huggingface_hub[hf_xet]"

    from z_image_turbo_app import load_pipeline, create_demo

    pipe = load_pipeline()
    demo = create_demo(pipe)
    demo.queue().launch()

Note: I cannot access Hugging Face from this environment,
so please confirm the correct `MODEL_ID` and parameter names
from the Z-Image Turbo model card / docs.
"""

import os
from typing import Optional

import torch
from diffusers import DiffusionPipeline
import gradio as gr


# Replace this with the exact model id from Hugging Face if needed.
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(
    model_id: str = MODEL_ID,
    use_fp16: bool = True,
) -> DiffusionPipeline:
    """
    Load the Z-Image Turbo diffusion pipeline.

    This assumes the model is compatible with `DiffusionPipeline.from_pretrained`
    and supports both text-to-image and image-to-image via `image=` argument.
    """
    dtype = torch.float16 if use_fp16 and DEVICE == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe.to(DEVICE)
    return pipe


def text_to_image(
    pipe: DiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    seed: Optional[int],
):
    if not prompt:
        return None

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]


def image_to_image(
    pipe: DiffusionPipeline,
    prompt: str,
    init_image,
    strength: float,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    seed: Optional[int],
):
    if init_image is None or not prompt:
        return None

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Many Stable-Diffusion–style models support `image=` and `strength=`
    # for image-to-image. If Z-Image Turbo uses different parameter names,
    # adjust this call according to the model docs.
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
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


def create_demo(pipe: DiffusionPipeline):
    with gr.Blocks() as demo:
        gr.Markdown("# Z-Image Turbo Web UI (Kaggle)")

        with gr.Tab("Text to Image"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=3)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=3)
            with gr.Row():
                steps = gr.Slider(5, 50, value=30, step=1, label="Steps")
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

            btn = gr.Button("Generate")
            output = gr.Image(label="Output")

            btn.click(
                fn=lambda p, np, st, g, s: text_to_image(
                    pipe, p, np, st, g, _normalize_seed(s)
                ),
                inputs=[prompt, negative_prompt, steps, guidance, seed],
                outputs=output,
            )

        with gr.Tab("Image to Image"):
            with gr.Row():
                init_image = gr.Image(label="Init image", type="pil")
                with gr.Column():
                    prompt_i2i = gr.Textbox(label="Prompt", lines=3)
                    negative_prompt_i2i = gr.Textbox(label="Negative prompt", lines=3)
            with gr.Row():
                strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Strength")
                steps_i2i = gr.Slider(5, 50, value=30, step=1, label="Steps")
                guidance_i2i = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            with gr.Row():
                seed_i2i = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

            btn_i2i = gr.Button("Generate")
            output_i2i = gr.Image(label="Output")

            btn_i2i.click(
                fn=lambda p, img, strg, np, st, g, s: image_to_image(
                    pipe, p, img, strg, np, st, g, _normalize_seed(s)
                ),
                inputs=[
                    prompt_i2i,
                    init_image,
                    strength,
                    negative_prompt_i2i,
                    steps_i2i,
                    guidance_i2i,
                    seed_i2i,
                ],
                outputs=output_i2i,
            )

    return demo


if __name__ == "__main__":
    z_pipe = load_pipeline()
    ui = create_demo(z_pipe)
    ui.queue().launch(server_name="0.0.0.0", server_port=7860)

