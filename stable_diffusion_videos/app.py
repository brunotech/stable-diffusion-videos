import time

import gradio as gr
import torch

from .stable_diffusion_pipeline import StableDiffusionWalkPipeline
from .upsampling import RealESRGANModel

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")


def fn_images(
    prompt,
    seed,
    guidance_scale,
    num_inference_steps,
    upsample,
):
    if upsample:
        if getattr(pipeline, "upsampler", None) is None:
            pipeline.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
        pipeline.upsampler.to(pipeline.device)

    with torch.autocast("cuda"):
        img = pipeline(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=pipeline.device).manual_seed(
                seed
            ),
            output_type="numpy" if upsample else "pil",
        )["sample"][0]
        return pipeline.upsampler(img) if upsample else img


def fn_videos(
    prompt_1,
    seed_1,
    prompt_2,
    seed_2,
    guidance_scale,
    num_inference_steps,
    num_interpolation_steps,
    output_dir,
    upsample,
):
    prompts = [prompt_1, prompt_2]
    seeds = [seed_1, seed_2]

    prompts = [x for x in prompts if x.strip()]
    seeds = seeds[: len(prompts)]

    return pipeline.walk(
        guidance_scale=guidance_scale,
        prompts=prompts,
        seeds=seeds,
        num_interpolation_steps=num_interpolation_steps,
        num_inference_steps=num_inference_steps,
        output_dir=output_dir,
        name=time.strftime("%Y%m%d-%H%M%S"),
        upsample=upsample,
    )


interface_videos = gr.Interface(
    fn_videos,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Number(42, label="Seed 1", precision=0),
        gr.Textbox("strawberry spaghetti"),
        gr.Number(42, label="Seed 2", precision=0),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Slider(3, 240, 10),
        gr.Textbox(
            "dreams",
            placeholder=("Folder where outputs will be saved. Each output will be saved in a new folder."),
        ),
        gr.Checkbox(False),
    ],
    outputs=gr.Video(),
)

interface_images = gr.Interface(
    fn_images,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Number(42, label="Seed", precision=0),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Checkbox(False),
    ],
    outputs=gr.Image(type="pil"),
)

interface = gr.TabbedInterface([interface_images, interface_videos], ["Images!", "Videos!"])

if __name__ == "__main__":
    interface.launch(debug=True)
