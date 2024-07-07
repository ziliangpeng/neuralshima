import math
import os
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import time
import statsd
import click

s = statsd.StatsClient("localhost", 8125)

# Get current epoch second
CURRENT_EPOCH_SECOND = int(time.time())
STEPS = 50
img_count = 5
gap = img_count - 1
steps = math.ceil(STEPS / gap) * gap
csteps = steps // gap

def san(p: str) -> str: return '_'.join(p.split())


@click.pass_context
def get_suffix(ctx) -> str:
    p = ctx.params.get("prompt")
    pf = ctx.params.get("prompt_file")
    prompt = get_prompt(p, pf)
    suffix = san(prompt)
    if ctx.params.get("refine"):
        suffix += "_refine"
    return suffix

def get_prompt(prompt, prompt_file) -> str:
    if prompt:
        return prompt
    if prompt_file:
        prompt = ""
        with open(prompt_file) as f:
            for line in f.read().splitlines():
                if not line.startswith("#"):
                    prompt = line
    return prompt


@click.command()
@click.option('--refine', is_flag=True, default=False)
@click.option('--prompt', default=None)
@click.option('--prompt_file', default=None)
@click.pass_context
def main(ctx, refine, prompt, prompt_file):
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    base.enable_sequential_cpu_offload()
    if refine:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.enable_sequential_cpu_offload()

    def save_cb(step: int, timestep: int, latents: torch.FloatTensor):
        print(step, steps)
        if step % csteps != 0 and step != steps - 1:
            return
        if step != steps - 1:
            return

        with torch.no_grad():
            latents = 1 / base.vae.config.scaling_factor * latents
            de = base.vae.to(torch.float16).decode(latents.to(torch.float16))
            # de = base.vae.decode(latents)
            image = de.sample

        # Convert to PIL Image and save
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])

        image.save(f"/tmp/sdxl/{CURRENT_EPOCH_SECOND}_{step}_of_{steps}_{get_suffix()}.png")


    prompt = get_prompt(prompt, prompt_file)
    print(f"epoch: {CURRENT_EPOCH_SECOND}, prompt: {prompt}")

    if refine:
        high_noise_frac = 0.8
        image = base(
            prompt=prompt,
            num_inference_steps=steps,
            # callback=save_cb,
            # callback_steps=1,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_start=high_noise_frac,
            image=image,
            # callback=save_cb,
            # callback_steps=1,
        ).images[0]
    else:
        image = base(
            prompt=prompt,
            num_inference_steps=steps,
            # callback=save_cb,
            # callback_steps=1,
        ).images[0]

    suffix = get_suffix()
    directory = "/tmp/sdxl/" + san(prompt)
    filename = f"{CURRENT_EPOCH_SECOND}_{suffix}.jpg"
    os.makedirs(directory, exist_ok=True)
    image.save(f"{directory}/{filename}")
    s.incr(f"sd.sdxl.total_count")

if __name__ == "__main__":
    main()
