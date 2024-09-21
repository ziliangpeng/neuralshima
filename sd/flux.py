import torch
from diffusers import  FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

prompt = "A cat holding a sign that says hello world"
prompt = "photo of tokyo city in year 2100 with mega skyscrapers and mega structures in the sky"
prompt = "Tokyo and New York blended into one city"
prompt = "a futuristic city with a giant robot in the middle"
images = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=256,
    num_images_per_prompt=5,
).images

for i, img in enumerate(images):
    img.save(f"image_{i}.png")