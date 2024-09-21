import math
import time

import requests
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import io
import base64
from PIL import Image
from loguru import logger

# Load the SDXL model
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
# pipe = pipe.to("cuda")
pipe.enable_sequential_cpu_offload()

# Define the function to generate images
def generate_images(prompt, scaled_iamge, negative_prompt=None, conditioning_scale=0.5, height=512, width=512, num_inference_steps=20, guidance_scale=7.5, num_images_per_prompt=1, strength=0.8):
    # Load the initial image
    # init_image = load_image(image_path).convert("RGB")
    init_image = scaled_iamge.convert("RGB")
    
    # Generate images
    images = pipe(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        # controlnet_conditioning_scale=conditioning_scale,
        # height=height,
        # width=width,
        num_inference_steps=num_inference_steps,
        # guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        strength=strength,
        image=init_image
    ).images
    
    # Convert images to base64
    finished_images = []
    for image in images:
        # buffered = io.BytesIO()
        # image.save(buffered, format="PNG")
        # finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        finished_images.append(image)
    
    return {"images": finished_images}

# Example usage
epoch = int(time.time())
prompt = "A fantastical landscape with mountains and rivers"
prompt = "realistic photos of tokyo city with mountain and sky in the background"
prompt = "girl statue under water"
prompt = "a bird shaped stone in the middle of the sky with clouds"
url = '/tmp/sdxl-init/twitter.jpg'
url = 'https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg' #elon
url = 'https://techcrunch.com/wp-content/uploads/2019/06/waymo-ipace.jpeg' # waymo
url = 'https://cdn.mos.cms.futurecdn.net/z3bn6deaxmrjmQHNEkpcZE-1200-80.jpg' # twitter logo
response = requests.get(url)
image = Image.open(io.BytesIO(response.content))

multiple = int(max(math.ceil(image.width / 512), math.ceil(image.height / 512)))
logger.info(f" scaled width: {image.width // multiple}, scaled height: {image.height // multiple}")
scaled_image = image.resize((image.width // multiple, image.height // multiple))
for s in range(80, 101, 1):
    strength = s / 100
    logger.info(f"strength: {strength}")
    result = generate_images(prompt, scaled_image, num_inference_steps=int(50*50/s), strength=strength, num_images_per_prompt=10)


    # Display the result
    for i, img_str in enumerate(result["images"]):
        # img_data = base64.b64decode(img_str)
        # img = Image.open(io.BytesIO(img_data))
        # img.show()
        img = img_str
        img.save(f'/tmp/sdxl-init/output-{epoch}-{strength:.2f}-{i:0{4}}.jpg')
