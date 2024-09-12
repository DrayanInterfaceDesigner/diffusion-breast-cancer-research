from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# pip install diffusers transformers accelerate
# pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

low_res_img = Image.open("original_image.jpg").convert("RGB")
# aumente o fator de downscale para acelerar o processamento
downscale_factor = 4
low_res_img = low_res_img.resize((low_res_img.width // downscale_factor, low_res_img.height // downscale_factor))
prompt = "a car in the road"


# guidance_scale is a hyperparameter that controls the strength of the guidance.
# 1.0 is almost no guidance, 0.0 is full guidance.
upscaled_image = pipeline(prompt=prompt, image=low_res_img, guidance_scale=1.0).images[0]
upscaled_image.save("output.png")