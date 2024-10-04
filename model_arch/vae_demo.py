from diffusers import AutoencoderKL
from PIL import Image
import torch
import numpy as np


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

device = "cuda"
model_path="CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained(
	model_path,
	subfolder="vae",
	revision="fp16"
).to(device)


raw_image = Image.open('iaa_pub6_.jpg')
h, w = raw_image.size
image = raw_image.resize((h//2,w//2))
p_image = preprocess(image).to(device)
vae.eval()
init_latent_dist = 0.18215 * vae.encode(p_image).latent_dist.sample()
print("Raw image size:", raw_image.size)
print("Image size:", image.size)
print("Resized Image size:", p_image.shape)
print("VAE feature:", init_latent_dist.shape)