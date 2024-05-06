import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")

pipeline.set_ip_adapter_scale(0.5)

image = load_image("./data/character/s15_girl_character/s6_behind_rgba_rmbg_2.png")
generator = torch.Generator(device="cpu").manual_seed(26)

image = pipeline(
    # prompt="A photo of Einstein as a chef, wearing an apron, cooking in a French restaurant",
    prompt = "a female anime character",
    ip_adapter_image=image,
    negative_prompt="lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=100,
    generator=generator,
).images[0]
image.save("test.png")