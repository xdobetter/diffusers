# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "path-to-your-trained-model"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# prompt = "A photo of sks  in a bucket"
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# image.save("dog-bucket.png")



from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("path/to/model/checkpoint-100/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")

image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")