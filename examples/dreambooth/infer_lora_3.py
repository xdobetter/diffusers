from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
import os
import torch
from peft import PeftModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
text = "A photo of sks dog in a bucket"
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,dtype=torch.float16,requires_safety_checker=False).to("cuda")
text_encoder_temp = pipe.text_encoder
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer"
)
inputs = tokenizer(text,return_tensors="pt").to("cuda")
output = text_encoder_temp(**inputs)
print(output)
# unet_temp = pipe.unet
# print(pipe.unet)
# print(pipe.text_encoder)
pipe.load_lora_weights("model/dog1_lora_v1_c1000")
# print(pipe.unet)
# print(pipe.text_encoder)
# print(pipe.text_encoder)
output = pipe.text_encoder(**inputs)
print(output)
# pipe.text_encoder = text_encoder_temp
# pipe.unet = unet_temp
# image = pipe(text,num_inference_steps=50,guidance_sacle=7).images[0]
# image.save("test-unet+text_encoder_v3.png")


        # pipe_kwargs = {
        #     # "tokenizer": None, # tokenizer为None!
        #     "safety_checker": None,
        #     "feature_extractor": None,
        #     "requires_safety_checker": False,
        #     "unet":None,
        #     "torch_dtype": torch.float32,
        # }
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     pretrained_model_name_or_path,
        #     **pipe_kwargs,
        # )
        # if lora_weights_path != None:
        #     pipe.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")