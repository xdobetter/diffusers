import os

from huggingface_hub.repocard import RepoCard
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import torch
from safetensors.torch import load_file


def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

def log_validation(
    pipeline,
    args,
    pipeline_args,
):
    print(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    # scheduler_args = {}
    #
    # if "variance_type" in pipeline.scheduler.config:
    #     variance_type = pipeline.scheduler.config.variance_type
    #
    #     if variance_type in ["learned", "learned_range"]:
    #         variance_type = "fixed_small"
    #
    #     scheduler_args["variance_type"] = variance_type
    #
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline.set_progress_bar_config(disable=False)

    # run inference
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None

    images = []
    for _ in range(args.num_validation_images):
        with torch.cuda.amp.autocast():
            image = pipeline(**pipeline_args, generator=generator).images[0]
            images.append(image)

    # save images
    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None
    for i, image in enumerate(images):
        image.save(f"{args.output_dir}/validation_image_{i}.png")

    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # lora_model_id = "model/s2_girl_character_lora_v1"
    # card = RepoCard.load(lora_model_id)
    # base_model_id = card.data.to_dict()["base_model"]

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",  torch_dtype=torch.float16, safety_checker=None)
    # pipe.load_lora_weights("model/s2_girl_character_lora_v1",weight_name="pytorch_lora_weights.safetensors")

    model = "model/s2_girl_character_lora_v1/"+"pytorch_lora_weights.safetensors"
    pipe = load_lora_weights(pipe,model)
    pipe = pipe.to("cuda")
    
    args = {
        "num_validation_images": 1,
        "validation_prompt": "a sks female anime character laughing",
        "output_dir": "./valid/s5/"
    }
    pipeline_args = {
        "num_inference_steps": 25,
    }

    log_validation(pipe, args, pipeline_args)