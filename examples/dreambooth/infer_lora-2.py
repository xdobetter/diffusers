from diffusers import DiffusionPipeline
# import torch
# from safetensors.torch import load_file
# from collections import defaultdict

# # def load_lora_weights(pipeline, checkpoint_path):
# #     # load base model
# #     pipeline.to("cuda")
# #     LORA_PREFIX_UNET = "lora_unet"
# #     LORA_PREFIX_TEXT_ENCODER = "lora_te"
# #     alpha = 0.75
# #     # load LoRA weight from .safetensors
# #     state_dict = load_file(checkpoint_path, device="cuda")
# #     visited = []

# #     # directly update weight in diffusers model
# #     for key in state_dict:
# #         # it is suggested to print out the key, it usually will be something like below
# #         # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

# #         # as we have set the alpha beforehand, so just skip
# #         if ".alpha" in key or key in visited:
# #             continue

# #         if "text" in key:
# #             layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
# #             curr_layer = pipeline.text_encoder
# #         else:
# #             layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
# #             curr_layer = pipeline.unet

# #         # find the target layer
# #         temp_name = layer_infos.pop(0)
# #         while len(layer_infos) > -1:
# #             try:
# #                 curr_layer = curr_layer.__getattr__(temp_name)
# #                 if len(layer_infos) > 0:
# #                     temp_name = layer_infos.pop(0)
# #                 elif len(layer_infos) == 0:
# #                     break
# #             except Exception:
# #                 if len(temp_name) > 0:
# #                     temp_name += "_" + layer_infos.pop(0)
# #                 else:
# #                     temp_name = layer_infos.pop(0)

# #         pair_keys = []
# #         if "lora_down" in key:
# #             pair_keys.append(key.replace("lora_down", "lora_up"))
# #             pair_keys.append(key)
# #         else:
# #             pair_keys.append(key)
# #             pair_keys.append(key.replace("lora_up", "lora_down"))

# #         # update weight
# #         if len(state_dict[pair_keys[0]].shape) == 4:
# #             weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
# #             weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
# #             curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
# #         else:
# #             weight_up = state_dict[pair_keys[0]].to(torch.float32)
# #             weight_down = state_dict[pair_keys[1]].to(torch.float32)
# #             curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

# #         # update visited list
# #         for item in pair_keys:
# #             visited.append(item)

# #     return pipeline



# def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
#     LORA_PREFIX_UNET = "lora_unet"
#     LORA_PREFIX_TEXT_ENCODER = "lora_te"
#     # load LoRA weight from .safetensors
#     state_dict = load_file(checkpoint_path, device=device)

#     updates = defaultdict(dict)
#     for key, value in state_dict.items():
#         # it is suggested to print out the key, it usually will be something like below
#         # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

#         layer, elem = key.split('.', 1)
#         updates[layer][elem] = value

#     # directly update weight in diffusers model
#     for layer, elems in updates.items():

#         if "text" in layer:
#             layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
#             curr_layer = pipeline.text_encoder
#         else:
#             layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
#             curr_layer = pipeline.unet

#         # find the target layer
#         temp_name = layer_infos.pop(0)
#         while len(layer_infos) > -1:
#             try:
#                 curr_layer = curr_layer.__getattr__(temp_name)
#                 if len(layer_infos) > 0:
#                     temp_name = layer_infos.pop(0)
#                 elif len(layer_infos) == 0:
#                     break
#             except Exception:
#                 if len(temp_name) > 0:
#                     temp_name += "_" + layer_infos.pop(0)
#                 else:
#                     temp_name = layer_infos.pop(0)

#         # get elements for this layer
#         weight_up = elems['lora_up.weight'].to(dtype)
#         weight_down = elems['lora_down.weight'].to(dtype)
#         alpha = elems['alpha']
#         if alpha:
#             alpha = alpha.item() / weight_up.shape[1]
#         else:
#             alpha = 1.0

#         # update weight
#         if len(weight_up.shape) == 4:
#             curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
#         else:
#             curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

#     return pipeline

# pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
# # pipe.load_lora_weights("model/s2_girl_character_lora_v1")
# model = "model/s2_girl_character_lora_v1/"+"pytorch_lora_weights.safetensors"
# pipe = load_lora_weights(pipe,model,0.5, 'cuda', torch.float16)
# pipe = pipe.to("cuda")
# image = pipe("a sks female anime character laughing", num_inference_steps=25).images[0]

# pipeline = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5"
# )

# # load attention processors
# pipeline.load_lora_weights("model/s15_girl_character_lora_v2", weight_name="pytorch_lora_weights.safetensors") # KeyError: 'to_k_lora.down.weight'