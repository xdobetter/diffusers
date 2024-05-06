set -v

# dog2_lora_full
 export project_name="dog2_lora_full" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="dog2" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="official" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/dog" # !modify

 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a photo sks dog" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=5e-4 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1000 \
   --validation_steps=250 \
   --validation_prompt="a photo sks dog with a tree and autumn leaves in the background"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a photo dog" \
   --num_class_images=1000 \
   --num_validation_images=4 \
   --train_text_encoder \
   --dataloader_num_workers=16

# # backpack_lora_v1
#  export project_name="backpack_lora_full" # !modify
#  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#  export INSTANCE_ID="backpack" # !modify
#  export OUTPUT_DIR="model/$project_name"
#  export dataset="official" # !modify
#  export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
#  export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
#  export CLASS_DIR="ppl/backpack" # !modify


#  accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path=$MODEL_NAME  \
#    --instance_data_dir=$INSTANCE_DIR \
#    --output_dir=$OUTPUT_DIR \
#    --instance_prompt="a photo sks backpack" \
#    --resolution=512 \
#    --train_batch_size=1 \
#    --gradient_accumulation_steps=1 \
#    --learning_rate=1e-4 \
#    --lr_scheduler="constant" \
#    --lr_warmup_steps=0 \
#    --max_train_steps=1000 \
#    --validation_steps=100 \
#    --validation_prompt="a photo sks backpack with a tree and autumn leaves in the background"\
#    --logging_dir=$LOGGING_DIR \
#    --enable_xformers_memory_efficient_attention \
#    --checkpointing_steps=500 \
#    --with_prior_preservation --prior_loss_weight=1.0 \
#    --class_data_dir=$CLASS_DIR \
#    --class_prompt="a photo backpack" \
#    --num_class_images=1000 \
#    --num_validation_images=16 \
#    --train_text_encoder \
#    --dataloader_num_workers=16




# dog1_lora_v1_only_unet
#  export project_name="dog1_lora_v4" # !modify
#  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#  export INSTANCE_ID="dog1" # !modify
#  export OUTPUT_DIR="model/$project_name"
#  export dataset="official" # !modify
#  export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
#  export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
#  export CLASS_DIR="ppl/dog" # !modify


#  accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path=$MODEL_NAME  \
#    --instance_data_dir=$INSTANCE_DIR \
#    --output_dir=$OUTPUT_DIR \
#    --instance_prompt="a photo sks dog" \
#    --resolution=512 \
#    --train_batch_size=1 \
#    --gradient_accumulation_steps=1 \
#    --learning_rate=1e-4 \
#    --lr_scheduler="constant" \
#    --lr_warmup_steps=0 \
#    --max_train_steps=1000 \
#    --validation_steps=250 \
#    --validation_prompt="a photo sks dog with a tree and autumn leaves in the background"\
#    --logging_dir=$LOGGING_DIR \
#    --enable_xformers_memory_efficient_attention \
#    --checkpointing_steps=500 \
#    --num_validation_images=8 \
#    --dataloader_num_workers=16 \
#   #  --train_text_encoder \
#   # --with_prior_preservation --prior_loss_weight=1.0 \
#   #  --class_data_dir=$CLASS_DIR \
#   #  --class_prompt="a photo dog" \
#   #  --num_class_images=10 \