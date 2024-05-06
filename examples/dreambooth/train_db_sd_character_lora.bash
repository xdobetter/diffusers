set -v

# s15_girl_character_lora_v4
 export project_name="s15_girl_character_lora_v4" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s15_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
#  export CLASS_DIR="ppl/anime_characters_v2" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a zoomed out DSLR photo of sks a female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-4 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1000 \
   --validation_steps=100 \
   --validation_prompt="a zoomed out DSLR photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --num_validation_images=16 \
   --train_text_encoder \
   --dataloader_num_workers=16 \


  #  --with_prior_preservation --prior_loss_weight=1.0 \
  #  --class_data_dir=$CLASS_DIR \
  #  --class_prompt="a zoomed out DSLR photo of female anime character" \
  #  --num_class_images=200 \


# # s13_girl_character_lora_v3
#  export project_name="s13_girl_character_lora_v3" # !modify
#  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#  export INSTANCE_ID="s13_girl_character" # !modify
#  export OUTPUT_DIR="model/$project_name"
#  export dataset="character" # !modify
#  export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
#  export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
#  export CLASS_DIR="ppl/anime_characters" # !modify


#  accelerate launch train_dreambooth_lora.py \
#    --pretrained_model_name_or_path=$MODEL_NAME  \
#    --instance_data_dir=$INSTANCE_DIR \
#    --output_dir=$OUTPUT_DIR \
#    --instance_prompt="a sks female anime character" \
#    --resolution=512 \
#    --train_batch_size=1 \
#    --gradient_accumulation_steps=1 \
#    --learning_rate=1e-4 \
#    --lr_scheduler="constant" \
#    --lr_warmup_steps=0 \
#    --max_train_steps=1500 \
#    --validation_steps=100 \
#    --validation_prompt="a zoomed out DSLR photo of sks female anime character,side view"\
#    --logging_dir=$LOGGING_DIR \
#    --enable_xformers_memory_efficient_attention \
#    --checkpointing_steps=500 \
#    --with_prior_preservation --prior_loss_weight=1.0 \
#    --class_data_dir=$CLASS_DIR \
#    --class_prompt="a female anime character" \
#    --num_class_images=1000 \
#    --num_validation_images=16 \
#    --train_text_encoder \
#    --dataloader_num_workers=16