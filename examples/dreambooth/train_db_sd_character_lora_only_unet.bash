set -v


# s13_girl_character_lora_v1
 export project_name="s13_girl_character_lora_v1" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s13_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --train_text_encoder \
   --dataloader_num_workers=16



 # s1_girl_character_lora_v2
 export project_name="s1_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s1_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16

 # s2_girl_character_lora_v2
 export project_name="s2_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s2_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view" \
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16

# s10_girl_character_lora_v2
 export project_name="s10_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s10_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16

# s12_girl_character_lora_v2
 export project_name="s12_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s12_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16

# s13_girl_character_lora_v2
 export project_name="s13_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s13_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16


# s14_girl_character_lora_v2
 export project_name="s14_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s14_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16

# s15_girl_character_lora_v2
 export project_name="s15_girl_character_lora_v2" # !modify
 export MODEL_NAME="runwayml/stable-diffusion-v1-5"
 export INSTANCE_ID="s15_girl_character" # !modify
 export OUTPUT_DIR="model/$project_name"
 export dataset="character" # !modify
 export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
 export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
 export CLASS_DIR="ppl/anime_characters" # !modify


 accelerate launch train_dreambooth_lora.py \
   --pretrained_model_name_or_path=$MODEL_NAME  \
   --instance_data_dir=$INSTANCE_DIR \
   --output_dir=$OUTPUT_DIR \
   --instance_prompt="a sks female anime character" \
   --resolution=512 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-5 \
   --lr_scheduler="constant" \
   --lr_warmup_steps=0 \
   --max_train_steps=1500 \
   --validation_steps=100 \
   --validation_prompt="A photo of a sks female anime character,side view"\
   --logging_dir=$LOGGING_DIR \
   --enable_xformers_memory_efficient_attention \
   --checkpointing_steps=500 \
   --with_prior_preservation --prior_loss_weight=1.0 \
   --class_data_dir=$CLASS_DIR \
   --class_prompt="a female anime character" \
   --num_class_images=1000 \
   --num_validation_images=8 \
   --dataloader_num_workers=16