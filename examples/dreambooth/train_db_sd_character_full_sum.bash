set -v


# # black_girl_character_v4
# export project_name="black_girl_character_v4"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="black_girl_character"
# export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=1500 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder


# # gray_girl_character_v4
# export project_name="gray_girl_character_v4"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="gray_girl_character"
# export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=1000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder


# # pink_girl_character_v4
# export project_name="pink_girl_character_v4"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="pink_girl_character"
# export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder


# # yellow_girl_character_v4
# export project_name="yellow_girl_character_v4"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="yellow_girl_character"
# export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder


# # s1_girl_character_v1
# export project_name="s1_girl_character_v1" # !modify
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="s1_girl_character" # !modify
# export OUTPUT_DIR="model/$project_name"
# export dataset="character" # !modify
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters" # !modify


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder


# # s2_girl_character_debug
# export project_name="s2_girl_character_debug"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="s2_girl_character"
# export OUTPUT_DIR="model/$project_name"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder

# # s10_girl_character_v1
# export project_name="s10_girl_character_v1"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="s10_girl_character"
# export OUTPUT_DIR="model/$project_name"
# export dataset="character"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/anime_characters"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks female anime character" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --validation_steps=50 \
#   --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a female anime character" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder

# s10_girl_character_v1
export project_name="s10_girl_character_v1"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="s10_girl_character"
export OUTPUT_DIR="model/$project_name"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks female anime character" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=50 \
  --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a female anime character" \
  --num_class_images=1000 \
  --num_validation_images=16 \
  --train_text_encoder