set -v

# s1_girl_character_if_v2
export project_name="s1_girl_character_if_v1" # !modify
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_ID="s1_girl_character" # !modify
export OUTPUT_DIR="model/$project_name"
export dataset="character" # !modify
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters" # !modify

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks female anime character" \
  --resolution=64 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=50 \
  --validation_prompt="a sks female anime character waving hand  with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --checkpointing_steps=500 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a female anime character" \
  --num_class_images=1000 \
  --num_validation_images=16 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --mixed_precision="fp16" 
