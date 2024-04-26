set -v

# dog v1
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="dog"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/dog"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=600 \
  --validation_steps=100 \
  --validation_prompt="a photo sks dog with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=5000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo dog" \
  --num_class_images=200 \
  --num_validation_images=16 \
  --train_text_encoder
