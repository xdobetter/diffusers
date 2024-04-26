set -v


# # dog1_v2
# export project_name="dog1_v2"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_ID="dog1"
# export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
# export dataset="official"
# export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
# export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
# export CLASS_DIR="ppl/dog"


# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=150 \
#   --validation_steps=25 \
#   --validation_prompt="a sks dog with a tree and autumn leaves in the background" \
#   --logging_dir=$LOGGING_DIR \
#   --enable_xformers_memory_efficient_attention \
#   --checkpointing_steps=500 \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a dog" \
#   --num_class_images=1000 \
#   --num_validation_images=16 \
#   --train_text_encoder

# duck_toy_v2
export project_name="duck_toy_v2"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="duck_toy"
export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
export dataset="duck_toy"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
export CLASS_DIR="ppl/toy"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks duck toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=150 \
  --validation_steps=25 \
  --validation_prompt="a sks duck toy with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a duck toy" \
  --num_class_images=1000 \
  --num_validation_images=16 \
  --train_text_encoder


# backpack_v2
export project_name="backpack_v2"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="backpack"
export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
export dataset="backpack"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
export CLASS_DIR="ppl/backpack"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks backpack" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=150 \
  --validation_steps=25 \
  --validation_prompt="a sks backpack with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a backpack" \
  --num_class_images=1000 \
  --num_validation_images=16 \
  --train_text_encoder


# teapot_v2
export project_name="teapot_v2"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="teapot"
export OUTPUT_DIR="model/$project_name/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$project_name/$INSTANCE_ID"
export CLASS_DIR="ppl/teapot"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks teapot" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=150 \
  --validation_steps=25 \
  --validation_prompt="a sks teapot with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a teapot" \
  --num_class_images=1000 \
  --num_validation_images=16 \
  --train_text_encoder