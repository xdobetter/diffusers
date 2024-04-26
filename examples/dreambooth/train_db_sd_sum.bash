set -v

# dog
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="dog"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --validation_steps=100 \
  --validation_prompt="a photo of sks dog in the snow" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention

# red_cartoon
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="red_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a red red_cartoon" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_steps=100 \
  --validation_prompt="a red catoon" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# red_cartoon v2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="red_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks red_cartoon" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_steps=100 \
  --validation_prompt="a sks red_catoon riding a bicycle" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention






# black_character
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_character"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a black_girl_character" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_steps=100 \
  --validation_prompt="a character on the table" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# black_character v2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a black sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_steps=100 \
  --validation_prompt="a black sks cartoon_girl" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# black_character v3
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# black_character v4
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder


  

# gray_character v1
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="gray_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# gray_character v2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="gray_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder



# pink_girl_cartoon v1
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# pink_girl_cartoon v2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder


# yellow_girl_cartoon v1
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# yellow_girl_cartoon v2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a sks cartoon_girl wearing a red hat" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder


# yellow_girl_cartoon v3
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a cartoon sks cartoon_girl" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention


# yellow_girl_cartoon v4
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks cartoon_girl" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a cartoon sks cartoon_girl" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder


# yellow_girl_cartoon v5
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a cartoon sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a cartoon sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a cartoon girl" \
  --num_class_images=200 \
  --num_validation_images=8 



# yellow_girl_cartoon v6 验证text_encoder不训练下的效果 质量会差很多
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a cartoon sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a cartoon sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a cartoon girl" \
  --num_class_images=200 \
  --num_validation_images=8 




# yellow_girl_cartoon v7 更换prompt
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/$INSTANCE_ID"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters cartoon girl" \
  --num_class_images=200 \
  --num_validation_images=8 \
  --train_text_encoder





# yellow_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder


# black_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder



# gray_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="gray_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder


# pink_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl waving hand with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder





# yellow_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder


# black_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder



# gray_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="gray_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder


# pink_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=3000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=1000 \
  --num_validation_images=8 \
  --train_text_encoder



set -v

# yellow_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# black_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="black_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder



# gray_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="gray_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# pink_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# pink_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# pink_girl_cartoon v8
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="pink_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl laughing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# yellow_girl_cartoon v9
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# yellow_girl_cartoon v10
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="yellow_girl_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="character"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/anime_characters"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a anime characters sks girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a anime characters sks girl dancing with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a anime characters girl" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


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
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a photo sks dog with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo sks dog" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder


# cartoon v1
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_ID="red_cartoon"
export OUTPUT_DIR="model/$INSTANCE_ID"
export dataset="official"
export INSTANCE_DIR="data/$dataset/$INSTANCE_ID"
export LOGGING_DIR="log/$INSTANCE_ID"
export CLASS_DIR="ppl/cartoon"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a cartoon sks character" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_steps=100 \
  --validation_prompt="a cartoon sks character with a tree and autumn leaves in the background" \
  --logging_dir=$LOGGING_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=1000 \
  --with_prior_preservation --prior_loss_weight=1.5 \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a cartoon character" \
  --num_class_images=5000 \
  --num_validation_images=16 \
  --train_text_encoder