export MODEL_NAME= "stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="dreambooth/dataset/dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-output"


accelerate launch --multi_gpu train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=5 \
  --gradient_accumulation_steps=2 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --rank 16 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" 