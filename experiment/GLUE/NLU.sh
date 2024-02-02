python3 NLU-lora.py \
    --model_name_or_path roberta-base  \
    --dataset_path data/CoLA \
    --task_name cola \
    --lora_r 8 \
    --lora_alpha 32 \
    --output_dir output/CoLA/lora_r8 \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --per_device_train_batch_size 100 \
    --learning_rate 2e-3 \
    --overwrite_output_dir y \
    --num_train_epochs 30 \
    --logging_steps 1
    