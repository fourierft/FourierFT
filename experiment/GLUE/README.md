Modified code for running experiments on the GLUE benchmark.

```bash
python3 NLU-fourier.py \
    --model_name_or_path roberta-base \
    --dataset_path data/CoLA \
    --task_name cola \
    --n_frequency 300 \
    --output_dir output/CoLA/fourier_300 \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --per_device_train_batch_size 200 \
    --learning_rate 0.03 \
    --overwrite_output_dir y \
    --num_train_epochs 30 \
    --logging_steps 30 \
    --evaluation_strategy epoch
```