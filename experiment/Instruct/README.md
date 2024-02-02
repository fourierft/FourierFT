Code for fine-tuning the llama or llama2 model on the instruction-following dataset.
```bash
python3 finetune.py \
  --model_name --model meta-llama/Llama-2-7b-hf \
  --batch_size 10 \
  --dataset_name dataset_path \
  --output_dir output_path \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 5 \
  --num_frequency 500
```
