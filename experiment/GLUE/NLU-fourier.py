import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    FourierConfig
)
import wandb
import evaluate
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, set_seed

wandb.init(project="Fourier-finetune", )
timestamp = datetime.now().strftime("%m-%d_%H-%M")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--task", type=str, default="mrpc")
parser.add_argument("--bs", type=int, default=256)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--save_weight", action='store_true')
parser.add_argument("--n_frequency", type=int, default=200)
parser.add_argument("--lr_classifier", type=float, default=5e-3)
parser.add_argument("--lr_spectrum", type=float, default=1e-1)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--train_ratio", type=float, default=1)

args = parser.parse_args()
wandb.config.update(args)
print(args)


batch_size = args.bs
peft_type = PeftType.FOURIER
device = torch.device(f"cuda:{args.device}")

peft_config = FourierConfig(task_type="SEQ_CLS", inference_mode=False, n_frequency=args.n_frequency)

if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if args.dataset:
    datasets = load_dataset(args.dataset)
else:
    datasets = load_dataset(args.task)
datasets['train'] = datasets['train'].select(range(int(len(datasets['train']) * args.train_ratio)))
metric = load_metric("./glue.py", args.task)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Labels
if args.task is not None:
    is_regression = args.task == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

# Preprocessing the datasets
if args.task is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task]
else:
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    input_text = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
    outputs = tokenizer(*input_text, truncation=True, max_length=args.max_length, padding=True)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", sentence1_key, sentence2_key] if sentence2_key else ["idx", sentence1_key]
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task
    )

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

classifier_parameters = list(map(id, model.classifier.parameters()))

spectrum_parameters = filter(lambda p: id(p) not in classifier_parameters, model.parameters())

# optimizer = AdamW(params=model.parameters(), lr=lr)
optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": args.lr_classifier, "weight_decay": args.weight_decay},
    {"params": spectrum_parameters, "lr": args.lr_spectrum, "weight_decay": args.weight_decay},
])

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)


model.to(device)
acc_list = []
for epoch in range(args.num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    wandb.log(eval_metric)
    if args.task == "stsb":
        acc_list.append(eval_metric['pearson'])
        print(f"epoch {epoch}:", eval_metric, ', current_best_pearson:',max(acc_list),'train_loss:',loss)
    elif args.task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}:", eval_metric, ', current_best_corr:',max(acc_list),'train_loss:',loss)
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}:", eval_metric, ', current_best_acc:',max(acc_list),'train_loss:',loss)
    