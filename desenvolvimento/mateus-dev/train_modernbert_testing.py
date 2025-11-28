#!/usr/bin/env python3
"""
train_modernbert_3060ti_bf16.py

Copy-paste ready. Optimized for NVIDIA 3060 Ti (Ampere Architecture).
Uses BFloat16 (bf16) for numerical stability and speed, removing the need
for GradScaler/FP16 fallback logic.
"""
import os
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model

# --------------------------
# Basic settings / hyperparams
# --------------------------
MODEL_NAME = "answerdotai/ModernBERT-large"
NUM_LABELS = 2

CHUNK_SIZE = 384
STRIDE = 96

MAX_TRAIN_EXAMPLES = None

NUM_EPOCHS = 1
TRAIN_BS = 8 
EVAL_BS = 16
LEARNING_RATE = 2e-4
SEED = 42

# --------------------------
# CUDA / PyTorch perf tweaks
# --------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
torch.use_deterministic_algorithms(False)

# --------------------------
# Paths
# --------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dados"
MODEL_DIR = ROOT / "modelos"
OUTPUT_DIR = MODEL_DIR / "detect_modernbert_bf16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_file_path = DATA_DIR / "detect/subtaskA_train_monolingual.jsonl"
dev_file_path = DATA_DIR / "detect/subtaskA_dev_monolingual.jsonl"

# --------------------------
# Load data
# --------------------------
df_train = pd.read_json(train_file_path, lines=True)
df_dev = pd.read_json(dev_file_path, lines=True)

df_train["label_id"] = df_train["model"].apply(lambda x: 0 if x == "human" else 1)
df_dev["label_id"] = df_dev["model"].apply(lambda x: 0 if x == "human" else 1)

# --------------------------
# Tokenizer / collator
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer)

# --------------------------
# Device
# --------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Load base model (BFloat16)
# --------------------------
print("Loading base model in BFloat16...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    dtype=torch.bfloat16,  # Native Ampere support
    device_map=None,       # Manual device placement for stability
)

# Move to GPU explicitly
base_model.to(device)
base_model.gradient_checkpointing_enable()
print("Base model loaded and moved to device.")

# --------------------------
# LoRA config + wrap with PEFT
# --------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["Wqkv", "Wo"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(device)

# --------------------------
# Chunking helper + dataset
# --------------------------
def chunk_text(text, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_overflowing_tokens=True,
        stride=stride,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    if isinstance(input_ids[0], int):
        return [{"input_ids": input_ids, "attention_mask": attn}]
    return [{"input_ids": ii, "attention_mask": am} for ii, am in zip(input_ids, attn)]

class ChunkDataset(Dataset):
    def __init__(self, df, tokenizer, limit=None):
        rows = df.itertuples()
        if limit:
            rows = list(rows)[:limit]

        self.items = []
        for row in tqdm(rows, desc="Chunking"):
            encs = chunk_text(row.text, tokenizer)
            for enc in encs:
                self.items.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": int(row.label_id),
                    "orig_id": int(row.Index),
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_fn(batch):
    enc = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    enc["orig_id"] = torch.tensor([b["orig_id"] for b in batch], dtype=torch.long)
    return enc

train_dataset = ChunkDataset(df_train, tokenizer, limit=MAX_TRAIN_EXAMPLES)
dev_dataset = ChunkDataset(df_dev, tokenizer)

print("Train chunks:", len(train_dataset))
print("Dev chunks:", len(dev_dataset))

# --------------------------
# TrainingArguments (BF16 Enabled)
# --------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=50,  # Lower logging steps since dataset might be small

    optim="adamw_torch",
    fp16=False,            # Disable FP16
    bf16=True,             # Enable BFloat16 (3060 Ti supported)
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
)

# --------------------------
# Metrics
# --------------------------
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"chunk_acc": (preds == pred.label_ids).mean()}

# --------------------------
# Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# --------------------------
# Train
# --------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nStarting training (BF16)...")
trainer.train()

# --------------------------
# Merge LoRA weights, save model + tokenizer
# --------------------------
print("\nMerging LoRA weights and saving final model...")
try:
    merged = model.merge_and_unload()
except Exception:
    print("merge_and_unload failed; saving PEFT-wrapped model instead.")
    merged = model

save_dir = OUTPUT_DIR / "final_model"
save_dir.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved final model to:", save_dir)

# --------------------------
# Document-level evaluation
# --------------------------
print("\nRunning document-level evaluation...")

# Ensure evaluation model is on device
model = merged
model.to(device)
model.eval()

loader = DataLoader(dev_dataset, batch_size=EVAL_BS, collate_fn=collate_fn)

logits_all = []
orig_ids_all = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Inferencing chunks"):
        orig_ids = batch.pop("orig_id").cpu().numpy()
        labels = batch.pop("labels").cpu().numpy()

        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu().numpy()

        logits_all.append(logits)
        orig_ids_all.extend(orig_ids)

if len(logits_all) == 0:
    print("No logits collected during evaluation. Exiting.")
    exit(0)

logits_all = np.vstack(logits_all)
orig_ids_all = np.array(orig_ids_all)

# group chunk logits by original doc id
docs = defaultdict(list)
for oid, logit in zip(orig_ids_all, logits_all):
    docs[int(oid)].append(logit)

final_preds = {}
for oid, logs in docs.items():
    avg = np.mean(np.vstack(logs), axis=0)
    final_preds[oid] = int(np.argmax(avg))

# Evaluate on those documents we predicted
y_true = df_dev.loc[list(final_preds.keys()), "label_id"].tolist()
y_pred = list(final_preds.values())

inv = {0: "human", 1: "AI"}
print("\n=== DOCUMENT-LEVEL REPORT ===\n")
print(classification_report([inv[i] for i in y_true], [inv[i] for i in y_pred]))