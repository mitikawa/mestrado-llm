import os
import math
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

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
    DataCollatorWithPadding
)

from peft import LoraConfig, get_peft_model


MODEL_NAME = "bert-base-uncased"

CHUNK_SIZE = 512     # max tokens per chunk
STRIDE = 128         # overlap
MAX_TRAIN_EXAMPLES = None 

NUM_EPOCHS = 1
TRAIN_BS = 8
EVAL_BS = 16
LEARNING_RATE = 2e-4
SEED = 42

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados")
MODEL_DIR = os.path.join(ROOT, "modelos")

OUTPUT_DIR = os.path.join(MODEL_DIR, "detect_chunked_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_file_path = os.path.join(DATA_DIR, "detect", "subtaskA_train_monolingual.jsonl")
dev_file_path = os.path.join(DATA_DIR, "detect", "subtaskA_dev_monolingual.jsonl")


df_train = pd.read_json(train_file_path, lines=True)
df_dev = pd.read_json(dev_file_path, lines=True)

def map_label(model_name):
    return "human" if model_name == "human" else "AI"

df_train["label_str"] = df_train["model"].apply(map_label)
df_dev["label_str"] = df_dev["model"].apply(map_label)

label_map = {"human": 0, "AI": 1}
df_train["label_id"] = df_train["label_str"].map(label_map)
df_dev["label_id"] = df_dev["label_str"].map(label_map)

print("Train label distribution:")
print(df_train["label_str"].value_counts())
print("\nDev label distribution:")
print(df_dev["label_str"].value_counts())


# Get the standard padding collator

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
default_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "human", 1: "AI"},
    label2id={"human": 0, "AI": 1},
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


def chunk_text_to_encodings(text, tokenizer, max_length=CHUNK_SIZE, stride=STRIDE):
    """Split text into chunks with overlap."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=stride,
        return_attention_mask=True,
    )
    # Normalize nested results
    input_ids = enc["input_ids"]
    attention = enc["attention_mask"]

    if isinstance(input_ids[0], int):  # Single chunk
        return [{"input_ids": input_ids, "attention_mask": attention}]
    else:
        # List of chunks
        return [{"input_ids": ii, "attention_mask": am} for ii, am in zip(input_ids, attention)]


class ChunkedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE, limit=None):
        self.items = []

        rows = df.itertuples()
        if limit:
            rows = list(rows)[:limit]

        for row in tqdm(rows, desc="Chunking documents"):
            orig_id = int(row.Index)
            text = row.text
            label = int(row.label_id)

            encs = chunk_text_to_encodings(text, tokenizer, max_length=max_len, stride=stride)
            for enc in encs:
                self.items.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": label,
                    "orig_id": orig_id,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """Pad tensors + keep labels and orig_id."""
    enc = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
    )

    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    enc["orig_id"] = torch.tensor([b["orig_id"] for b in batch], dtype=torch.long)
    return enc

train_dataset = ChunkedTextDataset(df_train, tokenizer, limit=MAX_TRAIN_EXAMPLES)
dev_dataset = ChunkedTextDataset(df_dev, tokenizer)

print(f"\nTrain chunks: {len(train_dataset)}")
print(f"Dev chunks:   {len(dev_dataset)}\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,             # <--- ALL OUTPUT HERE
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_total_limit=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    report_to="none",
)

def compute_chunk_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"chunk_accuracy": (preds == pred.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    # data_collator=collate_fn,
    compute_metrics=compute_chunk_metrics,
)


trainer.train()

print("\nRunning document-level evaluation...")

device = model.device
model.eval()

loader = DataLoader(dev_dataset, batch_size=EVAL_BS, collate_fn=collate_fn)

all_logits = []
all_orig_ids = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Inferencing chunks"):
        orig_ids = batch.pop("orig_id").cpu().numpy()
        labels = batch.pop("labels").cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu().numpy()

        all_logits.append(logits)
        all_orig_ids.extend(orig_ids)

all_logits = np.vstack(all_logits)
all_orig_ids = np.array(all_orig_ids)

doc_chunks = defaultdict(list)
for oid, logit in zip(all_orig_ids, all_logits):
    doc_chunks[int(oid)].append(logit)

doc_preds = {}
for oid, logs in doc_chunks.items():
    avg_logit = np.mean(np.vstack(logs), axis=0)
    doc_preds[int(oid)] = int(np.argmax(avg_logit))

# Build final lists
y_true = df_dev.loc[list(doc_preds.keys()), "label_id"].tolist()
y_pred = list(doc_preds.values())

inv = {0: "human", 1: "AI"}

print("\n=== DOCUMENT-LEVEL REPORT ===\n")
print(classification_report([inv[i] for i in y_true],
                            [inv[i] for i in y_pred]))


print("\nMerging LoRA weights and saving final model...")

merged_model = model.merge_and_unload()
final_dir = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_dir, exist_ok=True)

merged_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"\nSaved inference-ready model to:\n{final_dir}\n")
