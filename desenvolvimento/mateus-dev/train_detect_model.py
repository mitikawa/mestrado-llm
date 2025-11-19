import os
import json
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DATA_DIR = os.path.abspath(os.path.join(ROOT, "dados"))
MODEL_DIR = os.path.abspath(os.path.join(ROOT, "modelos"))

dev_file_path = os.path.join(DATA_DIR, "detect", "subtaskA_dev_monolingual.jsonl")
train_file_path = os.path.join(DATA_DIR, "detect", "subtaskA_train_monolingual.jsonl")

with open(train_file_path, "r") as f:
    df_train = pd.read_json(f, lines=True)
with open(dev_file_path, "r") as f:
    df_dev = pd.read_json(f, lines=True)


def map_label(model_name):
    return "human" if model_name == "human" else "AI"


df_train["label_str"] = df_train["model"].apply(map_label)
df_dev["label_str"] = df_dev["model"].apply(map_label)

label_map = {"human": 0, "AI": 1}

df_train["label_id"] = df_train["label_str"].map(label_map)
df_dev["label_id"] = df_dev["label_str"].map(label_map)

print(df_train["label_str"].value_counts())
print(df_dev["label_str"].value_counts())


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
        }


model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
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


train_dataset = TextDataset(df_train, tokenizer)
dev_dataset = TextDataset(df_dev, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer)


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = (preds == labels).mean()
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=2e-4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

pred_output = trainer.predict(dev_dataset)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

inv = {0: "human", 1: "AI"}

print("\n--- Classification Report ---")
print(classification_report([inv[i] for i in y_true], [inv[i] for i in y_pred]))

print("\nMerging LoRA into base model...")

merged_model = model.merge_and_unload()

output_dir = os.path.join(MODEL_DIR, "final_detect_model")
os.makedirs(output_dir, exist_ok=True)

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nSaved inference-ready model to: {output_dir}")
