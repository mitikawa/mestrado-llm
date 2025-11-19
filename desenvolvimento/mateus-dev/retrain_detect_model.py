from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "dados"))
MODEL_DIR = os.path.abspath(os.path.join(ROOT, "modelos"))

dev_file_path = os.path.join(DATA_DIR, "detect","subtaskA_dev_monolingual.jsonl")
train_file_path = os.path.join(DATA_DIR, "detect","subtaskA_train_monolingual.jsonl")

print("Loading datasets..")
with open(train_file_path, "r") as f:
    df_train = pd.read_json(f, lines=True)
with open(dev_file_path, "r") as f:
    df_dev   = pd.read_json(f, lines=True)

# Map model names to "human" vs "AI"
def map_label(model_name):
    return "human" if model_name == "human" else "AI"

df_train['label_str'] = df_train['model'].apply(map_label)
df_dev['label_str'] = df_dev['model'].apply(map_label)

label_map = {"human": 0, "AI": 1}
df_train["label_id"] = df_train["label_str"].map(label_map)
df_dev["label_id"] = df_dev["label_str"].map(label_map)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label
        }

saved_model_dir = os.path.join(MODEL_DIR, "final_detect_model")
saved_model_dir = str(Path(saved_model_dir).resolve())

tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)

train_dataset = TextDataset(df_train, tokenizer)
dev_dataset = TextDataset(df_dev, tokenizer)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    learning_rate=2e-4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

preds_output = trainer.predict(dev_dataset)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

inv_label_map = {0: "human", 1: "AI"}
y_true_labels = [inv_label_map[i] for i in y_true]
y_pred_labels = [inv_label_map[i] for i in y_pred]

print("\n--- Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels))


updated_dir = os.path.join(MODEL_DIR, "updated_detect_model")

model.save_pretrained(updated_dir)
tokenizer.save_pretrained(updated_dir)

print(f"Updated model saved to {updated_dir}")
