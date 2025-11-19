import os
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "dados"))
MODEL_DIR = os.path.abspath(os.path.join(ROOT, "modelos"))

dev_file_path = os.path.join(DATA_DIR, "detect","subtaskA_dev_monolingual.jsonl")
model_path = os.path.join(MODEL_DIR, "final_detect_model")

with open(dev_file_path, "r") as f:
    df_dev = pd.read_json(f, lines=True)

def map_label(model_name):
    return "human" if model_name == "human" else "AI"

df_dev["label_str"] = df_dev["model"].apply(map_label)
label_map = {"human": 0, "AI": 1}
df_dev["label_id"] = df_dev["label_str"].map(label_map)

print(f"Loaded {len(df_dev)} records from dev set.")


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx]
        }

print(f"Loading merged model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded onto device: {device}")

dev_dataset = TextDataset(df_dev, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer)


def compute_metrics(pred):
    """Calculates F1, precision, recall, and accuracy."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=32,
    do_train=False,
    do_predict=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("\nRunning evaluation on the dev set...")
pred_output = trainer.predict(dev_dataset)

print("\n--- Metrics (F1, Precision, Recall, Accuracy) ---")
print(pred_output.metrics)


print("\n--- Full Classification Report ---")
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)
inv_label_map = {0: "human", 1: "AI"}

print(classification_report(
    [inv_label_map[i] for i in y_true],
    [inv_label_map[i] for i in y_pred]
))

print("\nEvaluation complete.")