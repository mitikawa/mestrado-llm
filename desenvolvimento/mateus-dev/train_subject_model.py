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

dev_file_path = os.path.join(DATA_DIR, "subj", "dev_en.tsv")
train_file_path = os.path.join(DATA_DIR, "subj", "train_en.tsv")

with open(train_file_path, "r") as f:
    df_train = pd.read_csv(f, sep="\t")

with open(dev_file_path, "r") as f:
    df_dev = pd.read_csv(f, sep="\t")


label_map = {"OBJ": 0, "SUBJ": 1}
df_train["label"] = df_train["label"].map(label_map)
df_dev["label"] = df_dev["label"].map(label_map)

print("Label distribution (train):")
print(df_train["label"].value_counts())
print("\nLabel distribution (dev):")
print(df_dev["label"].value_counts())


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["sentence"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # "OBJ" and "SUBJ"
    id2label={0: "OBJ", 1: "SUBJ"},  # Adicionado para clareza
    label2id={"OBJ": 0, "SUBJ": 1},
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
    output_dir="./subj-checkpoints",
    num_train_epochs=20,
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

preds_output = trainer.predict(dev_dataset)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

inv_label_map = {0: "OBJ", 1: "SUBJ"}
y_true_labels = [inv_label_map[i] for i in y_true]
y_pred_labels = [inv_label_map[i] for i in y_pred]

print("\n--- Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels))
print("\nMerging LoRA into base model...")

merged_model = model.merge_and_unload()


output_dir = os.path.join(MODEL_DIR, "final_subject_model")
os.makedirs(output_dir, exist_ok=True)

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nSaved inference-ready model to: {output_dir}")
