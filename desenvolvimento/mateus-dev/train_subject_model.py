import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report


# --- CONFIGURATION ---
CHUNK_SIZE = 256   
STRIDE = 32         


model_name = "bert-base-uncased"
# ---------------------

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


output_dir = os.path.join(MODEL_DIR, "final_subject_model")
os.makedirs(output_dir, exist_ok=True)

label_map = {"OBJ": 0, "SUBJ": 1}
df_train["label_id"] = df_train["label"].map(label_map) 
df_dev["label_id"] = df_dev["label"].map(label_map)

df_train['orig_id'] = df_train.index
df_dev['orig_id'] = df_dev.index

print("Label distribution (train):")
print(df_train["label_id"].value_counts())
print("\nLabel distribution (dev):")
print(df_dev["label_id"].value_counts())

# df_train['length'] = df_train['sentence'].str.len()
# print(len(df_train[df_train["length"]>256]))
# raise ValueError

tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    input_ids = enc["input_ids"]
    attention = enc["attention_mask"]

    if isinstance(input_ids[0], int):
        return [{"input_ids": input_ids, "attention_mask": attention}]
    else:
        # List of chunks
        return [{"input_ids": ii, "attention_mask": am} for ii, am in zip(input_ids, attention)]


class ChunkedSentenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE):
        self.items = []
        rows = df.itertuples()

        for row in tqdm(rows, desc="Chunking sentences"):
            orig_id = int(row.Index)
            text = row.sentence
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
    """Pad tensors + keep labels and orig_id. This is the custom collator."""
    input_dicts = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
    labels_list = [b["labels"] for b in batch]
    orig_ids_list = [b["orig_id"] for b in batch]
    
    enc = tokenizer.pad(
        input_dicts,
        return_tensors="pt",
    )
    
    enc["labels"] = torch.tensor(labels_list, dtype=torch.long)
    enc["orig_id"] = torch.tensor(orig_ids_list, dtype=torch.long)
    return enc

train_dataset = ChunkedSentenceDataset(df_train, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE)
dev_dataset = ChunkedSentenceDataset(df_dev, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE)

print(f"\nTrain chunks: {len(train_dataset)}")
print(f"Dev chunks:   {len(dev_dataset)}\n")

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "OBJ", 1: "SUBJ"},
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

def compute_chunk_metrics(pred):
    """Compute metrics at the chunk level."""
    preds = np.argmax(pred.predictions, axis=1)
    return {"chunk_accuracy": (preds == pred.label_ids).mean()}

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="chunk_accuracy",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    #data_collator=collate_fn,
    compute_metrics=compute_chunk_metrics,
)

trainer.train()

print("\nMerging LoRA into base model...")

merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nSaved inference-ready model to: {output_dir}")

print("\nRunning sentence-level evaluation by aggregating chunk predictions...")

model = merged_model
device = model.device
model.eval()

loader = DataLoader(dev_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collate_fn)

all_logits = []
all_orig_ids = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Inferencing chunks"):
        # Pop custom keys before moving batch to device
        orig_ids = batch.pop("orig_id").cpu().numpy()
        batch.pop("labels") # Pop labels as they are not needed for inference here
        
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu().numpy()

        all_logits.append(logits)
        all_orig_ids.extend(orig_ids)

all_logits = np.vstack(all_logits)
all_orig_ids = np.array(all_orig_ids)

# Group logits by original sentence ID
doc_chunks = defaultdict(list)
for oid, logit in zip(all_orig_ids, all_logits):
    doc_chunks[int(oid)].append(logit)

# Aggregate: Average the logits for all chunks belonging to the same sentence
doc_preds = {}
for oid, logs in doc_chunks.items():
    avg_logit = np.mean(np.vstack(logs), axis=0)
    # The final prediction is the class with the highest average logit
    doc_preds[int(oid)] = int(np.argmax(avg_logit))

# Build final lists (y_true from original df_dev, y_pred from aggregated results)
y_true = df_dev.loc[list(doc_preds.keys()), "label_id"].tolist()
y_pred = list(doc_preds.values())

inv_label_map = {0: "OBJ", 1: "SUBJ"}
y_true_labels = [inv_label_map[i] for i in y_true]
y_pred_labels = [inv_label_map[i] for i in y_pred]

print("\n=== SENTENCE-LEVEL AGGREGATED REPORT ===\n")
print(classification_report(y_true_labels, y_pred_labels))

# === SENTENCE-LEVEL AGGREGATED REPORT ===

#               precision    recall  f1-score   support

#          OBJ       0.76      0.75      0.75       222
#         SUBJ       0.77      0.78      0.78       240

#     accuracy                           0.77       462
#    macro avg       0.77      0.77      0.77       462
# weighted avg       0.77      0.77      0.77       462