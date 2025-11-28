import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "bert-base-uncased"
CHUNK_SIZE = 256
STRIDE = 32
SEED = 42

NUM_EPOCHS = 5 
BATCH_SIZE = 16
LEARNING_RATE = 2e-4

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados")
OUTPUT_DIR = os.path.join(ROOT, "modelos", "complete_subj_model")

DATASET_CONFIG = [
    {
        "name": "Subj_Train",
        "path": os.path.join(DATA_DIR, "subj", "train_en.tsv"),
        "format": "tsv",
        "text_col": "sentence",
        "label_col": "label",
        "map_labels": {"OBJ": 0, "SUBJ": 1}
    },
    {
        "name": "Subj_Dev", # Agora o Dev entra no Treino!
        "path": os.path.join(DATA_DIR, "subj", "dev_en.tsv"),
        "format": "tsv",
        "text_col": "sentence",
        "label_col": "label",
        "map_labels": {"OBJ": 0, "SUBJ": 1}
    },
    {
        "name": "Subj_Test_Labeled",
        "path": os.path.join(DATA_DIR, "subj", "test_en_labeled.tsv"),
        "format": "tsv",
        "text_col": "sentence",
        "label_col": "label",
        "map_labels": {"OBJ": 0, "SUBJ": 1}
    }
]


def load_and_merge_datasets(configs):
    merged_list = []
    print("--- Loading ALL available data for Final Training ---")
    
    for cfg in configs:
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        print(f"Loading {cfg['name']}...")
        
        if cfg["format"] == "tsv":
            df = pd.read_csv(path, sep="\t")
        elif cfg["format"] == "csv":
            df = pd.read_csv(path)
        elif cfg["format"] == "json":
            df = pd.read_json(path, lines=True)
            
        temp_df = pd.DataFrame()
        temp_df["text"] = df[cfg["text_col"]]
        
        mapper = cfg["map_labels"]
        temp_df["label_id"] = df[cfg["label_col"]].apply(
            lambda x: mapper.get(str(x), mapper.get(int(x) if str(x).isdigit() else x))
        )
        
        temp_df = temp_df.dropna(subset=["label_id"])
        temp_df["label_id"] = temp_df["label_id"].astype(int)
        
        merged_list.append(temp_df)

    if not merged_list:
        raise ValueError("No datasets loaded.")
        
    final_df = pd.concat(merged_list, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    return final_df


df_full = load_and_merge_datasets(DATASET_CONFIG)
print(f"\nTOTAL DATASET SIZE: {len(df_full)} samples")
print(df_full["label_id"].value_counts())


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_text_to_encodings(text, tokenizer):
    enc = tokenizer(
        str(text),
        truncation=True,
        max_length=CHUNK_SIZE,
        return_overflowing_tokens=True,
        stride=STRIDE,
        return_attention_mask=True,
    )
    if isinstance(enc["input_ids"][0], int):
        return [{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}]
    else:
        return [{"input_ids": ii, "attention_mask": am} for ii, am in zip(enc["input_ids"], enc["attention_mask"])]

class FinalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.items = []
        for row in tqdm(df.itertuples(), total=len(df), desc="Chunking All Data"):
            chunks = chunk_text_to_encodings(row.text, tokenizer)
            for c in chunks:
                self.items.append({
                    "input_ids": c["input_ids"],
                    "attention_mask": c["attention_mask"],
                    "labels": row.label_id
                })
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    input_dicts = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
    enc = tokenizer.pad(input_dicts, return_tensors="pt")
    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return enc

full_dataset = FinalDataset(df_full, tokenizer)


base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "OBJ", 1: "SUBJ"},
    label2id={"OBJ": 0, "SUBJ": 1},
)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    eval_strategy="no", 
    learning_rate=LEARNING_RATE,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=full_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

print("\n>>> STARTING Final TRAINING (ALL DATA, NO EVAL) <<<")
trainer.train()

# === SAVING FINAL MODEL ===

print("\nMerging and Saving Final Final Model...")
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nSUCCESS! Final model saved to:\n{OUTPUT_DIR}")