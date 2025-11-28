import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, accuracy_score

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHUNK_SIZE = 256   
STRIDE = 32        
EVAL_BS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados")
MODEL_DIR = os.path.join(ROOT, "modelos")

MODEL_PATH = os.path.join(MODEL_DIR, "final_subject_model")

DEV_FILE = os.path.join(DATA_DIR, "subj", "dev_en.tsv")
TEST_FILE = os.path.join(DATA_DIR, "subj", "test_en_labeled.tsv")


def chunk_text_to_encodings(text, tokenizer, max_length=CHUNK_SIZE, stride=STRIDE):
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
        return [{"input_ids": ii, "attention_mask": am} for ii, am in zip(input_ids, attention)]

class SubjectivityDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE):
        self.items = []
        
        df = pd.read_csv(file_path, sep="\t")
        
        label_map = {"OBJ": 0, "SUBJ": 1}

        has_label = "label" in df.columns
        
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Loading {os.path.basename(file_path)}"):
            orig_id = getattr(row, "id", row.Index)
            text = row.sentence
            
            label = label_map.get(row.label, -1) if has_label else -1

            encs = chunk_text_to_encodings(text, tokenizer, max_length=max_len, stride=stride)
            
            for enc in encs:
                self.items.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": label,
                    "orig_id": orig_id,
                })
        
        self.original_df = df

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    input_dicts = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
    
    enc = tokenizer.pad(
        input_dicts,
        return_tensors="pt",
    )
    
    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    enc["orig_id"] = torch.tensor([b["orig_id"] for b in batch], dtype=torch.long)
    return enc


def evaluate_split(model, tokenizer, file_path, split_name):
    if not os.path.exists(file_path):
        print(f"[{split_name}] Arquivo não encontrado: {file_path}")
        return

    print(f"\n--- Avaliando: {split_name.upper()} ---")
    
    dataset = SubjectivityDataset(file_path, tokenizer)
    loader = DataLoader(dataset, batch_size=EVAL_BS, collate_fn=collate_fn, shuffle=False)
    
    model.eval()
    
    all_logits = []
    all_orig_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferência"):
            orig_ids = batch.pop("orig_id").cpu().numpy()
            _ = batch.pop("labels")
            
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            
            all_logits.append(logits)
            all_orig_ids.extend(orig_ids)
            
    all_logits = np.vstack(all_logits)
    all_orig_ids = np.array(all_orig_ids)
    
    doc_chunks = defaultdict(list)
    for oid, logit in zip(all_orig_ids, all_logits):
        doc_chunks[int(oid)].append(logit)
        
    y_true = []
    y_pred = []
    
    df = dataset.original_df
    label_map = {"OBJ": 0, "SUBJ": 1}
    inv_map = {0: "OBJ", 1: "SUBJ"}
    
    for row in df.itertuples():
        oid = getattr(row, "id", row.Index)
        
        if "label" in df.columns:
            y_true.append(label_map[row.label])
        
        if oid in doc_chunks:
            avg_logit = np.mean(np.vstack(doc_chunks[oid]), axis=0)
            pred_id = int(np.argmax(avg_logit))
            y_pred.append(pred_id)
        else:
            y_pred.append(0)

    if y_true:
        print(f"\n=== Relatório ({split_name}) ===")
        print(classification_report([inv_map[i] for i in y_true], 
                                    [inv_map[i] for i in y_pred], 
                                    digits=4))
        
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    else:
        print("Dataset sem labels (apenas inferência realizada).")

if __name__ == "__main__":
    print(f"Carregando modelo de: {MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        exit(1)
        
    model.to(DEVICE)
    
    evaluate_split(model, tokenizer, DEV_FILE, "dev")
    
# === Relatório (dev) ===
#               precision    recall  f1-score   support

#          OBJ     0.7615    0.7477    0.7545       222
#         SUBJ     0.7705    0.7833    0.7769       240

#     accuracy                         0.7662       462
#    macro avg     0.7660    0.7655    0.7657       462
# weighted avg     0.7662    0.7662    0.7661       462

# Accuracy: 0.7662
# Macro F1: 0.7657

    evaluate_split(model, tokenizer, TEST_FILE, "test")

# === Relatório (test) ===
#               precision    recall  f1-score   support

#          OBJ     0.9034    0.7395    0.8133       215
#         SUBJ     0.5484    0.8000    0.6507        85

#     accuracy                         0.7567       300
#    macro avg     0.7259    0.7698    0.7320       300
# weighted avg     0.8028    0.7567    0.7672       300

# Accuracy: 0.7567
# Macro F1: 0.7320