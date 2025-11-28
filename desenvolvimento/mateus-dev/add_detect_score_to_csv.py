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
)
CHUNK_SIZE = 512
STRIDE = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados", "detect")
MODEL_PATH = os.path.join(ROOT, "modelos", "detect_chunked_output", "final_model")

INPUT_FILE = os.path.join(DATA_DIR, "subtaskA_test_with_subjectivity.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "subtaskA_test_final_complete.csv")


def chunk_text_to_encodings(text, tokenizer, max_length=CHUNK_SIZE, stride=STRIDE):
    """Quebra o texto em chunks com overlap."""
    enc = tokenizer(
        str(text),
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
        return [
            {"input_ids": ii, "attention_mask": am}
            for ii, am in zip(input_ids, attention)
        ]

class ChunkedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE):
        self.items = []
        label_map = {"human": 0, "AI": 1}

        for row in tqdm(df.itertuples(), total=len(df), desc="Tokenizing & Chunking"):
            orig_id = getattr(row, "id", row.Index)
            text = row.text

            try:
                label_str = "human" if row.model == "human" else "AI"
                label = label_map[label_str]
            except:
                label = 0

            encs = chunk_text_to_encodings(
                text, tokenizer, max_length=max_len, stride=stride
            )

            for enc in encs:
                self.items.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": label,
                        "orig_id": orig_id,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch, tokenizer):
    """Pad tensors + keep labels and orig_id."""
    enc = tokenizer.pad(
        [
            {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
            for b in batch
        ],
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    enc["orig_id"] = torch.tensor([b["orig_id"] for b in batch], dtype=torch.long)
    return enc


def generate_scores(input_path, output_path, model, tokenizer):
    if not os.path.exists(input_path):
        print(f"Arquivo não encontrado: {input_path}")
        return

    print(f"\nCarregando dados de: {input_path}")
    df = pd.read_csv(input_path)

    dataset = ChunkedTextDataset(df, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    print(f"Iniciando inferência em {len(dataset)} chunks...")

    model.eval()
    all_logits = []
    all_orig_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processando Batches"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            orig_ids = batch.pop("orig_id").cpu().numpy()
            _ = batch.pop("labels")
            
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_orig_ids.extend(orig_ids)

    all_logits = np.vstack(all_logits)
    all_orig_ids = np.array(all_orig_ids)

    doc_chunks = defaultdict(list)
    for oid, logit in zip(all_orig_ids, all_logits):
        doc_chunks[int(oid)].append(logit)
    detect_scores = []
    
    print("Calculando probabilidades finais...")
    
    missing_docs = 0
    for i, row in df.iterrows():
        oid = row["id"] if "id" in row else i
        
        if oid in doc_chunks:
            avg_logit = np.mean(np.vstack(doc_chunks[oid]), axis=0)
            
            probs = torch.softmax(torch.tensor(avg_logit), dim=-1).numpy()
            
            prob_ai = probs[1]
            detect_scores.append(prob_ai)
        else:
            detect_scores.append(0.0) 
            missing_docs += 1

    if missing_docs > 0:
        print(f"Aviso: {missing_docs} documentos não geraram chunks (texto vazio?).")

    df['detect_score'] = detect_scores
    
    print(f"Salvando arquivo final em: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("\n=== AMOSTRA DO DATASET FINAL ===")
    cols_to_show = [c for c in ['id', 'model', 'subjectivity_index', 'detect_score'] if c in df.columns]
    print(df[cols_to_show].head())


if __name__ == "__main__":
    print(f"Carregando modelo de: {MODEL_PATH}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"Erro ao carregar modelo em {MODEL_PATH}")
        exit(1)

    model.to(DEVICE)
    print(f"Modelo carregado em {DEVICE}")

    generate_scores(INPUT_FILE, OUTPUT_FILE, model, tokenizer)