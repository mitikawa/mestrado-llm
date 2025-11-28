import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, accuracy_score

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

CHUNK_SIZE = 512
STRIDE = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DATA_DIR = os.path.join(ROOT, "dados")

MODEL_PATH = os.path.join(ROOT, "modelos", "detect_chunked_output", "final_model")

DEV_FILE = os.path.join(DATA_DIR, "detect", "subtaskA_dev_monolingual.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "detect", "subtaskA_test_monolingual.jsonl")


def chunk_text_to_encodings(text, tokenizer, max_length=CHUNK_SIZE, stride=STRIDE):
    """Quebra o texto em chunks com overlap para não perder contexto."""
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

            # Normalização simples do label
            label_str = "human" if row.model == "human" else "AI"
            label = label_map[label_str]

            encs = chunk_text_to_encodings(
                text, tokenizer, max_length=max_len, stride=stride
            )

            for enc in encs:
                self.items.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": label,
                        "orig_id": orig_id,  # Importante para reagrupar depois
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer):
    """Custom collator para lidar com o campo 'orig_id' extra."""
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


def evaluate_split(file_path, split_name, model, tokenizer):
    if not os.path.exists(file_path):
        print(f"[{split_name}] Arquivo não encontrado: {file_path}")
        return

    print(f"\nLoading {split_name} set from: {file_path}")
    df = pd.read_json(file_path, lines=True)

    dataset = ChunkedTextDataset(df, tokenizer, max_len=CHUNK_SIZE, stride=STRIDE)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    print(
        f"Running inference on {len(dataset)} chunks (derived from {len(df)} documents)..."
    )

    model.eval()
    all_logits = []
    all_orig_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            orig_ids = batch.pop("orig_id").cpu().numpy()
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_orig_ids.extend(orig_ids)

    all_logits = np.vstack(all_logits)
    all_orig_ids = np.array(all_orig_ids)

    doc_chunks = defaultdict(list)
    for oid, logit in zip(all_orig_ids, all_logits):
        doc_chunks[int(oid)].append(logit)

    y_pred = []
    y_true = []

    ids_in_df = df.index if "id" not in df.columns else df["id"]

    missing_docs = 0
    for i, row in df.iterrows():
        oid = row["id"] if "id" in row else i

        true_lbl_str = "human" if row["model"] == "human" else "AI"
        y_true.append(0 if true_lbl_str == "human" else 1)

        if oid in doc_chunks:
            avg_logit = np.mean(np.vstack(doc_chunks[oid]), axis=0)
            pred_id = int(np.argmax(avg_logit))
            y_pred.append(pred_id)
        else:
            y_pred.append(0)
            missing_docs += 1

    if missing_docs > 0:
        print(f"Warning: {missing_docs} documents were dropped during processing.")

    target_names = ["human", "AI"]
    print(f"\n=== CLASSIFICATION REPORT ({split_name.upper()}) ===")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"Macro F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"Erro: Não foi possível carregar o modelo em {MODEL_PATH}.")
        print("Verifique se o script de treino salvou o 'final_model' corretamente.")
        exit(1)

    model.to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    evaluate_split(DEV_FILE, "dev", model, tokenizer)

    # === CLASSIFICATION REPORT (DEV) ===
    #               precision    recall  f1-score   support

    #        human     0.8072    0.8188    0.8129      2500
    #           AI     0.8162    0.8044    0.8102      2500

    #     accuracy                         0.8116      5000
    #    macro avg     0.8117    0.8116    0.8116      5000
    # weighted avg     0.8117    0.8116    0.8116      5000

    # Macro F1: 0.8102
    # Accuracy: 0.8116

    evaluate_split(TEST_FILE, "test", model, tokenizer)

# === CLASSIFICATION REPORT (TEST) ===
#               precision    recall  f1-score   support

#        human     0.9536    0.4309    0.5936     16272
#           AI     0.6560    0.9811    0.7863     18000

#     accuracy                         0.7199     34272
#    macro avg     0.8048    0.7060    0.6899     34272
# weighted avg     0.7973    0.7199    0.6948     34272

# Macro F1: 0.7863
# Accuracy: 0.7199
