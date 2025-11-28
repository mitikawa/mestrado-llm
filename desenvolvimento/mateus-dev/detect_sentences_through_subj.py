import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BATCH_SIZE = 64 
MAX_LEN = 128 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DATA_DIR = os.path.join(ROOT, "dados", "detect")
MODEL_PATH = os.path.join(ROOT, "modelos", "complete_subj_model")

SENTENCES_FILE = os.path.join(DATA_DIR, "subtaskA_test_monolingual_sentences.jsonl")
DOCS_FILE = os.path.join(DATA_DIR, "subtaskA_test_monolingual.jsonl")

OUTPUT_CSV = os.path.join(DATA_DIR, "subtaskA_test_with_subjectivity.csv")



class SentenceInferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.ids = df["sentence_id"].tolist()
        self.doc_ids = df["doc_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",  
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "doc_id": self.doc_ids[idx],
        }


def main():
    print(f"--- Carregando modelo de subjetividade: {MODEL_PATH} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    model.to(DEVICE)
    model.eval()

    print(f"Carregando frases de: {SENTENCES_FILE}")
    df_sentences = pd.read_json(SENTENCES_FILE, lines=True)
    print(f"Total de frases: {len(df_sentences)}")

    dataset = SentenceInferenceDataset(df_sentences, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Iniciando inferência de subjetividade...")
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processando Batches"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1)

            subj_probs = probs[:, 1].cpu().numpy()

            all_probs.extend(subj_probs)

    df_sentences["subj_score"] = all_probs

    print("Agregando scores por doc_id...")
    doc_scores = df_sentences.groupby("doc_id")["subj_score"].mean().reset_index()
    doc_scores.rename(columns={"subj_score": "subjectivity_index"}, inplace=True)

    print(f"Carregando documentos originais de: {DOCS_FILE}")
    df_docs = pd.read_json(DOCS_FILE, lines=True)

    print("Mesclando dados...")
    df_final = pd.merge(
        df_docs, doc_scores, left_on="id", right_on="doc_id", how="left"
    )

    if "doc_id" in df_final.columns and "id" in df_final.columns:
        df_final.drop(columns=["doc_id"], inplace=True)

    missing = df_final["subjectivity_index"].isna().sum()
    if missing > 0:
        print(
            f"Aviso: {missing} documentos não tiveram frases correspondentes e ficaram com NaN."
        )
        df_final["subjectivity_index"] = df_final["subjectivity_index"].fillna(
            0.0
        )

    print(f"Salvando resultado final em: {OUTPUT_CSV}")
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("\n--- Amostra do Dataset Final ---")
    print(df_final[["id", "model", "subjectivity_index"]].head())
    print("-" * 30)


if __name__ == "__main__":
    main()
