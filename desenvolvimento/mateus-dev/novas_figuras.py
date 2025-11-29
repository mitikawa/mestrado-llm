import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import accuracy_score

# === CONFIGURAÇÃO GLOBAL ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados", "detect")
INPUT_FILE = os.path.join(DATA_DIR, "subtaskA_test_final_complete.csv")
OUTPUT_DIR = os.path.join(ROOT, "resultados_analise_pub_final")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo
sns.set_theme(style="whitegrid", context="paper", font_scale=2.2)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['grid.linewidth'] = 1.0

# Cores específicas
COLOR_AI = "#b30000"    # Vermelho
COLOR_HUMAN = "#005b96" # Azul
SAVE_KWARGS = {'dpi': 300, 'bbox_inches': 'tight'}

def main():
    print(f"--- Carregando dados: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print("Erro: Arquivo não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Pre-processamento
    df['true_label'] = df['model'].apply(lambda x: 0 if x == 'human' else 1)
    df['pred_label'] = (df['detect_score'] > 0.5).astype(int)
    
    # Recriar os Bins
    LABELS_ORDER = ["Muito Obj.", "Objetivo", "Subjetivo", "Muito Subj."]
    df['subj_bin_temp'] = pd.qcut(df['subjectivity_index'], q=4)
    unique_bins = sorted(df['subj_bin_temp'].unique())
    bin_mapping = {interval: label for interval, label in zip(unique_bins, LABELS_ORDER)}
    df['subj_bin_label'] = df['subj_bin_temp'].map(bin_mapping)

    print("-" * 40)

    # === FIGURA 5: PERFORMANCE EM IA (Taxa de Acerto) ===
    print("\n[5] Gerando Figura 5: Taxa de Acerto em IA")
    
    metrics_ai = []
    df_ai = df[df['true_label'] == 1]
    
    for bin_name in LABELS_ORDER:
        subset = df_ai[df_ai['subj_bin_label'] == bin_name]
        if len(subset) > 0:
            # Em subset de classe única, Acurácia = Recall
            acc = accuracy_score(subset['true_label'], subset['pred_label'])
            metrics_ai.append({"Faixa": bin_name, "Taxa de Acerto": acc})
    
    df_metrics_ai = pd.DataFrame(metrics_ai)
    
    plt.figure(figsize=(10, 7))
    # Barplot simples (sem hue)
    sns.barplot(data=df_metrics_ai, x="Faixa", y="Taxa de Acerto", 
                color=COLOR_AI, edgecolor='k', linewidth=1.5)
    
    # plt.title("Detecção de IA por Subjetividade", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Nível de Subjetividade", fontsize=22)
    plt.ylabel("Taxa de Detecção (Recall)", fontsize=22)
    plt.ylim(0.0, 1.1) # Margem superior para o texto
    
    # Adicionar os valores percentuais em cima
    for index, row in df_metrics_ai.iterrows():
        plt.text(index, row['Taxa de Acerto'] + 0.02, f"{row['Taxa de Acerto']*100:.1f}%", 
                 color='black', ha="center", fontsize=20, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig5_Performance_IA_Recall.png"), **SAVE_KWARGS)
    plt.close()


    # === FIGURA 6: PERFORMANCE EM HUMANOS (Taxa de Acerto) ===
    print("\n[6] Gerando Figura 6: Taxa de Acerto em Humanos")
    
    metrics_human = []
    df_human = df[df['true_label'] == 0]
    
    for bin_name in LABELS_ORDER:
        subset = df_human[df_human['subj_bin_label'] == bin_name]
        if len(subset) > 0:
            # Em subset de classe única, Acurácia = Especificidade (Taxa de Acerto Humano)
            acc = accuracy_score(subset['true_label'], subset['pred_label'])
            metrics_human.append({"Faixa": bin_name, "Taxa de Acerto": acc})
    
    df_metrics_human = pd.DataFrame(metrics_human)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_metrics_human, x="Faixa", y="Taxa de Acerto", 
                color=COLOR_HUMAN, edgecolor='k', linewidth=1.5)
    
    # plt.title("Detecção de Humano por Subjetividade", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Nível de Subjetividade", fontsize=22)
    plt.ylabel("Taxa de Acerto", fontsize=22)
    plt.ylim(0.0, 1.1)

    for index, row in df_metrics_human.iterrows():
        plt.text(index, row['Taxa de Acerto'] + 0.02, f"{row['Taxa de Acerto']*100:.1f}%", 
                 color='black', ha="center", fontsize=20, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Performance_Humano_Recall.png"), **SAVE_KWARGS)
    plt.close()
    
    print(f"\nFiguras ajustadas salvas em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()