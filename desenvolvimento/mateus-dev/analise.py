import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados", "detect")
INPUT_FILE = os.path.join(DATA_DIR, "subtaskA_test_final_complete.csv")
OUTPUT_DIR = os.path.join(ROOT, "resultados_analise")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {0: "#1f77b4", 1: "#d62728"} # Azul (Humano), Vermelho (AI)

def main():
    print(f"--- Carregando dados: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print("Erro: Arquivo não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    df['true_label'] = df['model'].apply(lambda x: 0 if x == 'human' else 1)
    
    df['pred_label'] = (df['detect_score'] > 0.5).astype(int)
    
    def get_outcome(row):
        if row['true_label'] == 1 and row['pred_label'] == 1: return "TP (AI Detectada)"
        if row['true_label'] == 0 and row['pred_label'] == 0: return "TN (Humano Correto)"
        if row['true_label'] == 0 and row['pred_label'] == 1: return "FP (Humano -> AI)"
        if row['true_label'] == 1 and row['pred_label'] == 0: return "FN (AI -> Humano)"
    
    df['outcome'] = df.apply(get_outcome, axis=1)

    print(f"Total de amostras: {len(df)}")
    print("-" * 40)

    print("\n[1] ANÁLISE DE GERAÇÃO (Subjetividade Intrínseca)")
    
    subj_human = df[df['true_label'] == 0]['subjectivity_index']
    subj_ai = df[df['true_label'] == 1]['subjectivity_index']
    
    print(f"Média Subjetividade (Humano): {subj_human.mean():.4f} (std: {subj_human.std():.4f})")
    print(f"Média Subjetividade (AI):     {subj_ai.mean():.4f} (std: {subj_ai.std():.4f})")
    
    t_stat, p_val = stats.ttest_ind(subj_human, subj_ai)
    print(f"Teste T (Diferença de Médias): p-value = {p_val:.4e}")
    if p_val < 0.05:
        dir_ = "MAIS" if subj_ai.mean() > subj_human.mean() else "MENOS"
        print(f">> CONCLUSÃO ESTATÍSTICA: IAs são significativamente {dir_} subjetivas que humanos.")
    else:
        print(">> CONCLUSÃO ESTATÍSTICA: Não há diferença significativa de subjetividade.")

    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=df, x="subjectivity_index", hue="true_label", fill=True, 
                palette=PALETTE, common_norm=False, alpha=0.5)
    plt.title("Distribuição de Subjetividade: Humano vs AI")
    plt.xlabel("Índice de Subjetividade (0=Obj, 1=Subj)")
    plt.ylabel("Densidade")
    plt.legend(title="Origem", labels=["AI", "Humano"])
    plt.savefig(os.path.join(OUTPUT_DIR, "1_distribuicao_subjetividade.png"))
    plt.close()

    print("\n[2] ANÁLISE DE CORRELAÇÃO (Detector Bias)")
    
    corr_pearson = df['subjectivity_index'].corr(df['detect_score'], method='pearson')
    print(f"Correlação de Pearson (Subjetividade vs Score Detecção): {corr_pearson:.4f}")

    plt.figure(figsize=(12, 7))
    
    sns.scatterplot(data=df, x="subjectivity_index", y="detect_score", 
                    alpha=0.05, color="grey", s=10, linewidth=0)

    sns.regplot(data=df, x="subjectivity_index", y="detect_score", 
                x_bins=15, # Divide subjetividade em 15 faixas e plota a média
                scatter_kws={'s': 60, 'color': '#1f77b4', 'edgecolor':'w'}, # Pontos médios azuis grandes
                line_kws={'color':'red', 'linewidth': 3}, # Linha de tendência vermelha grossa
                ci=95 # Mostra intervalo de confiança de 95% nas médias
               )

    plt.title(f"Tendência Média: Subjetividade vs Confiança do Detector (R={corr_pearson:.2f})", fontsize=16)
    plt.xlabel("Índice de Subjetividade (Agrupado em Bins)", fontsize=14)
    plt.ylabel("Média do Score do Detector (Prob. AI)", fontsize=14)
    plt.ylim(-0.05, 1.05) 
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    plt.annotate('Modelo muito confiante\nem textos objetivos', xy=(0.05, 0.95), xytext=(0.2, 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_correlacao_binned.png"))
    plt.close()

    print("\n[3] ANÁLISE DE SENSIBILIDADE (Performance por Faixa)")
    
    LABELS_ORDER = ["Muito Objetivo", "Objetivo", "Subjetivo", "Muito Subjetivo"]
    
    df['subj_bin'] = pd.qcut(df['subjectivity_index'], q=4, labels=LABELS_ORDER)
    
    metrics = []
    for bin_name in LABELS_ORDER:
        subset = df[df['subj_bin'] == bin_name]
        if len(subset) > 0:
            acc = accuracy_score(subset['true_label'], subset['pred_label'])
            f1 = f1_score(subset['true_label'], subset['pred_label'])
            metrics.append({"Faixa": bin_name, "Acurácia": acc, "F1-Score": f1, "N": len(subset)})
    
    df_metrics = pd.DataFrame(metrics)
    print(df_metrics)
    
    plt.figure(figsize=(10, 6))
    df_melted = df_metrics.melt(id_vars="Faixa", value_vars=["Acurácia", "F1-Score"], var_name="Métrica")
    
    sns.barplot(data=df_melted, x="Faixa", y="value", hue="Métrica", palette="viridis", order=LABELS_ORDER)
    
    plt.title("Performance do Detector por Nível de Subjetividade")
    plt.xlabel("Nível de Subjetividade")
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(OUTPUT_DIR, "3_performance_por_faixa.png"))
    plt.close()

    print("\n[4] ANÁLISE DE ERROS (FP/FN)")
    
    plt.figure(figsize=(10, 8)) 
    
    order_y = [
        "TN (Humano Correto)", 
        "FP (Humano -> AI)", 
        "FN (AI -> Humano)", 
        "TP (AI Detectada)"
    ]

    order_y.reverse()

    sns.boxplot(data=df, x="subjectivity_index", y="outcome", order=order_y, palette="Set2", orient="h")
    
    plt.title("Distribuição de Subjetividade por Tipo de Erro/Acerto")
    plt.xlabel("Índice de Subjetividade (0=Obj, 1=Subj)") 
    plt.ylabel("Resultado da Detecção")   
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_analise_erros_boxplot.png"))
    plt.close()
    
    mean_fp = df[df['outcome']=="FP (Humano -> AI)"]['subjectivity_index'].mean()
    mean_tn = df[df['outcome']=="TN (Humano Correto)"]['subjectivity_index'].mean()
    
    print(f"Subjetividade média dos Humanos classificados CORRETAMENTE: {mean_tn:.4f}")
    print(f"Subjetividade média dos Humanos classificados ERRADAMENTE (como AI): {mean_fp:.4f}")
    
    if mean_fp < mean_tn:
        print(">> INSIGHT: Humanos que escrevem de forma mais OBJETIVA tendem a ser confundidos com IA.")
    else:
        print(">> INSIGHT: A confusão em humanos não parece ligada à objetividade excessiva.")

    print(f"\nTodos os gráficos foram salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()