import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

# === CONFIGURAÇÃO GLOBAL ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dados", "detect")
INPUT_FILE = os.path.join(DATA_DIR, "subtaskA_test_final_complete.csv")
OUTPUT_DIR = os.path.join(ROOT, "resultados_analise_pub_final") # Diretório final

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ESTILO VISUAL "MAX READABILITY" ===
# font_scale=2.2 garante que tudo fique enorme por padrão
sns.set_theme(style="whitegrid", context="paper", font_scale=2.2)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 2.0 # Eixos mais grossos
mpl.rcParams['grid.linewidth'] = 1.0 

PALETTE = {0: "#005b96", 1: "#b30000"} 
SAVE_KWARGS = {'dpi': 300, 'bbox_inches': 'tight'}

def main():
    print(f"--- Carregando dados: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print("Erro: Arquivo não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Pré-processamento
    df['true_label'] = df['model'].apply(lambda x: 0 if x == 'human' else 1)
    df['pred_label'] = (df['detect_score'] > 0.5).astype(int)
    
    def get_outcome(row):
        if row['true_label'] == 1 and row['pred_label'] == 1: return "TP (IA Correta)"
        if row['true_label'] == 0 and row['pred_label'] == 0: return "TN (Humano Correto)"
        if row['true_label'] == 0 and row['pred_label'] == 1: return "FP (Humano → IA)"
        if row['true_label'] == 1 and row['pred_label'] == 0: return "FN (IA → Humano)"
    
    df['outcome'] = df.apply(get_outcome, axis=1)

    print(f"Total de amostras: {len(df)}")
    print("-" * 40)

    # === 1. ANÁLISE DE VIÉS DE GERAÇÃO (Distribuição) ===
    print("\n[1] Análise de Geração")
    
    # Tamanho aumentado para acomodar texto grande
    plt.figure(figsize=(11, 7)) 
    
    sns.kdeplot(data=df, x="subjectivity_index", hue="true_label", fill=True, 
                palette=PALETTE, common_norm=False, alpha=0.4, linewidth=4)
    
    # Títulos e Labels Gigantes
    # plt.title("Distribuição de Subjetividade por Origem", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Índice de Subjetividade", fontsize=22) # Simplificado para caber melhor
    plt.ylabel("Densidade", fontsize=22)
    plt.xlim(0, 1)
    
    # Ticks Gigantes (O pedido principal)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Legenda Interna, Canto Direito, Fonte Grande
    plt.legend(title="", labels=["IA", "Humano"], 
               loc='upper right', frameon=True, fontsize=20, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Distribuicao_Subjetividade.png"), **SAVE_KWARGS)
    plt.close()

    # === 2. ANÁLISE DE VIÉS DO DETECTOR (Correlação) ===
    print("\n[2] Análise de Correlação")
    
    corr_pearson = df['subjectivity_index'].corr(df['detect_score'], method='pearson')
    
    plt.figure(figsize=(11, 7))
    
    sns.scatterplot(data=df, x="subjectivity_index", y="detect_score", 
                    alpha=0.03, color="grey", s=20, linewidth=0, rasterized=True)

    sns.regplot(data=df, x="subjectivity_index", y="detect_score", 
                x_bins=15,
                scatter_kws={'s': 150, 'color': PALETTE[0], 'edgecolor':'k', 'linewidth': 1.5, 'alpha':1},
                line_kws={'color': PALETTE[1], 'linewidth': 5},
                ci=95, logistic=True)

    # plt.title(f"Confiança vs. Subjetividade (R = {corr_pearson:.2f})", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Índice de Subjetividade (Agrupado)", fontsize=22)
    plt.ylabel("Probabilidade IA", fontsize=22)
    plt.ylim(-0.05, 1.05)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # textstr = 'Alta confiança\nem textos objetivos'
    # props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', pad=0.5)
    # plt.annotate(textstr, xy=(0.05, 0.96), xytext=(0.20, 0.85),
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black', lw=3),
    #              bbox=props, fontsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Correlacao_Binned.png"), **SAVE_KWARGS)
    plt.close()

    # === 3. ANÁLISE DE SENSIBILIDADE (Desempenho por Faixa) ===
    print("\n[3] Análise de Sensibilidade")
    
    LABELS_ORDER = ["Muito Obj.", "Objetivo", "Subjetivo", "Muito Subj."] # Abreviei levemente para caber fonte 20
    
    # Precisamos refazer o binning com os labels abreviados
    # Note: remapeando a lógica para garantir consistência
    df['subj_bin_temp'] = pd.qcut(df['subjectivity_index'], q=4) # bins numéricos
    # Criar mapeamento manual para garantir ordem
    metrics = []
    
    # Mapear os intervalos originais para os nomes novos
    # A maneira mais segura é iterar nos bins numéricos ordenados
    unique_bins = sorted(df['subj_bin_temp'].unique())
    
    metrics_data = []
    for i, bin_interval in enumerate(unique_bins):
        label_name = LABELS_ORDER[i]
        subset = df[df['subj_bin_temp'] == bin_interval]
        acc = accuracy_score(subset['true_label'], subset['pred_label'])
        f1 = f1_score(subset['true_label'], subset['pred_label'])
        metrics_data.append({"Faixa": label_name, "Acurácia": acc, "F1-Score": f1})
    
    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(11, 7))
    df_melted = df_metrics.melt(id_vars="Faixa", value_vars=["Acurácia", "F1-Score"], var_name="Métrica")
    
    sns.barplot(data=df_melted, x="Faixa", y="value", hue="Métrica", palette="viridis", 
                edgecolor='k', linewidth=1.5)
    
    # plt.title("Desempenho por Quartil", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Nível de Subjetividade", fontsize=22)
    plt.ylabel("Métrica", fontsize=22)
    plt.ylim(0.0, 1.15) # Mais espaço no topo para a legenda
    
    plt.tick_params(axis='x', which='major', labelsize=19) # Fonte X ligeiramente menor para não sobrepor
    plt.tick_params(axis='y', which='major', labelsize=18)
    
    plt.legend(title="", loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), 
               frameon=False, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Desempenho_por_Faixa.png"), **SAVE_KWARGS)
    plt.close()

    # === 4. ANÁLISE DE ERRO (Boxplot) ===
    print("\n[4] Análise de Erro")
    
    plt.figure(figsize=(12, 8)) # Mais alto para caber as labels Y
    
    order_y = ["TN (Humano Correto)", "FP (Humano → IA)", "FN (IA → Humano)", "TP (IA Correta)"]
    order_y.reverse()
    
    sns.boxplot(data=df, x="subjectivity_index", y="outcome", order=order_y, 
                palette="Set2", orient="h", linewidth=2.5, fliersize=5)
    
    # plt.title("Subjetividade por Resultado", fontweight='bold', fontsize=24, pad=20)
    plt.xlabel("Índice de Subjetividade", fontsize=22)
    plt.ylabel("")
    plt.xlim(0, 1)
    
    plt.grid(axis='x', linestyle='--', linewidth=1, alpha=0.7)
    
    # Ticks Y (Labels das categorias) precisam ser muito legíveis
    plt.tick_params(axis='y', which='major', labelsize=20) 
    plt.tick_params(axis='x', which='major', labelsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_Analise_Erro_Boxplot.png"), **SAVE_KWARGS)
    plt.close()
    
    print(f"\nGráficos gigantes salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()