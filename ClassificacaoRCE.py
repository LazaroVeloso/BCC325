"""
Classificação do Padrão de Recuperação Cardíaca Pós-Esforço — Random Forest
Dataset: Cleveland Heart Disease (UCI)   |   

RÓTULO  → criado com 4 variáveis clínicas por sistema de pontuação ponderada
FEATURES → todas as 13 variáveis clínicas disponíveis no dataset

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    "processed.cleveland.data",
    header=None, names=COLS, na_values="?"
)

print(f"Dataset carregado: {len(df)} pacientes, {df.shape[1]} colunas")
print(f"Valores ausentes:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

# Remove linhas com qualquer valor ausente
df_clean = df.dropna().copy()
print(f"Pacientes após remoção de NaNs: {len(df_clean)}\n")

# ─────────────────────────────────────────────
# 2. RÓTULO — SCORE CLÍNICO PONDERADO
# ─────────────────────────────────────────────
# Score máximo possível = 9 pontos
#
#  oldpeak (indicador mais forte — vale até 3 pts)
#    < 1.0  → +3   alteração ST mínima
#    < 2.0  → +1   alteração ST moderada
#    >= 2.0 → +0   alteração ST preocupante
#
#  exang (angina durante exercício — vale até 2 pts)
#    0      → +2   sem angina
#    1      → +0   com angina
#
#  slope (forma da curva ST — vale até 2 pts)
#    1      → +2   ascendente (bom)
#    2      → +1   plana (neutro)
#    3      → +0   descendente (ruim)
#
#  thalach (frequência cardíaca máxima — vale até 2 pts)
#    >= 160 → +2   excelente capacidade
#    >= 140 → +1   razoável
#    < 140  → +0   baixa capacidade
#
#  Classificação:
#    score 7-9  → BOA
#    score 4-6  → MODERADA
#    score 0-3  → RUIM

def classificar_recuperacao(row):
    score = 0

    # oldpeak — peso maior (max 3 pts)
    if row["oldpeak"] < 1.0:
        score += 3
    elif row["oldpeak"] < 2.0:
        score += 1

    # exang (max 2 pts)
    if row["exang"] == 0:
        score += 2

    # slope (max 2 pts)
    if row["slope"] == 1:
        score += 2
    elif row["slope"] == 2:
        score += 1

    # thalach (max 2 pts)
    if row["thalach"] >= 160:
        score += 2
    elif row["thalach"] >= 140:
        score += 1

    if score >= 7:
        return "BOA"
    elif score >= 4:
        return "MODERADA"
    else:
        return "RUIM"

df_clean["recuperacao"] = df_clean.apply(classificar_recuperacao, axis=1)

# Distribuição com percentual
print("Distribuição dos rótulos (score ponderado):")
counts = df_clean["recuperacao"].value_counts()
total  = len(df_clean)
for classe, n in counts.items():
    print(f"  {classe:<10} {n:>3} pacientes  ({n/total*100:.1f}%)")
print()

# ─────────────────────────────────────────────
# 3. FEATURES — TODAS AS 13 VARIÁVEIS CLÍNICAS
# ─────────────────────────────────────────────
# O rótulo foi criado com thalach/exang/oldpeak/slope.
# O RF treina com as 13 colunas restantes.
# Isso permite descobrir relações não codificadas na regra de pontuação.

all_features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

X = df_clean[all_features].values
y = df_clean["recuperacao"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

# ─────────────────────────────────────────────
# 4. TREINAMENTO
# ─────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ─────────────────────────────────────────────
# 5. MÉTRICAS
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")

print("=" * 55)
print("MÉTRICAS DE AVALIAÇÃO")
print("=" * 55)
print(f"Acurácia no conjunto de teste : {accuracy_score(y_test, y_pred):.4f}")
print(f"Acurácia (CV 10-fold) – média : {cv_scores.mean():.4f}")
print(f"Acurácia (CV 10-fold) – desvio: {cv_scores.std():.4f}")
print()
print("Relatório de Classificação (teste):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 6. VISUALIZAÇÕES
# ─────────────────────────────────────────────
feat_imp    = rf.feature_importances_
sorted_idx  = np.argsort(feat_imp)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "Random Forest – Recuperação Cardíaca Pós-Esforço (v3)\n"
    "Score Ponderado | 13 Features | Dataset Cleveland (UCI)",
    fontsize=13, fontweight="bold"
)

# ── 6a. Distribuição dos rótulos ─────────────
ax = axes[0, 0]
order  = ["BOA", "MODERADA", "RUIM"]
counts_ordered = [counts.get(c, 0) for c in order]
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bars = ax.bar(order, counts_ordered, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("Distribuição das Classes de Recuperação")
ax.set_xlabel("Classe")
ax.set_ylabel("Número de Pacientes")
for bar, val in zip(bars, counts_ordered):
    pct = val / total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=9)

# ── 6b. Importância das 13 features ──────────
ax = axes[0, 1]
palette = plt.cm.tab20.colors
ax.barh(
    [all_features[i] for i in sorted_idx],
    feat_imp[sorted_idx],
    color=[palette[i % len(palette)] for i in sorted_idx],
    edgecolor="white"
)
ax.set_title("Importância das Features (Gini) — 13 variáveis")
ax.set_xlabel("Importância Média")
for i, idx in enumerate(sorted_idx):
    ax.text(feat_imp[idx] + 0.002, i,
            f"{feat_imp[idx]:.3f}", va="center", fontsize=8)

# ── 6c. Matriz de confusão ───────────────────
ax = axes[1, 0]
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matriz de Confusão (Conjunto de Teste)")

# ── 6d. Acurácia por fold ────────────────────
ax = axes[1, 1]
fold_nums  = np.arange(1, len(cv_scores) + 1)
bar_colors = ["#2ecc71" if s >= cv_scores.mean() else "#e74c3c" for s in cv_scores]
ax.bar(fold_nums, cv_scores, color=bar_colors, edgecolor="white", linewidth=1.2)
ax.axhline(cv_scores.mean(), color="navy", linestyle="--", linewidth=1.5,
           label=f"Média = {cv_scores.mean():.3f}")
ax.set_title("Acurácia por Fold – Validação Cruzada (10-fold)")
ax.set_xlabel("Fold")
ax.set_ylabel("Acurácia")
ax.set_ylim(0, 1.05)
ax.legend()
for i, val in enumerate(cv_scores):
    ax.text(fold_nums[i], val + 0.01, f"{val:.2f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("resultados_rf_v3.png", dpi=150, bbox_inches="tight")
print("\nGráfico salvo em: resultados_rf_v3.png")

# ─────────────────────────────────────────────
# 7. TOP 5 FEATURES MAIS IMPORTANTES
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("TOP 5 FEATURES MAIS IMPORTANTES")
print("=" * 55)
top5 = np.argsort(feat_imp)[::-1][:5]
for rank, idx in enumerate(top5, 1):
    print(f"  {rank}. {all_features[idx]:<10} {feat_imp[idx]:.4f}")

# ─────────────────────────────────────────────
# 8. EXEMPLOS DE PREDIÇÃO INDIVIDUAL
# ─────────────────────────────────────────────
# Ordem: age, sex, cp, trestbps, chol, fbs, restecg,
#        thalach, exang, oldpeak, slope, ca, thal
print("\n" + "=" * 55)
print("EXEMPLOS DE PREDIÇÃO INDIVIDUAL")
print("=" * 55)

exemplos = {
    "Paciente A – perfil excelente (esperado BOA)":
        [45, 0, 2, 120, 200, 0, 0, 170, 0, 0.3, 1, 0, 3],
    "Paciente B – perfil intermediário (esperado MODERADA)":
        [55, 1, 3, 140, 250, 0, 1, 148, 0, 1.5, 2, 1, 3],
    "Paciente C – perfil grave (esperado RUIM)":
        [63, 1, 4, 160, 280, 1, 2, 112, 1, 3.1, 3, 3, 7],
    "Paciente D – caso ambíguo":
        [58, 1, 2, 130, 240, 0, 0, 138, 0, 1.8, 2, 1, 3],
}

for nome, vals in exemplos.items():
    # score manual (apenas as 4 variáveis do rótulo)
    thalach = vals[7]; exang = vals[8]
    oldpeak = vals[9]; slope = vals[10]
    score = (
        (3 if oldpeak < 1.0 else 1 if oldpeak < 2.0 else 0) +
        (2 if exang == 0 else 0) +
        (2 if slope == 1 else 1 if slope == 2 else 0) +
        (2 if thalach >= 160 else 1 if thalach >= 140 else 0)
    )
    pred_enc = rf.predict([vals])[0]
    proba    = rf.predict_proba([vals])[0]
    classe   = le.inverse_transform([pred_enc])[0]
    proba_str = " | ".join(f"{c}: {p:.2f}" for c, p in zip(le.classes_, proba))
    print(f"{nome}")
    print(f"  Score clínico : {score}/9")
    print(f"  Predição RF   : {classe}")
    print(f"  Probabilidades: {proba_str}\n")

print("Script finalizado com sucesso.")