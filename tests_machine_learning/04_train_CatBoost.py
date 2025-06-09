import sys
import os
import numpy as np
import joblib
import logging
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuração de semente para reprodutibilidade
SEED = 18498

# Caminho para importar os dados locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados

# 1. Validação dos dados
required_columns = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO', 'CHURN']
if not all(col in dados.columns for col in required_columns):
    raise ValueError(f"O arquivo de dados deve conter as colunas: {required_columns}")

# 2. Dados e divisão
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
y = dados['CHURN']
cat_cols = X.columns.tolist()

# Tratamento de valores ausentes
X = X.fillna('Desconhecido')  # Corrige o aviso SettingWithCopyWarning

# Converte colunas categóricas para string
X[cat_cols] = X[cat_cols].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Ajuste de class_weights para penalizar mais os erros da classe minoritária
class_weights = [1, len(y_train) / sum(y_train)]

# 3. Treinamento do modelo CatBoost com class_weights
logging.info("Iniciando o treinamento...")
model = CatBoostClassifier(
    iterations=200,
    depth=10,
    learning_rate=0.1,
    l2_leaf_reg=3,
    subsample=0.8,
    random_seed=SEED,
    verbose=100,
    cat_features=cat_cols,
    class_weights=class_weights  # Penaliza mais os erros da classe minoritária
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=30
)
logging.info("Treinamento concluído.")

# 4. Avaliação
y_proba = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 200)
f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

def prever_com_threshold(modelo, dados, threshold):
    proba = modelo.predict_proba(dados)[:, 1]
    return (proba >= threshold).astype(int)

y_pred_otimo = prever_com_threshold(model, X_test, best_threshold)

# 5. Métricas complementares
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)  # Área sob a curva de precisão-recall
mcc = matthews_corrcoef(y_test, y_pred_otimo)  # Matthew Correlation Coefficient

logging.info(f"Melhor threshold pelo F1-score: {best_threshold:.2f} (F1 = {best_f1:.3f})")
logging.info(f"AUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}")
logging.info(f"PR AUC no teste: {pr_auc:.3f}")
logging.info(f"MCC no teste: {mcc:.3f}")

print("\n📊 Relatório final:")
print(classification_report(y_test, y_pred_otimo, target_names=['Não Evadido', 'Evadido']))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_otimo))
print(f"\nPR AUC: {pr_auc:.3f}")
print(f"MCC: {mcc:.3f}")

# 6. Curva F1
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label='F1-score', color='green')
plt.axvline(best_threshold, linestyle='--', color='red', label=f'Threshold ótimo = {best_threshold:.2f}')
plt.title('F1-score por Threshold (CatBoost)')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Curva de Precisão-Recall
plt.figure(figsize=(10, 5))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}', color='blue')
plt.title('Curva de Precisão-Recall (CatBoost)')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 8. Exportar modelo + threshold
output_path = "modelo_churn_catboost.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump({
    "modelo": model,
    "threshold": best_threshold
}, output_path)

logging.info(f"Modelo CatBoost salvo como '{output_path}'")
